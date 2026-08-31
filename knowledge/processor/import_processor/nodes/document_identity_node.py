"""生成可追溯的文档档案，并写入独立文档注册表。"""

import json
import os
import re
import hashlib
from typing import Any, Dict, List, Tuple

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import (
    MilvusError,
    StateFieldError,
)
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.core.settings import get_settings
from knowledge.prompts.import_prompt import (
    DOCUMENT_IDENTITY_SYSTEM_PROMPT,
    DOCUMENT_IDENTITY_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.document_identity_util import (
    build_document_profile_text,
    extract_model_codes,
    normalize_document_name,
    stable_document_hash,
    unique_strings,
)


class DocumentIdentityNode(BaseNode):
    """将文档标题、主要对象和别名拆分为独立字段。"""

    name = "document_identity_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, chunks = self._valid_state(state)
        context, title_candidates = self._build_evidence_context(
            file_title,
            chunks,
            self.config.theme_name_chunk_k,
            self.config.theme_name_chunk_size,
        )
        fallback_identity = self._build_fallback_identity(
            file_title=file_title,
            chunks=chunks,
            title_candidates=title_candidates,
            user_title=state.get("user_document_title") or "",
        )
        llm_identity = self._extract_identity_with_llm(
            file_title=file_title,
            title_candidates=title_candidates,
            context=context,
        )
        identity = self._validate_identity(
            llm_identity=llm_identity,
            fallback=fallback_identity,
            evidence_corpus="\n".join([file_title, *title_candidates, context]),
        )

        content_hash = stable_document_hash(chunks)
        identity["content_hash"] = content_hash
        identity["doc_id"] = f"doc_{content_hash[:24]}"
        identity["profile_text"] = build_document_profile_text(identity)

        aliases_json = json.dumps(identity["aliases"], ensure_ascii=False)
        model_codes_json = json.dumps(identity["model_codes"], ensure_ascii=False)
        for chunk in chunks:
            section_path = str(chunk.get("section_path") or "").strip()
            if section_path:
                chunk["section_id"] = self._stable_section_id(
                    identity["doc_id"],
                    section_path,
                )
                parent_path = " > ".join(section_path.split(" > ")[:-1])
                chunk["parent_section_id"] = (
                    self._stable_section_id(identity["doc_id"], parent_path)
                    if parent_path
                    else ""
                )
            chunk.update({
                "doc_id": identity["doc_id"],
                "canonical_title": identity["canonical_title"],
                "primary_subject": identity["primary_subject"],
                "document_type": identity["document_type"],
                "document_summary": identity["summary"],
                "aliases_json": aliases_json,
                "model_codes_json": model_codes_json,
                # 旧节点与旧数据仍使用 theme_name，暂时保留兼容。
                "theme_name": (
                    identity["primary_subject"]
                    or identity["canonical_title"]
                ),
            })

        state["chunks"] = chunks
        state["document_identity"] = identity
        state["doc_id"] = identity["doc_id"]
        state["canonical_title"] = identity["canonical_title"]
        state["primary_subject"] = identity["primary_subject"]
        state["theme_name"] = (
            identity["primary_subject"] or identity["canonical_title"]
        )
        if get_settings().import_debug_artifacts:
            self._backup_identity(identity, state)
        return state

    @staticmethod
    def _stable_section_id(doc_id: str, section_path: str) -> str:
        raw_value = f"{doc_id}\x1f{section_path}".encode("utf-8")
        return hashlib.sha256(raw_value).hexdigest()[:24]

    def _valid_state(
            self,
            state: ImportGraphState,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not isinstance(file_title, str) or not file_title.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="file_title",
                expected_type=str,
            )
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                expected_type=list,
            )
        return file_title.strip(), chunks

    @staticmethod
    def _build_evidence_context(
            file_title: str,
            chunks: List[Dict[str, Any]],
            max_chunks: int,
            max_chars: int,
    ) -> Tuple[str, List[str]]:
        title_candidates = [file_title]
        context_parts = []
        used_chars = 0

        for chunk in chunks[:max(max_chunks, 1)]:
            if not isinstance(chunk, dict):
                continue
            section_path = str(chunk.get("section_path") or "").strip()
            title = str(chunk.get("section_title") or chunk.get("title") or "").strip()
            if section_path:
                title_candidates.append(section_path.split(" > ", 1)[0].strip())
            if title:
                title_candidates.append(title)

            content = str(chunk.get("content") or "").strip()
            remaining = max(max_chars - used_chars, 0)
            if not content or remaining <= 0:
                continue
            excerpt = content[:remaining]
            context_parts.append(excerpt)
            used_chars += len(excerpt)

        return "\n\n".join(context_parts), unique_strings(
            title_candidates,
            max_items=8,
        )

    def _build_fallback_identity(
            self,
            file_title: str,
            chunks: List[Dict[str, Any]],
            title_candidates: List[str],
            user_title: str = "",
    ) -> Dict[str, Any]:
        cleaned_file_title = re.sub(r"[_\s]+", " ", file_title).strip()
        user_title = str(user_title or "").strip()

        if user_title:
            canonical_title = user_title
            title_source = "user_provided"
            confidence = 0.98
        else:
            heading_candidates = [
                candidate for candidate in title_candidates[1:]
                if normalize_document_name(candidate)
                != normalize_document_name(cleaned_file_title)
                and normalize_document_name(candidate) not in {
                    "摘要", "引言", "引论", "目录", "abstract",
                    "introduction", "contents", "preface", "前言",
                }
            ]
            if heading_candidates:
                canonical_title = heading_candidates[0]
                title_source = "first_heading"
                confidence = 0.82
            else:
                canonical_title = cleaned_file_title
                title_source = "file_name"
                confidence = 0.55

        summary_source = " ".join(
            re.sub(r"\s+", " ", str(chunk.get("content") or "")).strip()
            for chunk in chunks[:2]
            if isinstance(chunk, dict)
        )
        aliases = unique_strings([canonical_title, cleaned_file_title])
        model_codes = unique_strings(
            code
            for value in aliases
            for code in extract_model_codes(value)
        )
        return {
            "canonical_title": canonical_title,
            "primary_subject": canonical_title,
            "aliases": aliases,
            "model_codes": model_codes,
            "document_type": "其他",
            "summary": summary_source[:400],
            "evidence": [canonical_title],
            "title_source": title_source,
            "title_confidence": confidence,
            "requires_review": (
                confidence < self.config.document_title_review_threshold
            ),
        }

    def _extract_identity_with_llm(
            self,
            file_title: str,
            title_candidates: List[str],
            context: str,
    ) -> Dict[str, Any] | None:
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
            response = llm_client.invoke([
                ("system", DOCUMENT_IDENTITY_SYSTEM_PROMPT),
                (
                    "user",
                    DOCUMENT_IDENTITY_USER_PROMPT_TEMPLATE.format(
                        file_title=file_title,
                        title_candidates="\n".join(
                            f"- {candidate}" for candidate in title_candidates
                        ),
                        context=context,
                    ),
                ),
            ])
            content = response.content
            if isinstance(content, dict):
                return content
            if not isinstance(content, str):
                return None
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            return json.loads(content)
        except Exception as exc:
            self.logger.warning("文档档案抽取失败，使用证据回退: %s", exc)
            return None

    def _validate_identity(
            self,
            llm_identity: Dict[str, Any] | None,
            fallback: Dict[str, Any],
            evidence_corpus: str,
    ) -> Dict[str, Any]:
        if not isinstance(llm_identity, dict):
            return fallback

        corpus_key = normalize_document_name(evidence_corpus)
        evidence = unique_strings(llm_identity.get("evidence") or [], max_items=3)
        supported_evidence = [
            item for item in evidence
            if normalize_document_name(item) in corpus_key
        ]
        canonical_title = str(llm_identity.get("canonical_title") or "").strip()
        primary_subject = str(llm_identity.get("primary_subject") or "").strip()

        # 没有可回溯证据时，不采纳 LLM 生成的名称。
        canonical_supported = (
            bool(canonical_title)
            and normalize_document_name(canonical_title) in corpus_key
        )
        if not canonical_supported or not supported_evidence:
            return fallback
        if (
            not primary_subject
            or normalize_document_name(primary_subject) not in corpus_key
        ):
            primary_subject = canonical_title

        raw_aliases = unique_strings(
            [
                canonical_title,
                *list(llm_identity.get("aliases") or []),
                *fallback.get("aliases", []),
            ],
            max_items=12,
        )
        supported_aliases = [
            alias for alias in raw_aliases
            if normalize_document_name(alias) in corpus_key
            or alias in fallback.get("aliases", [])
        ]
        model_codes = unique_strings(
            [
                *fallback.get("model_codes", []),
                *(
                    code
                    for value in [canonical_title, primary_subject, *supported_aliases]
                    for code in extract_model_codes(value)
                ),
            ],
            max_items=12,
        )
        try:
            confidence = float(llm_identity.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        title_source = "llm_with_evidence"
        for source_name, candidate in (
            ("user_provided", fallback["canonical_title"] if fallback["title_source"] == "user_provided" else ""),
            ("first_heading", fallback["canonical_title"] if fallback["title_source"] == "first_heading" else ""),
            ("file_name", fallback["canonical_title"] if fallback["title_source"] == "file_name" else ""),
        ):
            if candidate and normalize_document_name(candidate) == normalize_document_name(canonical_title):
                title_source = source_name
                break

        summary = re.sub(
            r"\s+",
            " ",
            str(llm_identity.get("summary") or fallback.get("summary") or ""),
        ).strip()[:400]
        return {
            "canonical_title": canonical_title,
            "primary_subject": primary_subject or canonical_title,
            "aliases": supported_aliases or fallback["aliases"],
            "model_codes": model_codes,
            "document_type": str(
                llm_identity.get("document_type") or "其他"
            ).strip()[:80],
            "summary": summary,
            "evidence": supported_evidence,
            "title_source": title_source,
            "title_confidence": confidence,
            "requires_review": (
                confidence < self.config.document_title_review_threshold
            ),
        }

    @staticmethod
    def _embed_profile(profile_text: str) -> Tuple[List[float], Dict[int, float]]:
        model = AIClients.get_bge_m3_client()
        result = model.encode([profile_text], return_dense=True, return_sparse=True)
        dense_vector = result["dense_vecs"][0].tolist()
        sparse_vector = {
            int(key): float(value)
            for key, value in dict(result["lexical_weights"][0]).items()
        }
        return dense_vector, sparse_vector

    def _upsert_document_profile(
            self,
            identity: Dict[str, Any],
            dense_vector: List[float],
            sparse_vector: Dict[int, float],
    ) -> None:
        try:
            client = StorageClients.get_milvus()
            collection_name = self.config.document_registry_collection
            if not client.has_collection(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    schema=self._build_schema(client, len(dense_vector)),
                    index_params=self._build_index(client),
                )

            data = {
                "doc_id": identity["doc_id"],
                "canonical_title": identity["canonical_title"],
                "primary_subject": identity["primary_subject"],
                "aliases_json": json.dumps(identity["aliases"], ensure_ascii=False),
                "model_codes_json": json.dumps(identity["model_codes"], ensure_ascii=False),
                "document_type": identity["document_type"],
                "summary": identity["summary"],
                "profile_text": identity["profile_text"],
                "title_source": identity["title_source"],
                "title_confidence": float(identity["title_confidence"]),
                "requires_review": bool(identity["requires_review"]),
                "content_hash": identity["content_hash"],
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
            client.upsert(collection_name=collection_name, data=[data])
        except Exception as exc:
            raise MilvusError(
                node_name=self.name,
                message=f"文档档案写入 Milvus 失败: {exc}",
            ) from exc

    @staticmethod
    def _build_schema(client: MilvusClient, dim: int):
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        for field_name in (
            "canonical_title",
            "primary_subject",
            "aliases_json",
            "model_codes_json",
            "document_type",
            "summary",
            "profile_text",
            "title_source",
            "content_hash",
        ):
            schema.add_field(
                field_name=field_name,
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
        schema.add_field(field_name="title_confidence", datatype=DataType.FLOAT)
        schema.add_field(field_name="requires_review", datatype=DataType.BOOL)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        return schema

    @staticmethod
    def _build_index(client: MilvusClient):
        index_params = client.prepare_index_params()
        index_params.add_index(
            index_name="dense_vector_index",
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            index_name="sparse_vector_index",
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        return index_params

    @staticmethod
    def _backup_identity(
            identity: Dict[str, Any],
            state: ImportGraphState,
    ) -> None:
        file_dir = state.get("file_dir")
        if not file_dir:
            return
        os.makedirs(file_dir, exist_ok=True)
        path = os.path.join(file_dir, "document_identity.json")
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(identity, file, ensure_ascii=False, indent=2)
        except OSError:
            # 备份文件不影响主入库链路。
            return
