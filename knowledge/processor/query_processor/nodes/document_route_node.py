"""文档软路由：只有明确、唯一匹配时才产生 doc_id 硬过滤。"""

import json
import re
import time
from typing import Any, Dict, List, Tuple

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import (
    DOCUMENT_ROUTE_SYSTEM_PROMPT,
    DOCUMENT_ROUTE_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.document_identity_util import (
    extract_model_codes,
    normalize_document_name,
    parse_string_list,
    unique_strings,
)
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)
from knowledge.utils.mongo_history_util import get_recent_messages


class DocumentRouteNode(BaseNode):
    name = "document_route_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = (
            state.get("retrieval_query")
            or state.get("original_query")
            or ""
        ).strip()
        history = get_recent_messages(session_id=state.get("session_id") or "")
        state["history"] = history
        history_text = self._format_history(history)

        registry_documents = self._load_registry_documents()
        deterministic_documents, deterministic_is_soft = (
            self._deterministic_registry_matches(
                original_query,
                registry_documents,
            )
        )

        document_mentions, rewritten_query, retrieval_subqueries = self._understand_query(
            original_query,
            history_text,
        )
        if not retrieval_subqueries:
            retrieval_subqueries = self._fallback_subqueries(
                rewritten_query or original_query
            )
        active_documents = self._active_documents_for_pronoun(
            original_query,
            history,
        )
        if not document_mentions and active_documents:
            document_mentions = unique_strings(
                document.get("canonical_title")
                for document in active_documents
            )

        state["rewritten_query"] = rewritten_query or original_query
        state["retrieval_queries"] = self._retrieval_queries(
            state["rewritten_query"],
            retrieval_subqueries,
        )
        state["query_decomposed"] = len(state["retrieval_queries"]) > 1
        state["document_mentions"] = document_mentions
        state["document_candidates"] = []
        state["selected_documents"] = []
        state["hard_filter_doc_ids"] = []
        state["soft_filter_doc_ids"] = []
        state["theme_names"] = []
        state["document_route_mode"] = "global"
        state["document_route_locked"] = False
        state["document_route_reason"] = "no_document_mention"
        state["document_route_scope_history"] = []
        state["vector_route_fallback"] = False
        state["hyde_route_fallback"] = False
        state["bm25_route_fallback"] = False

        if active_documents:
            selected_documents = self._deduplicate_documents(active_documents)
            active_doc_ids = [
                document.get("doc_id")
                for document in selected_documents
                if document.get("doc_id")
            ]
            if active_doc_ids:
                self._set_hard_route(state, selected_documents)
                state["document_route_locked"] = True
                state["document_route_reason"] = "conversation_document_reference"
                return state

        if deterministic_documents:
            state["document_candidates"] = deterministic_documents
            if deterministic_is_soft:
                self._set_soft_route(state, deterministic_documents)
                state["document_route_reason"] = "explicit_document_ambiguous"
            else:
                self._set_hard_route(state, deterministic_documents)
                state["document_route_reason"] = "explicit_document_match"
            state["document_route_locked"] = True
            return state

        # 主题型问题没有明确文档指向时，直接放行全库检索。
        if not document_mentions:
            return state

        candidates_by_mention = {
            mention: self._search_registry(mention)
            for mention in document_mentions
        }
        all_candidates = self._deduplicate_documents(
            candidate
            for candidates in candidates_by_mention.values()
            for candidate in candidates
        )
        state["document_candidates"] = all_candidates

        exact_by_mention = {
            mention: self._exact_document_matches(
                mention,
                candidates,
                original_query=original_query,
            )
            for mention, candidates in candidates_by_mention.items()
        }

        logical_matches = {
            mention: self._logical_groups(matches)
            for mention, matches in exact_by_mention.items()
        }
        ambiguous_candidates = self._deduplicate_documents(
            candidate
            for groups in logical_matches.values()
            if len(groups) > 1
            for group in groups.values()
            for candidate in group
        )
        if ambiguous_candidates:
            self._set_soft_route(state, ambiguous_candidates)
            state["document_route_locked"] = True
            state["document_route_reason"] = "explicit_document_ambiguous"
            return state

        # 所有明确指向都必须各自唯一命中，否则不能只根据向量高分硬过滤。
        if all(
            len(logical_matches.get(mention) or {}) == 1
            for mention in document_mentions
        ):
            selected_documents = self._deduplicate_documents(
                candidate
                for groups in logical_matches.values()
                for group in groups.values()
                for candidate in group
            )
            self._set_hard_route(state, selected_documents)
            state["document_route_locked"] = True
            state["document_route_reason"] = "explicit_document_match"

        elif all_candidates:
            # 纯向量相似只能说明“可能相关”，不能证明用户明确指向这些文档。
            # 候选保留用于日志与评测，但正文检索仍走全库；只有精确别名/型号
            # 命中多个逻辑文档时，前面的 ambiguous_candidates 分支才启用软路由。
            state["document_candidates"] = (
                all_candidates[:self.config.document_route_max_options]
            )
            state["document_route_reason"] = "weak_document_candidate"

        # 只有弱候选时保留候选用于观测，不改变全库正文检索。
        return state

    def _understand_query(
            self,
            original_query: str,
            history_text: str,
    ) -> Tuple[List[str], str, List[str]]:
        fallback_mentions = self._deterministic_mentions(original_query)
        try:
            llm_client = AIClients.get_llm_client(
                response_format=True,
                role="fast",
                thinking="none",
            )
            started_at = time.perf_counter()
            try:
                response = llm_client.invoke([
                    ("system", DOCUMENT_ROUTE_SYSTEM_PROMPT),
                    (
                        "user",
                        DOCUMENT_ROUTE_USER_PROMPT_TEMPLATE.format(
                            history_text=history_text,
                            query=original_query,
                        ),
                    ),
                ])
            finally:
                self.logger.info(
                    "LLM latency: node_name=document_route model_name=%s "
                    "thinking_level=none elapsed_ms=%.1f",
                    getattr(llm_client, "model_name", "unknown"),
                    (time.perf_counter() - started_at) * 1000,
                )
            content = response.content
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        content,
                    )
                parsed = json.loads(content)
            elif isinstance(content, dict):
                parsed = content
            else:
                parsed = {}
            mentions = unique_strings(parsed.get("document_mentions") or [])
            rewritten_query = str(
                parsed.get("rewritten_query") or original_query
            ).strip()
            subqueries = unique_strings(
                parsed.get("retrieval_subqueries") or [],
                max_items=3,
            )
            return mentions, rewritten_query, subqueries
        except Exception as exc:
            self.logger.warning("查询改写失败，使用原查询放行: %s", exc)
            return fallback_mentions, original_query, []

    @staticmethod
    def _retrieval_queries(
            rewritten_query: str,
            subqueries: List[str],
    ) -> List[str]:
        return unique_strings(
            [rewritten_query, *(subqueries or [])],
            max_items=4,
        ) or [rewritten_query]

    @staticmethod
    def _fallback_subqueries(query: str) -> List[str]:
        """LLM漏拆时，对明显的比较/多事实并列问句做保守拆分。"""
        if not isinstance(query, str):
            return []
        normalized = re.sub(r"\s+", " ", query).strip()
        complex_markers = re.search(
            r"分别|区别|不同|对比|相比|各自|同时|哪些|为何|为什么|"
            r"结论是什么|多少|有什么影响",
            normalized,
        )
        clauses = [
            clause.strip(" ，,；;。？?")
            for clause in re.split(r"[，,；;]", normalized)
            if clause.strip(" ，,；;。？?")
        ]
        if not complex_markers or len(clauses) < 2:
            return []

        # 过短的“为什么/多少”等从句离开前文无法独立理解，不单独检索。
        usable = [
            f"{clause}？"
            for clause in clauses
            if len(normalize_document_name(clause)) >= 6
        ]
        return unique_strings(usable, max_items=3) if len(usable) >= 2 else []

    def _load_registry_documents(self) -> List[Dict[str, Any]]:
        """读取小型文档注册表，供标题/别名确定性匹配。"""
        try:
            client = StorageClients.get_milvus()
            rows = client.query(
                collection_name=self.config.document_registry_collection,
                filter="",
                output_fields=[
                    "doc_id", "canonical_title", "primary_subject",
                    "aliases_json", "model_codes_json", "document_type",
                    "summary", "title_confidence", "requires_review",
                ],
                limit=self.config.document_registry_scan_limit,
            )
        except Exception as exc:
            self.logger.warning("读取文档注册表失败，继续使用语义路由: %s", exc)
            return []
        return self._deduplicate_documents(
            self._registry_document(row) for row in (rows or [])
        )

    @staticmethod
    def _registry_document(source: Any) -> Dict[str, Any]:
        getter = source.get if hasattr(source, "get") else lambda _key: None
        return {
            "doc_id": getter("doc_id"),
            "canonical_title": getter("canonical_title"),
            "primary_subject": getter("primary_subject"),
            "aliases": parse_string_list(getter("aliases_json")),
            "model_codes": parse_string_list(getter("model_codes_json")),
            "document_type": getter("document_type"),
            "summary": getter("summary"),
            "title_confidence": getter("title_confidence"),
            "requires_review": getter("requires_review"),
        }

    @classmethod
    def _deterministic_registry_matches(
            cls,
            query: str,
            documents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """完整标题/别名优先；唯一型号其次。返回(候选, 是否软路由)。"""
        query_key = normalize_document_name(query)
        alias_map: Dict[str, List[Dict[str, Any]]] = {}
        for document in documents:
            names = [
                (document.get("canonical_title"), True),
                *((alias, False) for alias in (document.get("aliases") or [])),
            ]
            for alias, is_canonical in names:
                alias_key = normalize_document_name(alias)
                if (
                    len(alias_key) >= 4
                    and alias_key in query_key
                    and cls._is_routable_document_name(
                        alias,
                        query,
                        is_canonical=is_canonical,
                    )
                ):
                    alias_map.setdefault(alias_key, []).append(document)

        if alias_map:
            # 长标题优先，避免其内部较短别名制造伪歧义。
            max_length = max(map(len, alias_map))
            strong_aliases = {
                key: value for key, value in alias_map.items()
                if len(key) == max_length
                or not any(key in longer for longer in alias_map if key != longer)
            }
            matched = cls._deduplicate_documents(
                item for values in strong_aliases.values() for item in values
            )
            return matched, len(cls._logical_groups(matched)) > len(strong_aliases)

        query_codes = set(extract_model_codes(query))
        if not query_codes:
            return [], False
        code_matches = cls._deduplicate_documents(
            document for document in documents
            if query_codes <= set(document.get("model_codes") or [])
        )
        return code_matches, len(cls._logical_groups(code_matches)) > 1

    @staticmethod
    def _logical_groups(
            documents: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for document in documents or []:
            key = normalize_document_name(document.get("canonical_title"))
            key = key or str(document.get("doc_id") or "")
            if key:
                groups.setdefault(key, []).append(document)
        return groups

    @staticmethod
    def _set_hard_route(
            state: QueryGraphState,
            documents: List[Dict[str, Any]],
    ) -> None:
        selected = DocumentRouteNode._deduplicate_documents(documents)
        doc_ids = unique_strings(
            document.get("doc_id") for document in selected
        )
        if not doc_ids:
            return
        state["selected_documents"] = selected
        state["hard_filter_doc_ids"] = doc_ids
        state["soft_filter_doc_ids"] = []
        state["theme_names"] = unique_strings(
            document.get("primary_subject") or document.get("canonical_title")
            for document in selected
        )
        state["document_route_mode"] = "hard"

    @staticmethod
    def _set_soft_route(
            state: QueryGraphState,
            documents: List[Dict[str, Any]],
    ) -> None:
        candidates = DocumentRouteNode._deduplicate_documents(documents)
        doc_ids = unique_strings(
            document.get("doc_id") for document in candidates
        )
        if not doc_ids:
            return
        state["document_candidates"] = candidates
        state["soft_filter_doc_ids"] = doc_ids
        state["document_route_mode"] = "soft"

    @staticmethod
    def _deterministic_mentions(query: str) -> List[str]:
        mentions = re.findall(r"《([^》]{2,100})》", query or "")
        mentions.extend(extract_model_codes(query))
        return unique_strings(mentions)

    def _search_registry(self, mention: str) -> List[Dict[str, Any]]:
        try:
            embedding_client = AIClients.get_bge_m3_client()
            milvus_client = StorageClients.get_milvus()
            embedding = embedding_client.encode(
                [mention],
                return_dense=True,
                return_sparse=True,
            )
            dense_vector = embedding["dense_vecs"][0].tolist()
            sparse_vector = {
                int(key): float(value)
                for key, value in dict(embedding["lexical_weights"][0]).items()
            }
            requests = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=self.config.document_route_limit,
            )
            response = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.document_registry_collection,
                search_requests=requests,
                ranker_weights=(0.5, 0.5),
                limit=self.config.document_route_limit,
                output_fields=[
                    "doc_id",
                    "canonical_title",
                    "primary_subject",
                    "aliases_json",
                    "model_codes_json",
                    "document_type",
                    "summary",
                    "title_confidence",
                    "requires_review",
                ],
            )
        except Exception as exc:
            self.logger.warning("文档注册表查询失败，回退全库检索: %s", exc)
            return []

        documents = []
        for hit in response[0] if response else []:
            entity = hit.get("entity") if hasattr(hit, "get") else None
            source = entity if isinstance(entity, dict) else hit
            getter = source.get if hasattr(source, "get") else lambda _key: None
            distance = (
                getattr(hit, "distance", None)
                if not isinstance(hit, dict)
                else hit.get("distance")
            )
            documents.append({
                "doc_id": getter("doc_id") or getattr(hit, "id", None),
                "canonical_title": getter("canonical_title"),
                "primary_subject": getter("primary_subject"),
                "aliases": parse_string_list(getter("aliases_json")),
                "model_codes": parse_string_list(getter("model_codes_json")),
                "document_type": getter("document_type"),
                "summary": getter("summary"),
                "title_confidence": getter("title_confidence"),
                "requires_review": getter("requires_review"),
                "route_score": float(distance) if isinstance(distance, (int, float)) else None,
            })
        return self._deduplicate_documents(documents)

    @classmethod
    def _exact_document_matches(
            cls,
            mention: str,
            documents: List[Dict[str, Any]],
            original_query: str = "",
    ) -> List[Dict[str, Any]]:
        mention_key = normalize_document_name(mention)
        alias_matches = []
        for document in documents:
            names = [
                (document.get("canonical_title"), True),
                *((alias, False) for alias in (document.get("aliases") or [])),
            ]
            if mention_key and any(
                mention_key == normalize_document_name(name)
                and cls._is_routable_document_name(
                    name,
                    original_query or mention,
                    is_canonical=is_canonical,
                )
                for name, is_canonical in names
            ):
                alias_matches.append(document)

        # 完整文档名/别名精确匹配的优先级高于型号共现。
        # 例如用户说出“RS-12使用手册”时，不应再因同型号
        # 的“RS-12安全手册”而产生歧义。
        if alias_matches:
            return cls._deduplicate_documents(alias_matches)

        mention_codes = set(extract_model_codes(mention))
        if not mention_codes:
            return []
        return cls._deduplicate_documents(
            document
            for document in documents
            if mention_codes <= set(document.get("model_codes") or [])
        )

    @staticmethod
    def _is_routable_document_name(
            name: Any,
            query: str,
            *,
            is_canonical: bool,
    ) -> bool:
        """区分真正的文档名/产品型号与 STAP 一类通用技术缩写。"""
        name_key = normalize_document_name(name)
        if not name_key:
            return False
        if is_canonical and len(name_key) >= 6:
            return True
        if len(name_key) >= 12 or extract_model_codes(str(name or "")):
            return True
        return bool(re.search(
            r"《[^》]+》|(?:论文|文档|报告|手册|说明书|文章|资料|标准|指南)",
            query or "",
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _deduplicate_documents(documents) -> List[Dict[str, Any]]:
        result = []
        seen = set()
        for document in documents:
            if not isinstance(document, dict):
                continue
            key = document.get("doc_id") or normalize_document_name(
                document.get("canonical_title")
            )
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(document)
        return result

    @staticmethod
    def _format_history(history: List[Dict[str, Any]]) -> str:
        lines = []
        for message in reversed(history):
            if not isinstance(message, dict):
                continue
            text = str(message.get("text") or "").strip()
            if text:
                lines.append(f"{message.get('role') or 'unknown'}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _active_documents_for_pronoun(
            query: str,
            history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not re.search(r"(?:这篇|这份|这个|该文档|该论文|它)", query or ""):
            return []
        for message in history:
            selected = message.get("selected_documents") if isinstance(message, dict) else None
            if isinstance(selected, list) and selected:
                return [item for item in selected if isinstance(item, dict)]
        return []
