"""Rerank 后的层级上下文扩展节点。"""

import re
from typing import Any, Dict, List

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.storage_clients import StorageClients


class ContextExpansionNode(BaseNode):
    """为高排名切片补充同一章节内的相邻切片。

    扩展严格使用 ``section_id`` 作为边界，不会跨章节拼接。
    旧数据没有层级字段或 Milvus 查询失败时，保留原重排结果。
    """

    name = "context_expansion_node"
    _SECTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    _OUTPUT_FIELDS = [
        "chunk_id",
        "content",
        "title",
        "theme_name",
        "doc_id",
        "canonical_title",
        "primary_subject",
        "document_type",
        "document_summary",
        "parent_title",
        "file_title",
        "section_id",
        "section_path",
        "parent_summary",
        "chunk_index",
        "section_chunk_index",
    ]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        reranked_docs = state.get("reranked_docs") or []
        if not reranked_docs:
            state["expanded_docs"] = []
            return state

        top_k = max(int(self.config.context_expansion_top_k or 0), 0)
        expanded_docs: List[Dict[str, Any]] = []

        for rank, doc in enumerate(reranked_docs):
            if not isinstance(doc, dict):
                continue
            normalized_doc = dict(doc)
            # 比较题/多事实题中被覆盖保护的证据即使排在普通 Top-K
            # 之外，也需要扩展相邻上下文，避免只保留孤立切片。
            should_expand = top_k > 0 and (
                rank < top_k
                or bool(normalized_doc.get("coverage_protected"))
                or bool(normalized_doc.get("conflict_judge_promoted"))
            )
            if should_expand:
                normalized_doc = self._expand_document(normalized_doc)
            expanded_docs.append(normalized_doc)

        state["expanded_docs"] = expanded_docs
        return state

    def _expand_document(self, anchor: Dict[str, Any]) -> Dict[str, Any]:
        section_id = str(anchor.get("section_id") or "").strip()
        section_chunk_index = anchor.get("section_chunk_index")
        if (
            not self._SECTION_ID_PATTERN.fullmatch(section_id)
            or not isinstance(section_chunk_index, int)
        ):
            return anchor

        window = max(int(self.config.context_expansion_window or 0), 0)
        if window <= 0:
            return anchor

        lower_bound = max(section_chunk_index - window, 0)
        upper_bound = section_chunk_index + window
        doc_id = str(anchor.get("doc_id") or "").strip()
        document_filter = (
            f' and doc_id == "{doc_id}"'
            if self._SECTION_ID_PATTERN.fullmatch(doc_id)
            else ""
        )

        try:
            milvus_client = StorageClients.get_milvus()
            neighbors = milvus_client.query(
                collection_name=self.config.chunks_collection,
                filter=(
                    f'section_id == "{section_id}" and '
                    f"section_chunk_index >= {lower_bound} and "
                    f"section_chunk_index <= {upper_bound}"
                    f"{document_filter}"
                ),
                output_fields=self._OUTPUT_FIELDS,
                limit=window * 2 + 1,
            )
        except Exception as exc:
            self.logger.warning(
                "章节上下文扩展失败，使用原切片: "
                f"section_id={section_id}, error={exc}"
            )
            return anchor

        valid_neighbors = [
            item for item in (neighbors or [])
            if isinstance(item, dict)
            and item.get("section_id") == section_id
            and isinstance(item.get("section_chunk_index"), int)
            and lower_bound <= item["section_chunk_index"] <= upper_bound
        ]
        if not valid_neighbors:
            return anchor

        return self._build_expanded_document(anchor, valid_neighbors)

    def _build_expanded_document(
            self,
            anchor: Dict[str, Any],
            neighbors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        anchor_index = anchor.get("section_chunk_index")
        anchor_chunk_id = anchor.get("chunk_id")
        max_chars = max(int(self.config.context_expansion_max_chars or 0), 0)

        # 命中切片固定排在最前面，预算紧张时也不会被摘要或相邻块挤掉。
        anchor_content = (anchor.get("content") or "").strip()
        candidate_blocks = (
            [f"【命中切片】\n{anchor_content}"] if anchor_content else []
        )

        parent_summary = (anchor.get("parent_summary") or "").strip()
        if parent_summary:
            candidate_blocks.append(f"【父块摘要】\n{parent_summary}")

        before = sorted(
            (
                item for item in neighbors
                if item.get("section_chunk_index") < anchor_index
            ),
            key=lambda item: item["section_chunk_index"],
            reverse=True,
        )
        after = sorted(
            (
                item for item in neighbors
                if item.get("section_chunk_index") > anchor_index
            ),
            key=lambda item: item["section_chunk_index"],
        )

        for label, items in (("前文", before), ("后文", after)):
            for item in items:
                content = (item.get("content") or "").strip()
                if content:
                    candidate_blocks.append(f"【{label}】\n{content}")

        blocks: list[str] = []
        current_chars = 0
        for block in candidate_blocks:
            separator = "\n\n" if blocks else ""
            remaining = max_chars - current_chars - len(separator)
            if max_chars > 0 and remaining <= 0:
                break
            body = block if max_chars <= 0 else block[:remaining]
            if not body:
                break
            blocks.append(body)
            current_chars += len(separator) + len(body)
            if len(body) < len(block):
                break

        expanded_content = "\n\n".join(blocks)

        expanded_chunk_ids = [anchor_chunk_id]
        expanded_chunk_ids.extend(
            item.get("chunk_id")
            for item in before + after
            if item.get("chunk_id") is not None
            and item.get("chunk_id") != anchor_chunk_id
        )

        return {
            **anchor,
            "anchor_content": anchor_content,
            "content": expanded_content or anchor_content,
            "expanded_chunk_ids": expanded_chunk_ids,
            "context_expanded": len(expanded_chunk_ids) > 1,
        }
