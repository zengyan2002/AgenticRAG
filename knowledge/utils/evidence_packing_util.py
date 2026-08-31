"""证据上下文的稳定排序工具。"""

from __future__ import annotations

from typing import Any, Iterable


def coverage_ordered_documents(
    documents: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """覆盖保护证据优先，其余证据按最终重排分数降序排列。"""
    unique_documents: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            continue
        chunk_id = document.get("chunk_id")
        key = str(chunk_id) if chunk_id is not None else f"index:{index}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_documents.append(document)

    def sort_key(document: dict[str, Any]) -> tuple[int, float]:
        protected = bool(
            document.get("coverage_protected")
            or document.get("conflict_judge_promoted")
        )
        try:
            score = float(
                document.get("final_rerank_score")
                if document.get("final_rerank_score") is not None
                else document.get("rerank_score") or 0.0
            )
        except (TypeError, ValueError):
            score = 0.0
        return (0 if protected else 1, -score)

    return sorted(unique_documents, key=sort_key)
