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
        """计算key的排序键。

        Args:
            document: 待处理的文档数据。

        Returns:
            处理结果。
        """
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

    if any(d.get("retrieved_for_subquestions") for d in unique_documents):
        # 多子问题的分数只在组内有意义；保持合并器给出的轮流取证顺序。
        return unique_documents
    return sorted(unique_documents, key=sort_key)


def format_grouped_evidence(documents, max_chars, per_doc_max_chars, selected_chunk_ids=None):
    """先覆盖不同子问题，再按剩余字数公平分配正文；元数据计入总预算。"""
    documents = [d for d in coverage_ordered_documents(documents) if str(d.get("content") or "").strip()]
    primary, secondary, seen = [], [], set()
    for doc in documents:
        indices = set(doc.get("retrieved_for_subquestions") or [])
        (primary if indices - seen else secondary).append(doc)
        seen.update(indices)
    records, header_chars = [], 0
    for doc in primary + secondary:
        indices = doc.get("retrieved_for_subquestions") or []
        header = (
            f"【资料{len(records) + 1}】\n"
            f"候选证据归属（不代表已充分支持）：子问题{','.join(map(str, indices))}\n"
            f"文档：{str(doc.get('canonical_title') or doc.get('file_title') or doc.get('theme_name') or '')[:120]}\n"
            f"章节：{str(doc.get('section_path') or doc.get('title') or '')[:120]}\n"
            f"切片ID：{doc.get('expanded_chunk_ids') or [doc.get('chunk_id')]}\n正文：\n"
        )
        body = str(doc.get("content") or "").strip()[:per_doc_max_chars]
        # 给已选中的每份资料和新资料都保留一段正文，避免只剩空标题。
        minimum = sum(min(64, len(r[1])) for r in records) + min(64, len(body))
        required = header_chars + len(header) + 2 * len(records) + minimum
        if required > max_chars:
            continue
        records.append((header, body, doc))
        header_chars += len(header)
    if not records:
        return ""
    remaining = max_chars - header_chars - 2 * (len(records) - 1)
    allocations = [0] * len(records)
    while remaining > 0:
        active = [i for i, (_, body, _) in enumerate(records) if allocations[i] < len(body)]
        if not active:
            break
        share = max(1, remaining // len(active))
        for i in active:
            take = min(share, remaining, len(records[i][1]) - allocations[i])
            allocations[i] += take
            remaining -= take
    if selected_chunk_ids is not None:
        selected_chunk_ids.extend(doc.get("chunk_id") for _, _, doc in records)
    return "\n\n".join(header + body[:size] for (header, body, _), size in zip(records, allocations))
