"""检索结果规整与分支内RRF工具。"""

from typing import Any, Iterable


CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "content", "theme_name", "title", "doc_id",
    "canonical_title", "primary_subject", "document_type",
    "document_summary", "parent_title", "file_title", "section_id",
    "section_path", "parent_summary", "chunk_index",
    "section_chunk_index",
]


def hit_to_chunk(hit: Any) -> dict[str, Any]:
    getter = hit.get if hasattr(hit, "get") else lambda _key: None
    hit_id = getattr(hit, "id", None)
    if isinstance(hit, dict):
        hit_id = hit.get("id", hit_id)
    distance = getattr(hit, "distance", None)
    if isinstance(hit, dict):
        distance = hit.get("distance", hit.get("score", distance))
    chunk_id = getter("chunk_id") or hit_id
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "score": float(distance) if isinstance(distance, (int, float)) else None,
        **{field: getter(field) for field in CHUNK_OUTPUT_FIELDS if field != "chunk_id"},
    }


def merge_ranked_chunk_lists(
        ranked_lists: Iterable[tuple[str, list[dict[str, Any]], float]],
        limit: int,
        rrf_k: int = 60,
        min_per_list: int = 0,
) -> list[dict[str, Any]]:
    """按chunk_id融合多个排名列表，并保留各列表名次。"""
    ranked_lists = list(ranked_lists)
    scores: dict[Any, float] = {}
    documents: dict[Any, dict[str, Any]] = {}
    for list_name, chunks, weight in ranked_lists:
        for rank, chunk in enumerate(chunks or [], start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id is None:
                continue
            if chunk_id not in documents:
                documents[chunk_id] = dict(chunk)
            else:
                for key, value in chunk.items():
                    if documents[chunk_id].get(key) in (None, "", []):
                        documents[chunk_id][key] = value
            documents[chunk_id].setdefault("branch_ranks", {})[list_name] = rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)

    ordered_all = sorted(scores, key=lambda item: scores[item], reverse=True)
    ordered = ordered_all[:limit] if limit > 0 else ordered_all

    # 多查询融合时，避免仅在一条查询中命中的证据被重复命中的通用块挤出。
    if limit > 0 and min_per_list > 0 and len(ordered_all) > limit:
        protected: list[Any] = []
        for _list_name, chunks, _weight in ranked_lists:
            added = 0
            for chunk in chunks or []:
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id is None or chunk_id in protected:
                    continue
                protected.append(chunk_id)
                added += 1
                if added >= min_per_list:
                    break

        selected = set(ordered)
        protected_set = set(protected)
        for chunk_id in protected:
            if chunk_id in selected:
                continue
            replacement_index = next(
                (
                    index for index in range(len(ordered) - 1, -1, -1)
                    if ordered[index] not in protected_set
                ),
                None,
            )
            if replacement_index is None:
                break
            selected.discard(ordered[replacement_index])
            ordered[replacement_index] = chunk_id
            selected.add(chunk_id)
        ordered.sort(key=lambda item: scores[item], reverse=True)
    return [
        {**documents[chunk_id], "branch_fusion_score": scores[chunk_id]}
        for chunk_id in ordered
    ]
