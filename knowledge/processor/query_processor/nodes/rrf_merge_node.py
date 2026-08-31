from typing import Any, Dict, List

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState


class RRFMergeNode(BaseNode):
    name = "rrf_merge_node"
    def __init__(self, config=None):
        super().__init__(config=config)
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k


    def process(self, state: QueryGraphState) -> dict[str, list]:
        #1 获取各路检索结果
        #混合向量检索的查询结果
        embedding_chunks =  state.get('embedding_chunks') or []
        #Hyde检索出的结果
        hyde_embedding_chunks = state.get('hyde_embedding_chunks') or []
        bm25_chunks = state.get('bm25_chunks') or []

        #2 格式规整化，同时保留层级元数据供后续重排与上下文扩展使用。
        embedding_chunks_normalized = [
            self._normalize_chunk(chunk)
            for chunk in embedding_chunks
            if isinstance(chunk, dict)
        ]
        hyde_embedding_chunks_normalized = [
            self._normalize_chunk(chunk)
            for chunk in hyde_embedding_chunks
            if isinstance(chunk, dict)
        ]
        bm25_chunks_normalized = [
            self._normalize_chunk(chunk)
            for chunk in bm25_chunks
            if isinstance(chunk, dict)
        ]

        #3 为不同路的搜索结果设置不同的权重
        search_source = {
            "direct_hybrid": (
                embedding_chunks_normalized,
                self.config.rrf_direct_weight,
            ),
            "hyde": (
                hyde_embedding_chunks_normalized,
                self.config.rrf_hyde_weight,
            ),
            "bm25": (
                bm25_chunks_normalized,
                self.config.rrf_bm25_weight,
            ),
        }
        # 4. 构建 rrf_inputs
        rrf_inputs = [
            (chunks, weight, source_name)
            for source_name, (chunks, weight) in search_source.items()
        ]
        
        # 5 采用rrf进行融合
        rrf_result = self._rrf_merge(
            rrf_inputs,
            self._rrf_k,
            self._top_k,
            self.config.rrf_min_exclusive_per_branch,
        )

        # 6 回填融合分数和命中来源，后续节点无需重新推断。
        rrf_chunks = [
            {
                **chunk,
                "fusion_score": float(score),
                "rrf_rank": rank,
                "retrieval_sources": sorted(chunk.get("retrieval_sources") or []),
                "retrieval_query_names_hit": sorted(
                    (chunk.get("subquery_ranks") or {}).keys(),
                    key=self._query_name_order,
                ),
            }
            for rank, (chunk, score) in enumerate(rrf_result, start=1)
        ]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_result:
            scores = [s for _, s in rrf_result]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        return {"rrf_chunks": rrf_chunks}

    @staticmethod
    def _query_name_order(query_name: str) -> tuple[int, str]:
        """让 query_2 排在 query_10 前，并兼容非标准名称。"""
        try:
            return int(query_name.rsplit("_", 1)[-1]), query_name
        except (TypeError, ValueError):
            return 10 ** 9, str(query_name)

    @staticmethod
    def _normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        fields = (
            "chunk_id",
            "title",
            "content",
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
            "branch_ranks",
        )
        normalized = {field: chunk.get(field) for field in fields}
        normalized["retrieval_score"] = chunk.get("score")
        return normalized

    def _rrf_merge(
            self,
            rrf_inputs: List,
            _rrf_k: int,
            _top_k,
            min_exclusive_per_branch: int = 0,
    ):

        #声明记录分数的字典  key是chunk_id  value是分数
        rrf_scores = {}

        #声明记录内容的字典  key是chunk_id  value是文档内容
        rrf_datas={}

        #遍历rrf_inputs
        for input_item in rrf_inputs:
            if len(input_item) == 3:
                chunks, weight, source_name = input_item
            else:
                chunks, weight = input_item
                source_name = "unknown"
            for rank, chunk in enumerate(chunks, start=1):
                #拿到chunk_id
                chunk_id = chunk.get("chunk_id")
                if chunk_id is None:
                    continue

                # 去重，并用后续分支补齐先前结果中缺少的层级字段。
                if chunk_id not in rrf_datas:
                    rrf_datas[chunk_id] = {
                        **chunk,
                        "retrieval_sources": {source_name},
                    }
                else:
                    existing = rrf_datas[chunk_id]
                    existing["retrieval_sources"].add(source_name)
                    for key, value in chunk.items():
                        if existing.get(key) in (None, "", []):
                            existing[key] = value

                # 分支内部已经记录了 query_0/query_1/... 的召回名次。
                # 在跨分支 RRF 时按检索分支归档，供 Rerank 后进行
                # 比较题/多事实题的子问题覆盖保护。
                subquery_ranks = rrf_datas[chunk_id].setdefault(
                    "subquery_ranks",
                    {},
                )
                for query_name, query_rank in self._query_branch_ranks(
                        chunk
                ).items():
                    subquery_ranks.setdefault(query_name, {})[
                        source_name
                    ] = query_rank

                rrf_datas[chunk_id][f"{source_name}_rank"] = rank

                #计算分数 RRF 公式: score += weight / (k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)+ weight / (_rrf_k + rank)

        # 按得分降序排序，截取前 _top_k 条
        sorted_results = sorted(
            [(rrf_datas[cid], score) for cid, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        if not _top_k or len(sorted_results) <= _top_k:
            return sorted_results

        selected = sorted_results[:_top_k]
        if min_exclusive_per_branch <= 0:
            return selected

        protected_ids = set()
        for _chunks, _weight, source_name in rrf_inputs:
            source_exclusive = [
                chunk_id
                for chunk_id, data in rrf_datas.items()
                if data.get("retrieval_sources") == {source_name}
            ]
            source_exclusive.sort(key=lambda cid: rrf_scores[cid], reverse=True)
            protected_ids.update(source_exclusive[:min_exclusive_per_branch])

        selected_ids = {item[0].get("chunk_id") for item in selected}
        for protected_id in protected_ids - selected_ids:
            replacement_index = next(
                (
                    index for index in range(len(selected) - 1, -1, -1)
                    if selected[index][0].get("chunk_id") not in protected_ids
                ),
                None,
            )
            if replacement_index is None:
                break
            selected[replacement_index] = (
                rrf_datas[protected_id],
                rrf_scores[protected_id],
            )
            selected_ids.add(protected_id)

        return sorted(selected, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _query_branch_ranks(chunk: Dict[str, Any]) -> Dict[str, int]:
        """提取分支内各子问题的有效召回名次。"""
        branch_ranks = chunk.get("branch_ranks") or {}
        if not isinstance(branch_ranks, dict):
            return {}
        return {
            query_name: int(query_rank)
            for query_name, query_rank in branch_ranks.items()
            if isinstance(query_name, str)
            and query_name.startswith("query_")
            and isinstance(query_rank, int)
            and query_rank > 0
        }
