"""Milvus原生BM25第三路检索；不可用时安全退化为空分支。"""

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.milvus_util import _doc_ids_filter
from knowledge.utils.retrieval_result_util import (
    CHUNK_OUTPUT_FIELDS,
    hit_to_chunk,
    merge_ranked_chunk_lists,
)


class BM25SearchNode(BaseNode):
    name = "bm25_search_node"

    def process(self, state: QueryGraphState) -> dict[str, list | bool]:
        if (
            state.get("agentic_active")
            and "bm25" not in (state.get("agent_selected_tools") or [])
        ):
            return {"bm25_chunks": [], "bm25_route_fallback": False}
        if not self.config.bm25_enabled:
            return {"bm25_chunks": [], "bm25_route_fallback": False}

        rewritten_query = str(state.get("rewritten_query") or "").strip()
        queries = state.get("retrieval_queries") or [rewritten_query]
        queries = [query.strip() for query in queries if isinstance(query, str) and query.strip()]
        if not queries:
            return {"bm25_chunks": [], "bm25_route_fallback": False}

        hard_ids = state.get("hard_filter_doc_ids") or []
        soft_ids = state.get("soft_filter_doc_ids") or []
        try:
            client = StorageClients.get_milvus()
            query_results = []
            route_fallback = False
            for index, query in enumerate(queries):
                chunks, fallback = self._search_one(
                    client,
                    query,
                    hard_ids,
                    soft_ids,
                )
                route_fallback = route_fallback or fallback
                query_results.append((f"query_{index}", chunks, 1.0))
            return {
                "bm25_chunks": merge_ranked_chunk_lists(
                    query_results,
                    self.config.bm25_search_limit,
                    min_per_list=1 if len(query_results) > 1 else 0,
                ),
                "bm25_route_fallback": route_fallback,
            }
        except Exception as exc:
            # BM25是补充召回路，任何索引/版本/连接问题都不应中断主问答。
            self.logger.warning("BM25分支不可用，退化为原两路检索: %s", exc)
            return {"bm25_chunks": [], "bm25_route_fallback": False}

    def _search_one(self, client, query, hard_ids, soft_ids):
        limit = self.config.bm25_search_limit

        def search(doc_ids=None):
            expr, expr_params = _doc_ids_filter(doc_ids or [])
            response = client.search(
                collection_name=self.config.bm25_collection,
                data=[query],
                anns_field="bm25_sparse_vector",
                filter=expr,
                filter_params=expr_params,
                limit=limit,
                output_fields=CHUNK_OUTPUT_FIELDS,
                search_params={"metric_type": "BM25"},
            )
            return [hit_to_chunk(hit) for hit in (response[0] if response else [])]

        if hard_ids:
            filtered = search(hard_ids)
            if filtered:
                return filtered, False
            self.logger.warning("doc_id精确范围内无BM25结果，自动回退全库")
            return search(), True

        global_chunks = search()
        if not soft_ids:
            return global_chunks, False
        candidate_chunks = search(soft_ids)
        return merge_ranked_chunk_lists(
            [
                ("global", global_chunks, 1.0),
                ("soft_route", candidate_chunks, self.config.soft_route_weight),
            ],
            limit,
        ), False
