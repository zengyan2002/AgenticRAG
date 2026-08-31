from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.milvus_util import (
    _doc_ids_filter,
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)
from knowledge.utils.retrieval_result_util import (
    CHUNK_OUTPUT_FIELDS,
    hit_to_chunk,
    merge_ranked_chunk_lists,
)


class VectorSearchNode(BaseNode):
    name = 'vector_search_node'

    def process(self, state: QueryGraphState) -> dict[str, list]:
        if (
            state.get("agentic_active")
            and "vector" not in (state.get("agent_selected_tools") or [])
        ):
            return {"embedding_chunks": [], "vector_route_fallback": False}

        # 1 参数校验
        retrieval_queries, hard_filter_doc_ids, soft_filter_doc_ids = self._valid_state(state)

        # 2 查询向量化
        # 2.1 构建bge-m3模型的客户端
        try:
            bgem3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to BGE M3. Reason: {e}")
            return {"embedding_chunks": []}

        # 2.2 创建milvus客户端
        try:
            milvus_client = StorageClients.get_milvus()
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to Milvus. Reason: {e}")
            return {"embedding_chunks": []}

        # 3.3 将问题向量化
        try:
            # 调用模型将问题向量化
            embedding_result = bgem3_client.encode(
                retrieval_queries,
                return_dense=True,
                return_sparse=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to invoke bge-m3 model. Reason: {e}")
            return {"embedding_chunks": []}

        # 取出稠密向量和稀疏向量
        query_results = []
        route_fallback = False
        for index, query in enumerate(retrieval_queries):
            dense_vector = embedding_result["dense_vecs"][index].tolist()
            sparse_vector = {
                int(token_id): float(weight)
                for token_id, weight in dict(
                    embedding_result["lexical_weights"][index]
                ).items()
            }
            chunks, fallback = self._search_one_query(
                milvus_client,
                dense_vector,
                sparse_vector,
                hard_filter_doc_ids,
                soft_filter_doc_ids,
            )
            route_fallback = route_fallback or fallback
            query_results.append((f"query_{index}", chunks, 1.0))

        return {
            "embedding_chunks": merge_ranked_chunk_lists(
                query_results,
                self.config.embedding_search_limit,
                min_per_list=1 if len(query_results) > 1 else 0,
            ),
            "vector_route_fallback": route_fallback,
        }

    def _search_one_query(
            self,
            milvus_client,
            dense_vector,
            sparse_vector,
            hard_filter_doc_ids,
            soft_filter_doc_ids,
    ):
        limit = self.config.embedding_search_limit

        def search(doc_ids=None):
            expr, expr_params = _doc_ids_filter(doc_ids or [])
            requests = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                expr_params=expr_params,
                limit=limit,
            )
            response = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=requests,
                ranker_weights=(
                    self.config.hybrid_dense_weight,
                    self.config.hybrid_sparse_weight,
                ),
                limit=limit,
                output_fields=CHUNK_OUTPUT_FIELDS,
            )
            return [hit_to_chunk(hit) for hit in (response[0] if response else [])]

        if hard_filter_doc_ids:
            filtered = search(hard_filter_doc_ids)
            if filtered:
                return filtered, False
            self.logger.warning("doc_id精确范围内无结果，自动回退全库原问题检索")
            return search(), True

        global_chunks = search()
        if not soft_filter_doc_ids:
            return global_chunks, False
        candidate_chunks = search(soft_filter_doc_ids)
        return merge_ranked_chunk_lists(
            [
                ("global", global_chunks, 1.0),
                ("soft_route", candidate_chunks, self.config.soft_route_weight),
            ],
            limit,
        ), False

    def _valid_state(self, state: QueryGraphState):

        # 获取theme_names和rewritten_query字段
        hard_filter_doc_ids = state.get("hard_filter_doc_ids") or []
        soft_filter_doc_ids = state.get("soft_filter_doc_ids") or []
        rewritten_query = state.get("rewritten_query", "")
        retrieval_queries = state.get("retrieval_queries") or [rewritten_query]

        # 校验参数
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name,
                                  field_name="rewritten_query",
                                  expected_type=str)

        return retrieval_queries, hard_filter_doc_ids, soft_filter_doc_ids
