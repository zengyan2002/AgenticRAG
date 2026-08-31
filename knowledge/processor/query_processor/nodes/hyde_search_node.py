from aiohttp import ClientError
import time

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError, LLMError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE, HYDE_SYSTEM_PROMPT_TEMPLATE
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


class HydeSearchNode(BaseNode):
    name = "hyde_search_node"
    def process(self, state: QueryGraphState) -> dict[str, list]:

        if (
            state.get("agentic_active")
            and (
                not self.config.agent_hyde_enabled
                or "hyde" not in (state.get("agent_selected_tools") or [])
            )
        ):
            return {"hyde_embedding_chunks": [], "hyde_route_fallback": False}

        #1 参数校验
        rewritten_query, hard_filter_doc_ids, soft_filter_doc_ids, document_context = self._valid_state(state)
        #2 利用LLM生成假设性文档
        hypothetical_answer = self._generate_hypothetical_document(
            rewritten_query,
            document_context,
        )

        #3 将问题和假设性文档进行拼接，形成拼接文本
        hypothetical_document = "\n".join([rewritten_query,hypothetical_answer])

        #4 将拼接文本向量化，然后在向量数据库中进行查询，得到查询结果
        #4.1 获取bge-m3模型和milvus客户端
        try:
            bgem3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to BGEM3. Reason: {e}")
            return {"hyde_embedding_chunks": []}

        try:
            milvus_client = StorageClients.get_milvus()
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to Milvus. Reason: {e}")
            return {"hyde_embedding_chunks": []}

        #4.2 调用bge_m3模型生成向量
        try:
            embedding_result = bgem3_client.encode([hypothetical_document],return_dense=True,return_sparse=True)
        except Exception as e:
            self.logger.error(f"Failed to invoke bge-m3 model. Reason: {e}")
            return {"hyde_embedding_chunks": []}

        #4.3 获取稠密向量和稀疏向量
        dense_vector = embedding_result["dense_vecs"][0].tolist()
        sparse_vector = embedding_result["lexical_weights"][0]
        sparse_vector = {
            int(token_id): float(weight)
            for token_id, weight in dict(sparse_vector).items()}

        def search(doc_ids=None):
            expr, expr_params = _doc_ids_filter(doc_ids or [])
            if not self.config.hyde_use_sparse:
                response = milvus_client.search(
                    collection_name=self.config.chunks_collection,
                    data=[dense_vector],
                    anns_field="dense_vector",
                    filter=expr,
                    filter_params=expr_params,
                    limit=self.config.hyde_search_limit,
                    output_fields=CHUNK_OUTPUT_FIELDS,
                    search_params={"metric_type": "COSINE"},
                )
            else:
                requests = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    expr=expr,
                    expr_params=expr_params,
                    limit=self.config.hyde_search_limit,
                )
                response = execute_hybrid_search_query(
                    milvus_client=milvus_client,
                    collection_name=self.config.chunks_collection,
                    search_requests=requests,
                    ranker_weights=(
                        self.config.hybrid_dense_weight,
                        self.config.hybrid_sparse_weight,
                    ),
                    limit=self.config.hyde_search_limit,
                    output_fields=CHUNK_OUTPUT_FIELDS,
                )
            return [hit_to_chunk(hit) for hit in (response[0] if response else [])]

        route_fallback = False
        if hard_filter_doc_ids:
            chunks = search(hard_filter_doc_ids)
            if not chunks:
                route_fallback = True
                self.logger.warning("doc_id精确范围内无结果，自动回退全库HyDE检索")
                chunks = search()
        else:
            chunks = search()
            if soft_filter_doc_ids:
                candidate_chunks = search(soft_filter_doc_ids)
                chunks = merge_ranked_chunk_lists(
                    [
                        ("global", chunks, 1.0),
                        ("soft_route", candidate_chunks, self.config.soft_route_weight),
                    ],
                    self.config.hyde_search_limit,
                )

        return {
            "hyde_embedding_chunks": chunks,
            "hyde_route_fallback": route_fallback,
        }

    def _valid_state(self, state: QueryGraphState):

        # 获取theme_names和rewritten_query字段
        hard_filter_doc_ids = state.get("hard_filter_doc_ids") or []
        soft_filter_doc_ids = state.get("soft_filter_doc_ids") or []
        if state.get("agentic_active"):
            queries = [
                query.strip()
                for query in (state.get("retrieval_queries") or [])
                if isinstance(query, str) and query.strip()
            ]
            # 显式开启 Agentic HyDE 时也只处理单个 focused query，
            # 禁止把多个缺失事实拼成一份容易漂移的假设文档。
            focused_queries = queries[1:] or queries
            rewritten_query = (
                focused_queries[:self.config.agent_hyde_max_queries] or [""]
            )[0]
        else:
            rewritten_query = state.get("rewritten_query") or ""
        selected_documents = state.get("selected_documents") or []
        document_context = [
            document.get("canonical_title")
            or document.get("primary_subject")
            for document in selected_documents
            if isinstance(document, dict)
        ]
        if not document_context:
            document_context = (
                state.get("document_mentions")
                or state.get("theme_names")
                or []
            )

        # 校验参数
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name,
                                  field_name="rewritten_query",
                                  expected_type=str)

        return rewritten_query, hard_filter_doc_ids, soft_filter_doc_ids, document_context

    def _generate_hypothetical_document(self, rewritten_query, document_context) -> str:

        #获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(
                response_format=False,
                role="fast",
                thinking="none",
            )
        except ConnectionError as e:
            #连接不上客户端就返回空字符串
            self.logger.error(f"Failed to connect to LLM. Reason: {e}")
            return ""

        #生成提示词
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            theme_names="、".join(document_context),
            rewritten_query=rewritten_query,
        )
        system_prompt = HYDE_SYSTEM_PROMPT_TEMPLATE
        try:
            #调用大模型
            started_at = time.perf_counter()
            try:
                llm_result = llm_client.invoke([
                    ("system", system_prompt),
                    ("user", user_prompt)
                ])
            finally:
                self.logger.info(
                    "LLM latency: node_name=hyde model_name=%s "
                    "thinking_level=none elapsed_ms=%.1f",
                    getattr(llm_client, "model_name", "unknown"),
                    (time.perf_counter() - started_at) * 1000,
                )
        except LLMError as e:
            self.logger.error(f"Failed to invoke the LLM. Reason: {e}")
            return ""

        #返回结果
        return llm_result.content
