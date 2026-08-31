"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_processor.nodes.agent_router_node import AgentRouterNode
from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_processor.nodes.bm25_search_node import BM25SearchNode
from knowledge.processor.query_processor.nodes.context_expansion_node import ContextExpansionNode
from knowledge.processor.query_processor.nodes.conflict_judge_node import ConflictJudgeNode
from knowledge.processor.query_processor.nodes.document_route_node import DocumentRouteNode
from knowledge.processor.query_processor.nodes.evidence_evaluator_node import EvidenceEvaluatorNode
from knowledge.processor.query_processor.nodes.hyde_search_node import HydeSearchNode
from knowledge.processor.query_processor.nodes.image_query_node import ImageQueryNode
from knowledge.processor.query_processor.nodes.query_type_node import QueryTypeNode
from knowledge.processor.query_processor.nodes.planner_node import PlannerNode
from knowledge.processor.query_processor.nodes.replan_node import ReplanNode
from knowledge.processor.query_processor.nodes.rerank_node import RerankNode
from knowledge.processor.query_processor.nodes.rrf_merge_node import RRFMergeNode
from knowledge.processor.query_processor.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_processor.state import QueryGraphState

RETRIEVAL_NODES = [
    "vector_search_node",
    "hyde_search_node",
    "bm25_search_node",
]

# 路由函数
def route_after_document_route(state: QueryGraphState) -> str | list[str]:
    answer = state.get("answer") or ""

    if answer.strip():
        return "answer_output_node"
    return "agent_router_node"


def route_after_agent_router(state: QueryGraphState) -> str | list[str]:
    if state.get("agentic_active"):
        return "planner_node"
    return RETRIEVAL_NODES


def route_after_conflict_judge(state: QueryGraphState) -> str:
    if state.get("agentic_active"):
        return "evidence_evaluator_node"
    return "context_expansion_node"


def route_after_evidence_evaluator(state: QueryGraphState) -> str:
    if state.get("answer"):
        return "answer_output_node"
    if state.get("agent_stop_reason"):
        return "context_expansion_node"
    return "replan_node"


def route_after_replan(state: QueryGraphState) -> str | list[str]:
    if state.get("agent_stop_reason"):
        return "context_expansion_node"
    return RETRIEVAL_NODES


def route_after_query_type(state: QueryGraphState) -> str:
    query_type = state.get("query_type")
    if query_type == "text":
        return "document_route_node"
    if query_type in {"image", "multimodal"}:
        return "image_query_node"
    raise ValueError(f"Unsupported query type: {query_type}")


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    选择性 Agentic 扩展::

        document_route → agent_router
                             │
              ┌── 简单查询 ─────┐
              │                │
              └── 复杂查询 → planner
                               │
                               ▼
                vector / hyde / bm25
                               │
                               ▼
              RRF → Reranker → conflict_judge
                               │
                   evidence_evaluator
                         ┌──┴──┐
                      充分     不足
                       │        ▼
                       │      replan
                       │        │
                       │        └──→ 检索分支
                       ▼
              context_expansion → answer

    Agentic 模式默认关闭；hybrid 模式仅对比较、多事实和多跳问题
    触发。Planner 只能选择 vector、hyde、bm25 三种受控检索工具；
    Evaluator 只评审证据覆盖，不生成答案。补检受轮次、工具数和超时上限
    保护；控制节点异常时保留已有检索结果，回退确定性流程。

    Returns:
        编译后的 StateGraph 实例。

    确定性基础流程（Agentic 关闭或简单查询）::

        START
          │
          ▼
        query_type_node
        输入类型识别
          │
          ├── text ───────────────────────────────────────┐
          │                                               │
          └── image / multimodal                          │
                       │                                  │
                       ▼                                  │
                 image_query_node                         │
                 生成图片语义查询                          │
                       │                                  │
                       └────────────────┬─────────────────┘
                                        ▼
                              document_route_node
                    查询改写、问题拆分与可回退文档软路由
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
               已有提前答案                             需要知识检索
                    │                                       │
                    │                 ┌─────────────────────┼─────────────────────┐
                    │                 │                     │                     │
                    │                 ▼                     ▼                     ▼
                    │       vector_search_node       hyde_search_node      bm25_search_node
                    │       原问题 BGE-M3 混合检索    HyDE 假设文档检索     BM25 关键词检索
                    │       Dense + Sparse            默认 Dense            稀疏检索
                    │                 │                     │                     │
                    │                 └─────────────────────┼─────────────────────┘
                    │                                       ▼
                    │                                rrf_merge_node
                    │                                 加权 RRF 融合
                    │                                       │
                    │                                       ▼
                    │                                  rerank_node
                    │                                BGE-Reranker 重排
                    │                                       │
                    │                                       ▼
                    │                             conflict_judge_node
                    │                         高分歧边界候选 LLM 仲裁
                    │                                       │
                    │                                       ▼
                    │                            context_expansion_node
                    │                          父块、章节路径及相邻块扩展
                    │                                       │
                    └───────────────────────┬───────────────┘
                                            ▼
                                   answer_output_node
                                   证据生成与 SSE 输出
                                            │
                                            ▼
                                           END

    文档软路由模式：
        - hard：标题、别名或型号唯一命中时限定 doc_id；无结果回退全库。
        - soft：存在多个逻辑候选时，全库检索并对候选文档结果加权。
        - global：没有明确文档指向时直接检索全库。

    检索分支说明：
        - vector_search_node 同时使用 BGE-M3 稠密向量和稀疏向量。
        - hyde_search_node 默认使用稠密向量，可通过 HYDE_USE_SPARSE
          开启稠密/稀疏混合检索。
        - bm25_search_node 使用 Milvus 原生 BM25；不可用时返回空结果，
          主链路自动退化为原问题混合检索与 HyDE 两路融合。

    """

    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)  # type:ignore

    # 2. 实例化节点
    nodes = {
        "query_type_node": QueryTypeNode(),
        "image_query_node": ImageQueryNode(),
        "document_route_node": DocumentRouteNode(),
        "agent_router_node": AgentRouterNode(),
        "planner_node": PlannerNode(),
        "vector_search_node": VectorSearchNode(),
        "hyde_search_node": HydeSearchNode(),
        "bm25_search_node": BM25SearchNode(),
        "rrf_merge_node": RRFMergeNode(),
        "rerank_node": RerankNode(),
        "conflict_judge_node": ConflictJudgeNode(),
        "evidence_evaluator_node": EvidenceEvaluatorNode(),
        "replan_node": ReplanNode(),
        "context_expansion_node": ContextExpansionNode(),
        "answer_output_node": AnswerOutputNode(),
    }

    # 3. 添加节点
    for node_name, node in nodes.items():
        workflow.add_node(node_name, node)

    # 4. 添加边
    workflow.add_edge(START, "query_type_node")
    workflow.add_conditional_edges(
        "query_type_node",
        route_after_query_type,
        ["image_query_node", "document_route_node"],
    )
    workflow.add_edge("image_query_node", "document_route_node")
    workflow.add_conditional_edges(
        "document_route_node",
        route_after_document_route,
        [
            "answer_output_node",
            "agent_router_node",
        ],
    )
    workflow.add_conditional_edges(
        "agent_router_node",
        route_after_agent_router,
        ["planner_node", *RETRIEVAL_NODES],
    )
    for retrieval_node in RETRIEVAL_NODES:
        workflow.add_edge("planner_node", retrieval_node)
    workflow.add_edge(
        RETRIEVAL_NODES,
        "rrf_merge_node",
    )
    workflow.add_edge("rrf_merge_node", "rerank_node")
    workflow.add_edge("rerank_node", "conflict_judge_node")
    workflow.add_conditional_edges(
        "conflict_judge_node",
        route_after_conflict_judge,
        ["evidence_evaluator_node", "context_expansion_node"],
    )
    workflow.add_conditional_edges(
        "evidence_evaluator_node",
        route_after_evidence_evaluator,
        ["answer_output_node", "context_expansion_node", "replan_node"],
    )
    workflow.add_conditional_edges(
        "replan_node",
        route_after_replan,
        ["context_expansion_node", *RETRIEVAL_NODES],
    )
    workflow.add_edge("context_expansion_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    # 5. 编译并返回
    return workflow.compile()

query_app = create_query_graph()
