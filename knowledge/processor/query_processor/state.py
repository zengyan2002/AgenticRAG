"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import Any, TypedDict, List, Literal
import copy


class QueryGraphState(TypedDict):
    """
    Represents the state of our query graph.
    Attributes:
    各个属性的结构
    """
    session_id: str # 会话ID
    task_id:str # 任务ID
    original_query: str # 原始查询
    display_query: str # 会话中展示和持久化的用户查询
    query_type: Literal["text", "image", "multimodal"] # 查询类型
    retrieval_query: str # 文本或图片经语义增强后用于检索的查询
    query_image_bytes: bytes # 用户上传的查询图片
    query_image_mime_type: str # 查询图片的 MIME 类型
    image_query_description: str # VLM 生成的图片语义描述
    embedding_chunks: list # 已向量化的切片
    hyde_embedding_chunks: list # 已向量化的假设性问题切片
    bm25_chunks: list # BM25关键词检索切片
    rrf_chunks: list # rrf排序后的切片
    rerank_candidates: list  # BGE-Reranker 完整排序候选
    reranked_docs: list  # 排序后的文档
    conflict_judge_triggered: bool  # 是否触发选择性 LLM 仲裁
    conflict_judge_decisions: list  # 冲突候选仲裁记录
    expanded_docs: list  # 基于Rerank锚点扩展后的层级上下文
    final_context_chunk_ids: List[str]  # 最终答案上下文实际使用的切片
    answer: str #答案
    theme_names: List[str] # 兼容 Milvus 现有 theme_name 字段
    document_mentions: List[str]  # 用户明确提到的文档/实体/型号
    document_candidates: list  # 文档注册表召回的候选
    selected_documents: list  # 已精确确认的文档档案
    hard_filter_doc_ids: List[str]  # 只有精确唯一匹配时才设置
    soft_filter_doc_ids: List[str]  # 多候选软路由，仅提升优先级
    document_route_mode: Literal["global", "soft", "hard"]
    document_route_locked: bool  # 用户明确限定文档时禁止跨范围扩检
    document_route_reason: str  # 当前文档路由的可观测原因
    document_route_scope_history: list  # 证据完善闭环中的范围变化
    vector_route_fallback: bool
    hyde_route_fallback: bool
    bm25_route_fallback: bool
    rewritten_query: str  #重写答案
    retrieval_queries: List[str]  # 原问题及比较/多事实子问题
    query_decomposed: bool
    agentic_active: bool  # 当前查询是否进入 Agentic 闭环
    agentic_route_reason: str  # 进入或跳过 Agentic 的原因
    agent_plan: dict[str, Any]  # Planner 的结构化计划
    agent_selected_tools: List[str]  # 本轮允许执行的检索工具
    agent_iteration: int  # 当前检索轮次，从1开始
    agent_tool_calls: int  # 计划内累计检索工具调用数
    agent_started_at: float  # Agentic 执行起始时间戳
    agent_current_query: str  # 当前轮供 HyDE 等节点使用的查询
    agent_retrieval_history: List[str]  # 已执行过的检索问题
    agent_evidence_pool: list  # 跨轮保留的证据候选
    agent_new_chunk_ids: List[str]  # 当前轮相对历史证据新增的切片
    coverage_missing_subquestion_indices: List[int]
    coverage_protected_chunk_ids: List[str]
    evidence_evaluation: dict[str, Any]  # 证据覆盖评审结果
    evidence_sufficient: bool
    missing_aspects: List[str]
    evidence_next_action: str
    agent_stop_reason: str
    history: list   # 历史对话
    is_stream: bool # 是否流式输出


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",               # 会话ID
    "task_id": "",               # 任务ID
    "original_query": "",           # 原始查询
    "display_query": "",            # 会话展示文本
    "query_type": "text",           # 查询类型
    "retrieval_query": "",          # 用于主题识别与检索的查询
    "query_image_bytes": b"",       # 查询图片原始数据
    "query_image_mime_type": "",    # 查询图片 MIME 类型
    "image_query_description": "",  # 图片语义描述
    "embedding_chunks": [],         # 已向量化的切片
    "hyde_embedding_chunks": [],    # 已向量化的假设性问题切片
    "bm25_chunks": [],              # BM25关键词检索切片
    "rrf_chunks": [],               # rrf排序后的切片
    "rerank_candidates": [],        # BGE-Reranker 完整排序候选
    "reranked_docs": [],            # 排序后的文档
    "conflict_judge_triggered": False,
    "conflict_judge_decisions": [],
    "expanded_docs": [],            # 层级上下文扩展结果
    "final_context_chunk_ids": [],  # 最终打包进答案上下文的切片
    "answer": "",                   # 答案
    "theme_names": [],               # 兼容 Milvus 现有字段
    "document_mentions": [],        # 明确文档/实体/型号表达
    "document_candidates": [],      # 软路由候选
    "selected_documents": [],      # 唯一精确文档
    "hard_filter_doc_ids": [],      # 硬过滤doc_id
    "soft_filter_doc_ids": [],      # 软路由候选doc_id
    "document_route_mode": "global",  # 默认全库检索
    "document_route_locked": False,
    "document_route_reason": "not_routed",
    "document_route_scope_history": [],
    "vector_route_fallback": False,
    "hyde_route_fallback": False,
    "bm25_route_fallback": False,
    "rewritten_query": "",          # 重写查询
    "retrieval_queries": [],         # 原问题及子问题
    "query_decomposed": False,
    "agentic_active": False,
    "agentic_route_reason": "disabled",
    "agent_plan": {},
    "agent_selected_tools": [],
    "agent_iteration": 0,
    "agent_tool_calls": 0,
    "agent_started_at": 0.0,
    "agent_current_query": "",
    "agent_retrieval_history": [],
    "agent_evidence_pool": [],
    "agent_new_chunk_ids": [],
    "coverage_missing_subquestion_indices": [],
    "coverage_protected_chunk_ids": [],
    "evidence_evaluation": {},
    "evidence_sufficient": False,
    "missing_aspects": [],
    "evidence_next_action": "",
    "agent_stop_reason": "",
    "history": [],                  # 历史对话
    "is_stream": False,             # 是否流式输出 (默认设为 False)
}

def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。

    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state
