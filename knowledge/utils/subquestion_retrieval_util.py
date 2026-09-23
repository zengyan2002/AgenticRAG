"""多子问题的调度、调用预算和证据归属。各召回分支只返回自己的状态键。"""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import time

from knowledge.utils.retrieval_result_util import merge_ranked_chunk_lists


GROUP_STATE_DEFAULTS = {
    "active_subquestion_indices": [],
    "active_retrieval_tasks": [],
    "subquestion_vector_chunks": {},
    "subquestion_bm25_chunks": {},
    "subquestion_hyde_chunks": {},
    "subquestion_rrf_chunks": {},
    "subquestion_rerank_candidates": {},
    "subquestion_evidence": {},
    "subquestion_retrieval_history": {},
    "subquestion_resolved_questions": {},
    "vector_retrieval_calls": 0,
    "bm25_retrieval_calls": 0,
    "hyde_retrieval_calls": 0,
    "vector_retrieval_errors": [],
    "bm25_retrieval_errors": [],
    "hyde_retrieval_errors": [],
    "agent_new_evidence_pairs": [],
}


def grouped_retrieval(state):
    plan = state.get("agent_plan") or {}
    return bool(state.get("agentic_active") and plan.get("retrieval_tasks")
                and len(plan.get("sub_questions") or []) > 1)


def reset_group_state(state):
    state.update(deepcopy(GROUP_STATE_DEFAULTS))


def schedule_tasks(state, tasks, config):
    """先给每个问题一次检索机会，再分配其它召回方式和查询变体。

    每项预留实际 Milvus 请求数；软路由最多两次请求。分支执行后按实际
    请求数结算，预留但未发生的请求不计费。依赖项由调用方先筛选。
    """
    remaining = max(0, config.agent_max_tool_calls - int(state.get("agent_tool_calls") or 0))
    cost = 2 if state.get("soft_filter_doc_ids") or (
        state.get("hard_filter_doc_ids") and not state.get("document_route_locked")
    ) else 1
    normalized = []
    for task in tasks:
        tools = [t for t in task.get("retrieval_tools", [])
                 if t in ("vector", "bm25", "hyde")
                 and (t != "bm25" or config.bm25_enabled)
                 and (t != "hyde" or config.agent_hyde_enabled)]
        tools.sort(key=lambda t: ("vector", "bm25", "hyde").index(t))
        if not tools:
            tools = ["vector"]
        queries = list(dict.fromkeys(q.strip() for q in task.get("search_queries", []) if q.strip()))[:2]
        operations = [(tool, query) for query in queries for tool in tools
                      if tool != "hyde" or query == queries[0]]
        normalized.append((task, operations))
    selected = {}
    for position in range(max((len(ops) for _, ops in normalized), default=0)):
        for task, operations in normalized:
            if position >= len(operations) or remaining < cost:
                continue
            index = task["subquestion_index"]
            item = selected.setdefault(index, {**task, "operations": []})
            tool, query = operations[position]
            item["operations"].append({"tool": tool, "query": query, "request_budget": cost})
            remaining -= cost
    state["active_retrieval_tasks"] = list(selected.values())
    state["active_subquestion_indices"] = list(selected)
    state["agent_selected_tools"] = list(dict.fromkeys(
        op["tool"] for task in selected.values() for op in task["operations"]
    ))
    state["retrieval_queries"] = list(dict.fromkeys(
        op["query"] for task in selected.values() for op in task["operations"]
    ))
    return bool(selected)


_REQUEST_SCOPE = ContextVar("subquestion_request_scope", default=None)


class RetrievalBudgetExceeded(RuntimeError):
    pass


class _BudgetedClient:
    def __init__(self, client, scope):
        self._client = client
        self._scope = scope

    def __getattr__(self, name):
        method = getattr(self._client, name)
        if name not in ("search", "hybrid_search"):
            return method

        def invoke(*args, **kwargs):
            scope = self._scope
            if scope["calls"] >= scope["limit"] or time.monotonic() >= scope["deadline"]:
                raise RetrievalBudgetExceeded("检索请求预算或时间预算已耗尽")
            scope["calls"] += 1  # 失败的远程请求同样占用预算。
            return method(*args, **kwargs)
        return invoke


def budgeted_retrieval_client(client):
    scope = _REQUEST_SCOPE.get()
    return _BudgetedClient(client, scope) if scope is not None else client


def grouped_scope_locked():
    scope = _REQUEST_SCOPE.get()
    return bool(scope and scope.get("locked"))


def run_grouped_branch(node, state, tool, output_field, limit):
    """复用单查询检索，按子问题合并变体；失败隔离在单次查询内。

    图本身并行执行三种召回方式；每个分支顺序执行，避免再嵌套线程池。
    """
    groups, errors, calls, fallback = {}, [], 0, False
    started = state.get("agent_started_at") or time.monotonic()
    deadline = started + node.config.agent_timeout_seconds
    for task in state.get("active_retrieval_tasks") or []:
        index = task["subquestion_index"]
        results = []
        for op in task.get("operations", []):
            if op["tool"] != tool:
                continue
            if time.monotonic() >= deadline:
                errors.append({"subquestion_index": index, "error": "timeout_reached"})
                break
            scope = {"calls": 0, "limit": op["request_budget"], "deadline": deadline,
                     "locked": bool(state.get("document_route_locked"))}
            token = _REQUEST_SCOPE.set(scope)
            try:
                local = {**state, "rewritten_query": op["query"],
                         "retrieval_queries": [op["query"]], "agent_selected_tools": [tool]}
                result = node._process_ungrouped(local)
                results.append((op["query"], result.get(output_field) or [], 1.0))
                fallback |= bool(result.get(f"{tool}_route_fallback"))
            except Exception as exc:
                node.logger.warning("子问题 %s 的 %s 检索失败: %s", index, tool, exc)
                errors.append({"subquestion_index": index, "error": str(exc)})
            finally:
                calls += scope["calls"]
                _REQUEST_SCOPE.reset(token)
        groups[str(index)] = merge_ranked_chunk_lists(results, limit, min_per_list=1)
    return {
        f"subquestion_{tool}_chunks": groups,
        f"{tool}_retrieval_calls": calls,
        f"{tool}_retrieval_errors": errors,
        output_field: [],
        f"{tool}_route_fallback": fallback,
    }


def merge_evidence_groups(groups, questions):
    """轮流取各组证据、全局去重；跨问题不比较分数，不把归属当作支持证明。"""
    merged = {}
    keys = sorted(groups, key=int)
    for rank in range(max((len(groups[k]) for k in keys), default=0)):
        for key in keys:
            if rank >= len(groups[key]):
                continue
            doc = groups[key][rank]
            cid = str(doc["chunk_id"])
            if cid not in merged:
                merged[cid] = {**deepcopy(doc), "source": "local",
                               "retrieved_for_subquestions": [], "subquestion_scores": {},
                               "subquestion_texts": {}}
            target = merged[cid]
            index = int(key)
            target["retrieved_for_subquestions"].append(index)
            target["subquestion_scores"][key] = doc.get("rerank_score")
            target["subquestion_texts"][key] = questions[index]
    return list(merged.values())
