"""只针对证据评审指出的缺失项生成下一轮检索问题。"""

from __future__ import annotations

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.document_identity_util import unique_strings
from knowledge.utils.subquestion_retrieval_util import grouped_retrieval, schedule_tasks


class ReplanNode(BaseNode):
    name = "replan_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 ReplanNode 的核心处理流程。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        next_action = str(
            state.get("evidence_next_action")
            or (state.get("evidence_evaluation") or {}).get("next_action")
            or "rewrite_missing"
        )
        if next_action == "ask_user":
            state["agent_stop_reason"] = "need_user_clarification"
            return state
        if next_action == "answer":
            state["agent_stop_reason"] = "evidence_sufficient"
            return state

        scope_before = self._scope_snapshot(state)
        if next_action == "expand_scope":
            self._expand_scope(state)
        scope_after = self._scope_snapshot(state)
        state.setdefault("document_route_scope_history", []).append({
            "iteration": int(state.get("agent_iteration") or 0) + 1,
            "action": next_action,
            "scope_before": scope_before,
            "scope_after": scope_after,
        })

        if grouped_retrieval(state):
            return self._replan_subquestions(state)

        missing = unique_strings(
            state.get("missing_aspects") or [],
            max_items=self.config.agent_max_subquestions,
        )
        history = list(state.get("agent_retrieval_history") or [])
        history_set = set(history)
        new_queries = []
        for query in missing:
            candidate = query
            if candidate in history_set:
                candidate = self._refine_duplicate_query(query, state)
            if candidate and candidate not in history_set and candidate not in new_queries:
                new_queries.append(candidate)
        if not new_queries:
            state["agent_stop_reason"] = "no_new_queries"
            return state

        if not state.get("agent_evidence_pool"):
            state["agent_evidence_pool"] = [
                dict(document)
                for document in (state.get("reranked_docs") or [])
                if isinstance(document, dict)
            ][:self.config.agent_evidence_pool_max]
        state["agent_iteration"] = int(state.get("agent_iteration") or 0) + 1
        state["agent_tool_calls"] = (
            int(state.get("agent_tool_calls") or 0)
            + len(state.get("agent_selected_tools") or [])
        )
        state["agent_retrieval_history"] = [*history, *new_queries]
        state["retrieval_queries"] = new_queries
        state["agent_current_query"] = "；".join(new_queries)
        state["query_decomposed"] = len(new_queries) > 1
        state["agent_stop_reason"] = ""

        for key, value in {
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
            "bm25_chunks": [],
            "rrf_chunks": [],
            "rerank_candidates": [],
            "reranked_docs": [],
            "expanded_docs": [],
            "conflict_judge_decisions": [],
            "conflict_judge_triggered": False,
            "vector_route_fallback": False,
            "hyde_route_fallback": False,
            "bm25_route_fallback": False,
        }.items():
            state[key] = value
        return state

    def _replan_subquestions(self, state):
        """仅重新检索缺失的答题维度；依赖项需要已覆盖证据及解析后的新查询。"""
        plan = state["agent_plan"]
        questions = plan["sub_questions"]
        evaluation = state.get("evidence_evaluation") or {}
        missing = state.get("coverage_missing_subquestion_indices") or []
        covered = set(evaluation.get("covered_sub_questions") or [])
        followups = {}
        resolved = dict(state.get("subquestion_resolved_questions") or {})
        explicit_resolutions = set()
        for task in evaluation.get("followup_tasks") or []:
            index = task.get("subquestion_index")
            if type(index) is int and index in missing:
                followups.setdefault(index, []).extend(task.get("search_queries") or [])
                if str(task.get("resolved_question") or "").strip():
                    resolved[str(index)] = str(task["resolved_question"]).strip()
                    explicit_resolutions.add(index)
        history = state.get("subquestion_retrieval_history") or {}
        tasks = []
        for task in plan["retrieval_tasks"]:
            index = task["subquestion_index"]
            if index not in missing or not set(task.get("depends_on") or []).issubset(covered):
                continue
            # 依赖问题的实体必须由模型依据上游证据补全；不能拿含“该方法”的模板去搜。
            if task.get("depends_on") and not followups.get(index):
                continue
            old = set(history.get(str(index), []))
            queries = unique_strings(followups.get(index) or task["search_queries"], max_items=2)
            queries = [q if q not in old else self._refine_duplicate_query(q, state) for q in queries]
            queries = [q for q in queries if q and q not in old]
            if queries:
                tasks.append({**task, "search_queries": queries})
                if task.get("depends_on") and index not in explicit_resolutions:
                    resolved[str(index)] = queries[0]
        # 未执行的问题先拿到预算，避免反复补查前一题耗尽其它题的机会。
        tasks.sort(key=lambda t: bool(history.get(str(t["subquestion_index"]))))
        if not tasks:
            state["agent_stop_reason"] = "no_ready_subquestion_queries"
            return state
        if not schedule_tasks(state, tasks, self.config):
            state["agent_stop_reason"] = "max_tool_calls_reached"
            return state
        state["agent_iteration"] = int(state.get("agent_iteration") or 0) + 1
        state["subquestion_resolved_questions"] = resolved
        state["agent_current_query"] = "；".join(state["retrieval_queries"])
        state["agent_stop_reason"] = ""
        state["expanded_docs"] = []
        # 保留 subquestion_evidence；本轮结果由各分支完全替换，避免读到上一轮候选。
        for tool in ("vector", "bm25", "hyde"):
            state[f"subquestion_{tool}_chunks"] = {}
            state[f"{tool}_retrieval_calls"] = 0
            state[f"{tool}_retrieval_errors"] = []
        state["subquestion_rrf_chunks"] = {}
        return state

    @staticmethod
    def _scope_snapshot(state: QueryGraphState) -> dict:
        """生成当前文档检索范围的稳定快照。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        return {
            "mode": state.get("document_route_mode") or "global",
            "hard_filter_doc_ids": list(state.get("hard_filter_doc_ids") or []),
            "soft_filter_doc_ids": list(state.get("soft_filter_doc_ids") or []),
            "locked": bool(state.get("document_route_locked")),
        }

    @staticmethod
    def _expand_scope(state: QueryGraphState) -> None:
        """非锁定范围按 hard → soft → global 逐级扩大。"""
        if state.get("document_route_locked"):
            state["document_route_reason"] = "locked_scope_preserved"
            return

        mode = state.get("document_route_mode") or "global"
        if mode == "hard":
            previous_ids = unique_strings(
                [
                    *(state.get("hard_filter_doc_ids") or []),
                    *(
                        document.get("doc_id")
                        for document in (state.get("document_candidates") or [])
                        if isinstance(document, dict)
                    ),
                ]
            )
            state["hard_filter_doc_ids"] = []
            state["soft_filter_doc_ids"] = previous_ids
            state["document_route_mode"] = "soft"
            state["document_route_reason"] = "evidence_expand_hard_to_soft"
            return

        if mode == "soft":
            state["hard_filter_doc_ids"] = []
            state["soft_filter_doc_ids"] = []
            state["document_route_mode"] = "global"
            state["document_route_reason"] = "evidence_expand_soft_to_global"
            return

        state["document_route_mode"] = "global"
        state["document_route_reason"] = "evidence_scope_already_global"

    @staticmethod
    def _refine_duplicate_query(query: str, state: QueryGraphState) -> str:
        """对“已搜过但证据仍不足”的子问题增加检索约束。

        不改变缺失事实，只补充文档范围和直接证据要求，避免将
        同一条查询原样再执行一次。
        """
        plan = state.get("agent_plan") or {}
        hints = unique_strings(plan.get("document_hints") or [], max_items=2)
        scope = f"在{'、'.join(hints)}中" if hints else "在知识库中"
        return (
            f"{scope}查找能直接回答“{query}”的证据，"
            "重点检索原因、条件、公式、实验数据或验证结论"
        )
