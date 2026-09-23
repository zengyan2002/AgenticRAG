"""为复杂科研查询生成受控、结构化的检索计划。"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import time
from typing import Any

from knowledge.processor.query_processor.agentic_models import (
    AgentPlan,
    RetrievalTask,
    parse_json_object,
)
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.agentic_prompt import (
    AGENT_PLANNER_SYSTEM_PROMPT,
    AGENT_PLANNER_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.document_identity_util import unique_strings
from knowledge.utils.subquestion_retrieval_util import grouped_retrieval, reset_group_state, schedule_tasks


class PlannerNode(BaseNode):
    """LLM 失败时复用文档路由阶段已有的确定性子问题。"""

    name = "planner_node"
    _VALID_TOOLS = ("vector", "hyde", "bm25")

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 PlannerNode 的核心处理流程。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        try:
            plan = self._invoke_planner(state)
            plan = self._normalize_plan(plan, state)
        except Exception as exc:
            self.logger.warning("Agent Planner 失败，使用保守计划: %s", exc)
            plan = self._normalize_plan(self._fallback_plan(state), state)

        plan_data = plan.model_dump()
        queries = plan_data["search_queries"]
        selected_tools = plan_data["retrieval_tools"]

        state["agent_plan"] = plan_data
        state["agent_selected_tools"] = selected_tools
        state["agent_iteration"] = 1
        state["agent_tool_calls"] = len(selected_tools)
        state["agent_current_query"] = str(
            state.get("rewritten_query") or state.get("original_query") or ""
        ).strip()
        state["agent_retrieval_history"] = list(queries)
        state["retrieval_queries"] = list(queries)
        state["query_decomposed"] = len(queries) > 1
        state["agent_stop_reason"] = ""
        reset_group_state(state)
        if grouped_retrieval(state):
            state["agent_tool_calls"] = 0
            ready = [task for task in plan_data["retrieval_tasks"] if not task["depends_on"]]
            schedule_tasks(state, ready, self.config)
            state["agent_retrieval_history"] = []
        return state

    def _invoke_planner(self, state: QueryGraphState) -> AgentPlan:
        """调用规划模型生成结构化查询计划。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。

        Raises:
            ValueError: 输入无效或处理过程无法继续时抛出。
        """
        client = AIClients.get_llm_client(
            response_format=True,
            role="agent",
            thinking="deep",
        )
        started_at = time.perf_counter()
        try:
            response = client.invoke([
                ("system", AGENT_PLANNER_SYSTEM_PROMPT),
                (
                    "user",
                    AGENT_PLANNER_USER_PROMPT_TEMPLATE.format(
                        original_query=state.get("original_query") or "",
                        rewritten_query=state.get("rewritten_query") or "",
                        retrieval_queries=state.get("retrieval_queries") or [],
                        document_route_mode=state.get("document_route_mode") or "global",
                        document_hints=self._document_hints(state),
                    ),
                ),
            ])
        finally:
            self.logger.info(
                "LLM latency: node_name=planner model_name=%s "
                "thinking_level=deep elapsed_ms=%.1f",
                getattr(client, "model_name", "unknown"),
                (time.perf_counter() - started_at) * 1000,
            )
        parsed = parse_json_object(response.content)
        if not parsed:
            raise ValueError("Planner 未返回有效 JSON")
        return AgentPlan.model_validate(parsed)

    def _normalize_plan(
        self,
        plan: AgentPlan,
        state: QueryGraphState,
    ) -> AgentPlan:
        """规范化plan。

        Args:
            plan: 当前查询执行计划。
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        fallback_query = str(
            state.get("rewritten_query") or state.get("original_query") or ""
        ).strip()
        existing_queries = state.get("retrieval_queries") or []
        sub_questions = unique_strings(
            plan.sub_questions,
            max_items=self.config.agent_max_subquestions,
        )
        if not sub_questions:
            sub_questions = unique_strings(
                existing_queries or [fallback_query],
                max_items=self.config.agent_max_subquestions,
            ) or [fallback_query]

        # sub_questions 是答案必须覆盖的事实维度；search_queries 才是
        # Retriever 的输入。只有 Planner 没有提供 focused query 时，
        # 才用 sub_questions 兜底，避免每个事实维度机械复制成检索请求。
        focused_queries = self._deduplicate_search_queries(
            plan.search_queries,
            source_query=fallback_query,
            max_items=4,
        )
        fallback_searches = sub_questions if not focused_queries else []
        search_queries = self._deduplicate_search_queries(
            [fallback_query, *focused_queries, *fallback_searches],
            source_query=fallback_query,
            max_items=5,
        ) or [fallback_query]

        tools = [tool for tool in plan.retrieval_tools if tool in self._VALID_TOOLS]
        if not tools:
            tools = self._tools_for_profile(plan.retrieval_profile)
        if "vector" not in tools:
            tools.insert(0, "vector")
        if not self.config.agent_hyde_enabled:
            tools = [tool for tool in tools if tool != "hyde"]
        if "bm25" not in tools:
            tools.append("bm25")
        tools = list(dict.fromkeys(tools))
        tools = tools[: self.config.agent_max_tool_calls]

        criteria = unique_strings(
            plan.success_criteria,
            max_items=self.config.agent_max_subquestions,
        )
        if not criteria:
            criteria = [f"找到能够直接回答“{query}”的证据" for query in sub_questions]

        # 索引直接对应原始 sub_questions，不能先去重再让任务索引错位。
        if plan.retrieval_tasks:
            sub_questions = [q.strip() for q in plan.sub_questions[:self.config.agent_max_subquestions]]
            if not sub_questions or any(not q for q in sub_questions):
                raise ValueError("任务引用的子问题不能为空")
        task_map = {}
        for task in plan.retrieval_tasks:
            i = task.subquestion_index
            if i >= len(sub_questions):
                continue
            previous = task_map.get(i)
            queries = self._deduplicate_search_queries(
                [*(previous.search_queries if previous else []), *task.search_queries],
                source_query=sub_questions[i], max_items=2,
            ) or [sub_questions[i]]
            # 只允许依赖先前子问题，拒绝环和悬空依赖，而不是悄悄并行。
            dependencies = list(dict.fromkeys([*(previous.depends_on if previous else []), *task.depends_on]))
            if any(type(dep) is not int or dep < 0 or dep >= i for dep in dependencies):
                raise ValueError("子问题依赖必须指向前序子问题")
            task_tools = [t for t in (task.retrieval_tools or tools) if t in tools]
            task_map[i] = RetrievalTask(subquestion_index=i, search_queries=queries,
                                        retrieval_tools=task_tools or tools, depends_on=dependencies)
        for i, question in enumerate(sub_questions):
            task_map.setdefault(i, RetrievalTask(subquestion_index=i,
                search_queries=[question], retrieval_tools=tools))
        if len(sub_questions) == 1 and plan.retrieval_tasks:
            search_queries = self._deduplicate_search_queries(
                [fallback_query, *task_map[0].search_queries],
                source_query=fallback_query, max_items=5,
            )

        return plan.model_copy(update={
            "objective": plan.objective or fallback_query,
            "sub_questions": sub_questions,
            "search_queries": search_queries,
            "retrieval_tasks": [task_map[i] for i in range(len(sub_questions))],
            "document_hints": unique_strings(
                [*plan.document_hints, *self._document_hints(state)],
                max_items=self.config.document_route_max_options,
            ),
            "retrieval_tools": tools,
            "success_criteria": criteria,
        })

    @classmethod
    def _deduplicate_search_queries(
        cls,
        queries: list[str],
        *,
        source_query: str,
        max_items: int,
    ) -> list[str]:
        """删除高度重复查询，并限制中文问题最多一条纯英文表达。"""
        selected: list[str] = []
        normalized: list[str] = []
        source_is_chinese = bool(re.search(r"[\u4e00-\u9fff]", source_query))
        english_count = 0

        for raw_query in queries or []:
            query = re.sub(r"\s+", " ", str(raw_query or "")).strip()
            key = re.sub(r"[^\w\u4e00-\u9fff]+", "", query).lower()
            if not query or not key:
                continue

            is_english = cls._is_english_query(query)
            if source_is_chinese and is_english and english_count >= 1:
                continue
            if any(cls._queries_are_similar(key, old_key) for old_key in normalized):
                continue

            selected.append(query)
            normalized.append(key)
            english_count += int(source_is_chinese and is_english)
            if len(selected) >= max_items:
                break
        return selected

    @staticmethod
    def _queries_are_similar(left: str, right: str) -> bool:
        """判断两个查询是否可视为相同检索意图。

        Args:
            left: 参与比较的左侧值。
            right: 参与比较的右侧值。

        Returns:
            条件成立时返回 ``True``，否则返回 ``False``。
        """
        if left == right:
            return True
        shorter, longer = sorted((left, right), key=len)
        if len(shorter) >= 12 and shorter in longer:
            return True
        return SequenceMatcher(None, left, right).ratio() >= 0.90

    @staticmethod
    def _is_english_query(query: str) -> bool:
        """判断english查询是否满足条件。

        Args:
            query: 用户查询文本。

        Returns:
            条件成立时返回 ``True``，否则返回 ``False``。
        """
        return (
            not re.search(r"[\u4e00-\u9fff]", query)
            and len(re.findall(r"[A-Za-z]+", query)) >= 2
        )

    def _fallback_plan(self, state: QueryGraphState) -> AgentPlan:
        """在规划模型不可用时构造保守查询计划。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        reason = state.get("agentic_route_reason") or "multi_fact"
        if reason == "comparison":
            intent = "comparison"
            profile = "comparison"
        elif reason == "multi_hop":
            intent = "multi_hop"
            profile = "deep"
        else:
            intent = "multi_fact"
            profile = "deep"
        return AgentPlan(
            intent=intent,
            objective=str(
                state.get("rewritten_query") or state.get("original_query") or ""
            ),
            sub_questions=list(state.get("retrieval_queries") or []),
            search_queries=list(state.get("retrieval_queries") or []),
            document_hints=self._document_hints(state),
            retrieval_profile=profile,
            retrieval_tools=self._tools_for_profile(profile),
        )

    def _tools_for_profile(self, profile: str) -> list[str]:
        """根据查询档案选择需要执行的检索工具。

        Args:
            profile: 当前命中的文档档案。

        Returns:
            处理结果。
        """
        if profile == "fast" or not self.config.agent_hyde_enabled:
            return ["vector", "bm25"]
        return ["vector", "hyde", "bm25"]

    @staticmethod
    def _document_hints(state: QueryGraphState) -> list[str]:
        """整理查询涉及的候选文档提示。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        documents = (
            state.get("selected_documents")
            or state.get("document_candidates")
            or []
        )
        values: list[Any] = list(state.get("document_mentions") or [])
        for document in documents:
            if isinstance(document, dict):
                values.append(
                    document.get("canonical_title")
                    or document.get("primary_subject")
                )
        return unique_strings(values)
