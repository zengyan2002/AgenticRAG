"""为复杂科研查询生成受控、结构化的检索计划。"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import time
from typing import Any

from knowledge.processor.query_processor.agentic_models import (
    AgentPlan,
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


class PlannerNode(BaseNode):
    """LLM 失败时复用文档路由阶段已有的确定性子问题。"""

    name = "planner_node"
    _VALID_TOOLS = ("vector", "hyde", "bm25")

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            plan = self._invoke_planner(state)
        except Exception as exc:
            self.logger.warning("Agent Planner 失败，使用保守计划: %s", exc)
            plan = self._fallback_plan(state)

        plan = self._normalize_plan(plan, state)
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
        return state

    def _invoke_planner(self, state: QueryGraphState) -> AgentPlan:
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

        return plan.model_copy(update={
            "objective": plan.objective or fallback_query,
            "sub_questions": sub_questions,
            "search_queries": search_queries,
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
        if left == right:
            return True
        shorter, longer = sorted((left, right), key=len)
        if len(shorter) >= 12 and shorter in longer:
            return True
        return SequenceMatcher(None, left, right).ratio() >= 0.90

    @staticmethod
    def _is_english_query(query: str) -> bool:
        return (
            not re.search(r"[\u4e00-\u9fff]", query)
            and len(re.findall(r"[A-Za-z]+", query)) >= 2
        )

    def _fallback_plan(self, state: QueryGraphState) -> AgentPlan:
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
        if profile == "fast" or not self.config.agent_hyde_enabled:
            return ["vector", "bm25"]
        return ["vector", "hyde", "bm25"]

    @staticmethod
    def _document_hints(state: QueryGraphState) -> list[str]:
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
