"""决定当前查询是否进入选择性 Agentic RAG 闭环。"""

from __future__ import annotations

import re
import time
from typing import Iterable

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class AgentRouterNode(BaseNode):
    """使用低成本确定性规则路由简单与复杂查询。"""

    name = "agent_router_node"

    _COMPARISON_PATTERN = re.compile(
        r"区别|差异|异同|对比|比较|相比|优缺点|相同点|不同点|"
        r"各有什么特点|各自有什么特点|各有何特点",
        re.IGNORECASE,
    )
    _MULTI_HOP_PATTERN = re.compile(
        r"为什么.*(?:如何|怎么|影响|导致)|"
        r"原因.*(?:结果|结论|影响|机制)|"
        r"关系是什么|如何证明|结合.*说明|依据.*判断",
        re.IGNORECASE,
    )
    _MULTI_FACT_PATTERN = re.compile(
        r"分别|各自|各有|各是|同时|以及|并且|并说明|并给出|"
        r"(?:原理|特点|条件|公式|参数|流程|局限|限制|不足|结果|结论|指标)"
        r".*(?:和|与|及|、).*"
        r"(?:原理|特点|条件|公式|参数|流程|局限|限制|不足|结果|结论|指标)",
        re.IGNORECASE,
    )

    def process(self, state: QueryGraphState) -> QueryGraphState:
        state["agentic_active"] = False
        state["agentic_route_reason"] = "disabled"
        state["agent_plan"] = {}
        state["agent_selected_tools"] = []
        state["agent_iteration"] = 0
        state["agent_tool_calls"] = 0
        state["agent_started_at"] = 0.0
        state["agent_current_query"] = ""
        state["agent_retrieval_history"] = []
        state["agent_evidence_pool"] = []
        state["evidence_evaluation"] = {}
        state["evidence_sufficient"] = False
        state["missing_aspects"] = []
        state["agent_stop_reason"] = ""

        if not self.config.agentic_rag_enabled:
            return state

        mode = self.config.agentic_rag_mode
        if mode not in {"workflow", "agentic", "hybrid"}:
            self.logger.warning(
                "未知 AGENTIC_RAG_MODE=%s，安全回退 workflow", mode
            )
            state["agentic_route_reason"] = "invalid_mode"
            return state
        if mode == "workflow":
            state["agentic_route_reason"] = "workflow_mode"
            return state

        query = " ".join(
            dict.fromkeys(
                value.strip()
                for value in (
                    state.get("original_query"),
                    state.get("rewritten_query"),
                    state.get("retrieval_query"),
                )
                if isinstance(value, str) and value.strip()
            )
        )
        retrieval_queries = self._valid_queries(state.get("retrieval_queries") or [])

        active = mode == "agentic"
        reason = "forced_agentic" if active else "simple_query"
        if mode == "hybrid":
            active, reason = self._is_complex(query, retrieval_queries)

        state["agentic_active"] = active
        state["agentic_route_reason"] = reason
        if active:
            state["agent_started_at"] = time.monotonic()
        return state

    @classmethod
    def _is_complex(
        cls,
        query: str,
        retrieval_queries: Iterable[str],
    ) -> tuple[bool, str]:
        queries = cls._valid_queries(retrieval_queries)
        if cls._MULTI_HOP_PATTERN.search(query):
            return True, "multi_hop"
        if cls._COMPARISON_PATTERN.search(query):
            return True, "comparison"
        if cls._MULTI_FACT_PATTERN.search(query):
            return True, "multi_fact"

        # 多条检索表达只作为辅助信号：只有原问题本身也包含多个独立
        # 问句/事实要求时才进入 Agentic，避免中英文同义检索误触发。
        demand_markers = re.findall(
            r"为什么|为何|如何|怎么|多少|哪些|是什么|有什么|结论|结果",
            query,
            flags=re.IGNORECASE,
        )
        if len(queries) > 1 and len(demand_markers) >= 2:
            return True, "multi_fact"
        return False, "simple_query"

    @staticmethod
    def _valid_queries(values: Iterable[str]) -> list[str]:
        return [
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ]
