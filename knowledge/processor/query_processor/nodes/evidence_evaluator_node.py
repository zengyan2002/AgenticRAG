"""评审当前证据覆盖度，并决定回答、澄清或补检索。"""

from __future__ import annotations

import time
from typing import Any

from knowledge.processor.query_processor.agentic_models import (
    EvidenceEvaluation,
    parse_json_object,
)
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.agentic_prompt import (
    EVIDENCE_EVALUATOR_SYSTEM_PROMPT,
    EVIDENCE_EVALUATOR_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.document_identity_util import unique_strings
from knowledge.utils.evidence_packing_util import coverage_ordered_documents, format_grouped_evidence
from knowledge.utils.subquestion_retrieval_util import grouped_retrieval


class EvidenceEvaluatorNode(BaseNode):
    """只在 Agentic 查询中执行；异常时保留原流程结果。"""

    name = "evidence_evaluator_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # RerankNode 已经将历轮新旧证据统一评分并生成有序池；这里不再
        # previous-first 追加，避免第二轮关键证据天然落在旧证据后面。
        """执行 EvidenceEvaluatorNode 的核心处理流程。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        evidence_pool = [
            dict(document)
            for document in (
                state.get("agent_evidence_pool")
                or state.get("reranked_docs")
                or []
            )[:None if grouped_retrieval(state) else self.config.agent_evidence_pool_max]
            if isinstance(document, dict)
        ]
        state["agent_evidence_pool"] = evidence_pool
        state["reranked_docs"] = evidence_pool if grouped_retrieval(state) else evidence_pool[:self.config.rerank_max_top_k]

        try:
            evaluation = self._evaluate(state, evidence_pool)
        except Exception as exc:
            self.logger.warning(
                "Evidence Evaluator 失败，保留当前证据并结束 Agentic 循环: %s",
                exc,
            )
            evaluation = EvidenceEvaluation(
                sufficient=True,
                covered_sub_questions=list(
                    range(len(self._sub_questions(state)))
                ),
                next_action="answer",
                confidence=0.0,
            )
            state["agent_stop_reason"] = "evaluator_fallback"

        if grouped_retrieval(state):
            total = len(self._sub_questions(state))
            # 不接受越界索引、无候选证据的“已覆盖”，或只给 sufficient 而未逐项验收。
            covered = {i for i in evaluation.covered_sub_questions
                       if type(i) is int and 0 <= i < total
                       and (state.get("subquestion_evidence") or {}).get(str(i))}
            missing = {i for i in evaluation.missing_subquestion_indices if type(i) is int and 0 <= i < total}
            missing |= set(range(total)) - covered
            # 上游答案尚未可靠确定时，不能把依赖它的下游题标成已完成。
            for task in (state.get("agent_plan") or {}).get("retrieval_tasks") or []:
                if set(task.get("depends_on") or []) & missing:
                    missing.add(task["subquestion_index"])
            if not missing and (not evaluation.sufficient or evaluation.missing_aspects
                                or evaluation.confidence < self.config.agent_min_evidence_confidence):
                missing = set(range(total))
            covered -= missing
            evaluation = evaluation.model_copy(update={
                "covered_sub_questions": sorted(covered),
                "missing_subquestion_indices": sorted(missing),
                "sufficient": evaluation.sufficient and not missing,
                "next_action": "rewrite_missing" if missing and evaluation.next_action == "answer" else evaluation.next_action,
            })
            state["coverage_missing_subquestion_indices"] = sorted(missing)

        sufficient = (
            evaluation.sufficient
            and not evaluation.missing_aspects
            and evaluation.confidence >= self.config.agent_min_evidence_confidence
        )
        state["evidence_evaluation"] = evaluation.model_dump()
        state["evidence_next_action"] = evaluation.next_action
        state["evidence_sufficient"] = sufficient
        state["missing_aspects"] = unique_strings(
            evaluation.missing_aspects,
            max_items=self.config.agent_max_subquestions,
        )

        if state.get("agent_stop_reason") == "evaluator_fallback":
            return state
        if sufficient:
            state["evidence_next_action"] = "answer"
            state["agent_stop_reason"] = "evidence_sufficient"
            return state

        if evaluation.next_action == "ask_user":
            missing = state["missing_aspects"] or ["需要明确检索对象或范围"]
            state["answer"] = "当前问题还需要进一步确认：" + "；".join(missing)
            state["agent_stop_reason"] = "need_user_clarification"
            return state

        stop_reason = self._budget_stop_reason(state)
        if stop_reason:
            state["agent_stop_reason"] = stop_reason
            return state
        if (
            int(state.get("agent_iteration") or 0) > 1
            and not (state.get("agent_new_evidence_pairs") if grouped_retrieval(state)
                     else state.get("agent_new_chunk_ids"))
            and not self._has_pending_subquestion(state)
        ):
            state["agent_stop_reason"] = "no_new_evidence"
            return state
        state["agent_stop_reason"] = stop_reason
        return state

    def _evaluate(
        self,
        state: QueryGraphState,
        evidence_pool: list[dict[str, Any]],
    ) -> EvidenceEvaluation:
        """评估当前证据能否充分回答用户问题。

        Args:
            state: 当前工作流状态。
            evidence_pool: 待评估的候选证据集合。

        Returns:
            处理结果。

        Raises:
            ValueError: 输入无效或处理过程无法继续时抛出。
        """
        questions = self._sub_questions(state)
        if not evidence_pool:
            return EvidenceEvaluation(
                sufficient=False,
                missing_aspects=questions,
                next_action="rewrite_missing",
                confidence=1.0,
            )

        plan = state.get("agent_plan") or {}
        client = AIClients.get_llm_client(
            response_format=True,
            role="agent",
            thinking="deep",
        )
        started_at = time.perf_counter()
        try:
            response = client.invoke([
                ("system", EVIDENCE_EVALUATOR_SYSTEM_PROMPT),
                (
                    "user",
                    EVIDENCE_EVALUATOR_USER_PROMPT_TEMPLATE.format(
                        objective=plan.get("objective") or state.get("rewritten_query") or "",
                        sub_questions=questions,
                        success_criteria=plan.get("success_criteria") or [],
                        evidence=self._format_evidence(evidence_pool),
                        retrieval_tasks=plan.get("retrieval_tasks") or [],
                    ),
                ),
            ])
        finally:
            self.logger.info(
                "LLM latency: node_name=evidence_evaluator model_name=%s "
                "thinking_level=deep elapsed_ms=%.1f",
                getattr(client, "model_name", "unknown"),
                (time.perf_counter() - started_at) * 1000,
            )
        parsed = parse_json_object(response.content)
        if not parsed:
            raise ValueError("Evaluator 未返回有效 JSON")
        return EvidenceEvaluation.model_validate(parsed)

    def _budget_stop_reason(self, state: QueryGraphState) -> str:
        """判断查询是否因轮次或资源预算需要停止。

        Args:
            state: 当前工作流状态。

        Returns:
            处理后的字符串。
        """
        iteration = int(state.get("agent_iteration") or 0)
        if iteration >= self.config.agent_max_iterations:
            return "max_iterations_reached"

        selected_tools = state.get("agent_selected_tools") or []
        minimum = (2 if state.get("soft_filter_doc_ids") or (
            state.get("hard_filter_doc_ids") and not state.get("document_route_locked")) else 1)
        next_tool_calls = int(state.get("agent_tool_calls") or 0) + (
            minimum if grouped_retrieval(state) else len(selected_tools))
        if next_tool_calls > self.config.agent_max_tool_calls:
            return "max_tool_calls_reached"

        started_at = float(state.get("agent_started_at") or 0.0)
        if started_at and time.monotonic() - started_at >= self.config.agent_timeout_seconds:
            return "timeout_reached"
        return ""

    def _format_evidence(self, documents: list[dict[str, Any]]) -> str:
        """格式化证据。

        Args:
            documents: 待处理的文档集合。

        Returns:
            处理后的字符串。
        """
        if any(d.get("retrieved_for_subquestions") for d in documents):
            return format_grouped_evidence(documents, self.config.agent_evidence_max_chars,
                                           self.config.evidence_eval_max_chars_per_doc)
        blocks: list[str] = []
        current_chars = 0
        max_chars = self.config.agent_evidence_max_chars
        per_doc_max_chars = self.config.evidence_eval_max_chars_per_doc
        ordered_documents = coverage_ordered_documents(documents)
        for document in ordered_documents:
            content = str(document.get("content") or "").strip()
            if not content:
                continue
            index = len(blocks) + 1
            header = (
                f"【证据{index}】\n"
                f"文档：{document.get('canonical_title') or document.get('file_title') or document.get('theme_name') or ''}\n"
                f"章节：{document.get('section_path') or document.get('parent_title') or document.get('title') or ''}\n"
                f"正文："
            )
            separator = "\n\n" if blocks else ""
            remaining = max_chars - current_chars - len(separator) - len(header)
            if remaining <= 0:
                break
            body = content[:min(remaining, per_doc_max_chars)]
            blocks.append(header + body)
            current_chars += len(separator) + len(header) + len(body)
        return "\n\n".join(blocks)

    @staticmethod
    def _has_pending_subquestion(state):
        if not grouped_retrieval(state):
            return False
        history = state.get("subquestion_retrieval_history") or {}
        return any(not history.get(str(i)) for i in state.get("coverage_missing_subquestion_indices") or [])

    @staticmethod
    def _sub_questions(state: QueryGraphState) -> list[str]:
        """提取并规范化查询计划中的子问题。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        plan = state.get("agent_plan") or {}
        if grouped_retrieval(state):
            return list(plan["sub_questions"])
        return unique_strings(
            plan.get("sub_questions")
            or state.get("retrieval_queries")
            or [state.get("rewritten_query") or state.get("original_query") or ""]
        )
