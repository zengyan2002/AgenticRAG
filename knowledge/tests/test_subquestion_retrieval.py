"""多子问题链路的覆盖、隔离、补检索及预算回归测试；不访问外部服务。"""

import time
import unittest
from unittest.mock import MagicMock, patch

from langgraph.graph import StateGraph, START, END

from knowledge.processor.query_processor.agentic_models import AgentPlan, RetrievalTask, EvidenceEvaluation
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.state import QueryGraphState, create_default_state
from knowledge.processor.query_processor.nodes.planner_node import PlannerNode
from knowledge.processor.query_processor.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_processor.nodes.bm25_search_node import BM25SearchNode
from knowledge.processor.query_processor.nodes.hyde_search_node import HydeSearchNode
from knowledge.processor.query_processor.nodes.rrf_merge_node import RRFMergeNode
from knowledge.processor.query_processor.nodes.rerank_node import RerankNode
from knowledge.processor.query_processor.nodes.conflict_judge_node import ConflictJudgeNode
from knowledge.processor.query_processor.nodes.evidence_evaluator_node import EvidenceEvaluatorNode
from knowledge.processor.query_processor.nodes.replan_node import ReplanNode
from knowledge.processor.query_processor.nodes.context_expansion_node import ContextExpansionNode
from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutputNode
from knowledge.utils.subquestion_retrieval_util import (
    schedule_tasks, merge_evidence_groups, budgeted_retrieval_client,
)
from knowledge.utils.evidence_packing_util import format_grouped_evidence


def chunk(cid, text=None):
    return {"chunk_id": cid, "content": text or cid, "title": cid, "source": "local"}


def task(index, queries=None, tools=None, depends=None):
    return {"subquestion_index": index, "search_queries": queries or [f"Q{index}"],
            "retrieval_tools": tools or ["vector", "bm25"], "depends_on": depends or []}


def state_for(count=2, **overrides):
    state = create_default_state(agentic_active=True, rewritten_query="完整比较问题",
        agent_iteration=1, agent_started_at=time.monotonic(),
        agent_plan={"sub_questions": [f"Q{i}" for i in range(count)],
                    "retrieval_tasks": [task(i) for i in range(count)]},
        active_subquestion_indices=list(range(count)))
    state.update(overrides)
    return state


class SubquestionRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.config = QueryConfig(agent_max_tool_calls=24, agent_max_iterations=3,
            bm25_enabled=True, agent_hyde_enabled=False, agent_subquestion_top_k=2,
            rerank_coverage_min_score=0.1, rrf_max_results=4,
            rrf_min_exclusive_per_branch=0, context_expansion_top_k=0)

    def test_planner_maps_tasks_and_fills_missing_without_cross_assigning_flat_queries(self):
        plan = AgentPlan(sub_questions=["A原理", "B局限", "C条件"], search_queries=["全部混合查询"],
            retrieval_tasks=[RetrievalTask(subquestion_index=1, search_queries=["B失败条件"]),
                             RetrievalTask(subquestion_index=1, search_queries=["B失效案例"]),
                             RetrievalTask(subquestion_index=99, search_queries=["越界"])])
        normalized = PlannerNode(self.config)._normalize_plan(plan, state_for())
        self.assertEqual([0, 1, 2], [t.subquestion_index for t in normalized.retrieval_tasks])
        self.assertEqual(["A原理"], normalized.retrieval_tasks[0].search_queries)
        self.assertEqual(["B失败条件", "B失效案例"], normalized.retrieval_tasks[1].search_queries)

    def test_invalid_dependencies_do_not_silently_run_in_parallel(self):
        for dep in [0, 1, -1]:
            with self.subTest(dep=dep), self.assertRaises(ValueError):
                PlannerNode(self.config)._normalize_plan(AgentPlan(sub_questions=["A", "B"],
                    retrieval_tasks=[RetrievalTask(subquestion_index=0, depends_on=[dep])]), state_for())

    def test_rrf_fuses_modalities_within_each_question_without_global_limit(self):
        state = state_for(subquestion_vector_chunks={"0": [chunk("a"), chunk("common")],
                                                     "1": [chunk("b")]},
                          subquestion_bm25_chunks={"0": [chunk("common")], "1": [chunk("b2")]},
                          vector_retrieval_calls=2, bm25_retrieval_calls=2)
        result = RRFMergeNode(self.config).process(state)
        self.assertEqual("common", result["subquestion_rrf_chunks"]["0"][0]["chunk_id"])
        self.assertEqual({"b", "b2"}, {d["chunk_id"] for d in result["subquestion_rrf_chunks"]["1"]})
        self.assertEqual([], result["rrf_chunks"])
        self.assertEqual(4, result["agent_tool_calls"])

    def test_rerank_batches_question_document_pairs_and_preserves_both_groups(self):
        state = state_for(subquestion_rrf_chunks={"0": [chunk("a"), chunk("shared")],
                                                  "1": [chunk("b"), chunk("shared")]})
        client = MagicMock()
        client.compute_score.return_value = [8.0, 7.0, 1.0, 0.0]
        with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=client):
            result = RerankNode(self.config).process(state)
        client.compute_score.assert_called_once()
        pairs = client.compute_score.call_args.args[0]
        self.assertEqual(["Q0", "Q0", "Q1", "Q1"], [q for q, _ in pairs])
        self.assertEqual({"a", "b", "shared"}, {d["chunk_id"] for d in result["reranked_docs"]})
        shared = next(d for d in result["reranked_docs"] if d["chunk_id"] == "shared")
        self.assertEqual([0, 1], shared["retrieved_for_subquestions"])
        self.assertNotEqual(shared["subquestion_scores"]["0"], shared["subquestion_scores"]["1"])
        self.assertNotIn("retrieved_for_subquestions", state["subquestion_evidence"]["0"][1])

    def test_reranker_failure_keeps_each_question_rrf_order(self):
        state = state_for(subquestion_rrf_chunks={"0": [chunk("a")], "1": [chunk("b")]})
        with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", side_effect=RuntimeError("offline")):
            RerankNode(self.config).process(state)
        self.assertEqual(["a", "b"], [d["chunk_id"] for d in state["reranked_docs"]])
        self.assertTrue(all(d["rerank_score"] is None for d in state["reranked_docs"]))

    def test_low_relevance_is_not_filled_to_top_k(self):
        state = state_for(subquestion_rrf_chunks={"0": [chunk("a")], "1": [chunk("irrelevant")]})
        client = MagicMock()
        client.compute_score.return_value = [5, -100]
        with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=client):
            RerankNode(self.config).process(state)
        self.assertEqual([], state["subquestion_evidence"]["1"])

    def test_replan_only_missing_group_and_keeps_old_evidence(self):
        state = state_for(subquestion_evidence={"0": [chunk("done")], "1": [chunk("old")]},
            coverage_missing_subquestion_indices=[1],
            evidence_evaluation={"covered_sub_questions": [0], "followup_tasks": [task(1, ["B局限新查询"])]},
            subquestion_retrieval_history={"0": ["Q0"], "1": ["Q1"]})
        ReplanNode(self.config).process(state)
        self.assertEqual([1], state["active_subquestion_indices"])
        state["subquestion_rrf_chunks"] = {"1": [chunk("new")]}
        client = MagicMock()
        client.compute_score.return_value = [5, 1]
        with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=client):
            RerankNode(self.config).process(state)
        self.assertEqual([chunk("done")], state["subquestion_evidence"]["0"])
        self.assertEqual(["Q1", "Q1"], [q for q, _ in client.compute_score.call_args.args[0]])
        self.assertEqual(["new", "old"], [d["chunk_id"] for d in state["subquestion_evidence"]["1"]])

    def test_dependency_waits_for_covered_upstream_and_resolved_query(self):
        state = state_for(coverage_missing_subquestion_indices=[1])
        state["agent_plan"]["retrieval_tasks"][1]["depends_on"] = [0]
        state["evidence_evaluation"] = {"covered_sub_questions": [], "followup_tasks": [task(1, ["ESPRIT局限"])]}
        ReplanNode(self.config).process(state)
        self.assertEqual("no_ready_subquestion_queries", state["agent_stop_reason"])
        state["evidence_evaluation"] = {"covered_sub_questions": [0]}
        ReplanNode(self.config).process(state)
        self.assertEqual("no_ready_subquestion_queries", state["agent_stop_reason"])
        state["evidence_evaluation"]["followup_tasks"] = [task(1, ["ESPRIT局限"])]
        ReplanNode(self.config).process(state)
        self.assertEqual([1], state["active_subquestion_indices"])
        self.assertEqual(["ESPRIT局限"], state["retrieval_queries"])

    def test_initial_planner_does_not_schedule_dependent_question(self):
        state = state_for()
        plan = AgentPlan(sub_questions=["论文用什么方法", "该方法有什么局限"], retrieval_tasks=[
            RetrievalTask(subquestion_index=0, search_queries=["论文方法"]),
            RetrievalTask(subquestion_index=1, depends_on=[0], search_queries=["该方法局限"])])
        node = PlannerNode(self.config)
        with patch.object(node, "_invoke_planner", return_value=plan):
            node.process(state)
        self.assertEqual([0], state["active_subquestion_indices"])

    def test_evaluator_cannot_claim_missing_or_out_of_range_groups_covered(self):
        state = state_for(subquestion_evidence={"0": [chunk("a")], "1": []},
                          agent_evidence_pool=[chunk("a")])
        evaluation = EvidenceEvaluation(sufficient=True, covered_sub_questions=[0, 1, 99], confidence=.95)
        node = EvidenceEvaluatorNode(self.config)
        with patch.object(node, "_evaluate", return_value=evaluation):
            node.process(state)
        self.assertFalse(state["evidence_sufficient"])
        self.assertEqual([1], state["coverage_missing_subquestion_indices"])
        self.assertEqual("rewrite_missing", state["evidence_next_action"])

    def test_dependency_resolution_is_used_by_reranker(self):
        state = state_for(subquestion_resolved_questions={"1": "ESPRIT方法有哪些局限"},
                          active_subquestion_indices=[1], subquestion_rrf_chunks={"1": [chunk("a")]})
        client = MagicMock()
        client.compute_score.return_value = 2.0
        with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=client):
            RerankNode(self.config).process(state)
        self.assertIn("ESPRIT", client.compute_score.call_args.args[0][0][0])

    def test_missing_upstream_invalidates_downstream_coverage(self):
        state = state_for(subquestion_evidence={"0": [], "1": [chunk("b")]}, agent_evidence_pool=[chunk("b")])
        state["agent_plan"]["retrieval_tasks"][1]["depends_on"] = [0]
        node = EvidenceEvaluatorNode(self.config)
        with patch.object(node, "_evaluate", return_value=EvidenceEvaluation(
            sufficient=False, covered_sub_questions=[1], missing_subquestion_indices=[0], confidence=.95)):
            node.process(state)
        self.assertEqual([0, 1], state["coverage_missing_subquestion_indices"])

    def test_malformed_rerank_scores_fall_back_without_losing_groups(self):
        for scores in ([float("nan"), 1.0], [1.0]):
            state = state_for(subquestion_rrf_chunks={"0": [chunk("a")], "1": [chunk("b")]})
            client = MagicMock()
            client.compute_score.return_value = scores
            with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=client):
                RerankNode(self.config).process(state)
            self.assertEqual(["a", "b"], [d["chunk_id"] for d in state["reranked_docs"]])

    def test_rerank_microbatches_are_bounded(self):
        self.config.agent_rerank_batch_size = 2
        state = state_for(subquestion_rrf_chunks={"0": [chunk("a"), chunk("b")], "1": [chunk("c")]})
        client = MagicMock()
        client.compute_score.side_effect = [[5.0, 3.0], 4.0]
        with patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=client):
            RerankNode(self.config).process(state)
        self.assertEqual([2, 1], [len(c.args[0]) for c in client.compute_score.call_args_list])

    def test_single_question_uses_legacy_retrieval_and_rerank(self):
        state = state_for(1)
        vector, rerank = VectorSearchNode(self.config), RerankNode(self.config)
        with patch.object(vector, "_process_ungrouped", return_value={"embedding_chunks": [chunk("a")]}) as old:
            self.assertEqual("a", vector.process(state)["embedding_chunks"][0]["chunk_id"])
            old.assert_called_once()
        with patch.object(rerank, "_rerank_subquestions") as grouped, \
             patch.object(rerank, "_rerank_merge_docs", return_value=[]):
            rerank.process(state)
            grouped.assert_not_called()

    def test_group_request_budget_stops_excess_requests(self):
        state = state_for()
        self.config.agent_max_tool_calls = 1
        schedule_tasks(state, [task(0, tools=["vector"])], self.config)
        node, client = VectorSearchNode(self.config), MagicMock()
        def excessive(local):
            proxy = budgeted_retrieval_client(client)
            proxy.hybrid_search()
            proxy.hybrid_search()
        with patch.object(node, "_process_ungrouped", side_effect=excessive):
            result = node.process(state)
        self.assertEqual(1, client.hybrid_search.call_count)
        self.assertEqual(1, result["vector_retrieval_calls"])
        self.assertTrue(result["vector_retrieval_errors"])

    def test_each_group_first_anchor_can_expand(self):
        self.config.context_expansion_top_k = 1
        groups = {str(i): [chunk(f"c{i}")] for i in range(4)}
        state = state_for(4, reranked_docs=merge_evidence_groups(groups, [f"Q{i}" for i in range(4)]))
        node = ContextExpansionNode(self.config)
        with patch.object(node, "_expand_document", side_effect=lambda doc: doc) as expand:
            node.process(state)
        self.assertEqual(4, expand.call_count)

    def test_context_budget_preserves_four_groups_and_deduplicates_shared_text(self):
        groups = {str(i): [chunk(f"c{i}", chr(65+i) * 3000), chunk("shared", "共同证据")] for i in range(4)}
        docs = merge_evidence_groups(groups, [f"Q{i}" for i in range(4)])
        selected = []
        packed = format_grouped_evidence(docs, 1400, 2500, selected)
        self.assertLessEqual(len(packed), 1400)
        for i in range(4):
            self.assertIn(f"c{i}", selected)
            self.assertIn(chr(65+i) * 10, packed)
        self.assertLessEqual(packed.count("共同证据"), 1)
        self.assertEqual(len(selected), len(set(selected)))

    def test_evaluator_does_not_apply_legacy_global_topk(self):
        self.config.rerank_max_top_k = 1
        self.config.agent_evidence_pool_max = 1
        groups = {str(i): [chunk(f"c{i}")] for i in range(4)}
        state = state_for(4, subquestion_evidence=groups,
                          agent_evidence_pool=merge_evidence_groups(groups, [f"Q{i}" for i in range(4)]))
        node = EvidenceEvaluatorNode(self.config)
        with patch.object(node, "_evaluate", return_value=EvidenceEvaluation(
            sufficient=True, covered_sub_questions=[0, 1, 2, 3], confidence=.95)):
            node.process(state)
        self.assertEqual(4, len(state["reranked_docs"]))

    def test_budget_allocation_covers_questions_before_query_variants(self):
        self.config.agent_max_tool_calls = 4
        state = state_for(4)
        tasks = [task(i, [f"Q{i}", f"variant{i}"]) for i in range(4)]
        schedule_tasks(state, tasks, self.config)
        self.assertEqual([0, 1, 2, 3], state["active_subquestion_indices"])
        self.assertEqual(4, sum(len(t["operations"]) for t in state["active_retrieval_tasks"]))

    def test_real_bm25_requests_count_including_soft_route(self):
        state = state_for(soft_filter_doc_ids=["doc_a"])
        tasks = [task(i, tools=["bm25"]) for i in range(2)]
        schedule_tasks(state, tasks, self.config)
        client = MagicMock()
        client.search.side_effect = lambda **kw: [[chunk(kw["data"][0])]]
        with patch("knowledge.processor.query_processor.nodes.bm25_search_node.StorageClients.get_milvus", return_value=client):
            result = BM25SearchNode(self.config).process(state)
        self.assertEqual(4, result["bm25_retrieval_calls"])
        self.assertEqual(4, client.search.call_count)
        self.assertEqual({"0", "1"}, set(result["subquestion_bm25_chunks"]))

    def test_locked_scope_does_not_fallback_to_global(self):
        state = state_for(hard_filter_doc_ids=["doc_a"], document_route_locked=True)
        schedule_tasks(state, [task(0, tools=["bm25"])], self.config)
        client = MagicMock()
        client.search.return_value = [[]]
        with patch("knowledge.processor.query_processor.nodes.bm25_search_node.StorageClients.get_milvus", return_value=client):
            result = BM25SearchNode(self.config).process(state)
        self.assertEqual(1, client.search.call_count)
        self.assertEqual([], result["subquestion_bm25_chunks"]["0"])

    def test_failed_subquestion_does_not_erase_other_groups(self):
        state = state_for()
        schedule_tasks(state, [task(i, tools=["bm25"]) for i in range(2)], self.config)
        client = MagicMock()
        client.search.side_effect = [RuntimeError("offline for Q0"), [[chunk("b")]]]
        with patch("knowledge.processor.query_processor.nodes.bm25_search_node.StorageClients.get_milvus", return_value=client):
            result = BM25SearchNode(self.config).process(state)
        self.assertEqual([], result["subquestion_bm25_chunks"]["0"])
        self.assertEqual("b", result["subquestion_bm25_chunks"]["1"][0]["chunk_id"])
        self.assertEqual(2, result["bm25_retrieval_calls"])

    def test_expired_timeout_prevents_new_requests(self):
        state = state_for(agent_started_at=time.monotonic() - 1000)
        schedule_tasks(state, [task(0, tools=["bm25"])], self.config)
        with patch("knowledge.processor.query_processor.nodes.bm25_search_node.StorageClients.get_milvus") as client:
            result = BM25SearchNode(self.config).process(state)
        client.assert_not_called()
        self.assertEqual(0, result["bm25_retrieval_calls"])

    def test_hyde_gets_one_focused_query_per_subquestion(self):
        self.config.agent_hyde_enabled = True
        state = state_for()
        schedule_tasks(state, [task(i, [f"Q{i}", f"variant{i}"], ["hyde"]) for i in range(2)], self.config)
        node = HydeSearchNode(self.config)
        with patch.object(node, "_process_ungrouped", return_value={"hyde_embedding_chunks": []}) as call:
            node.process(state)
        self.assertEqual(["Q0", "Q1"], [c.args[0]["rewritten_query"] for c in call.call_args_list])

    def test_grouped_pipeline_uses_actual_langgraph_parallel_join(self):
        state = state_for()
        schedule_tasks(state, state["agent_plan"]["retrieval_tasks"], self.config)
        graph = StateGraph(QueryGraphState)
        vector, bm25, hyde = VectorSearchNode(self.config), BM25SearchNode(self.config), HydeSearchNode(self.config)
        for name, node in [("vector", vector), ("bm25", bm25), ("hyde", hyde),
                           ("rrf", RRFMergeNode(self.config)), ("rerank", RerankNode(self.config)),
                           ("judge", ConflictJudgeNode(self.config)), ("evaluate", EvidenceEvaluatorNode(self.config)),
                           ("replan", ReplanNode(self.config)),
                           ("expand", ContextExpansionNode(self.config))]:
            graph.add_node(name, node)
        for name in ("vector", "bm25", "hyde"):
            graph.add_edge(START, name)
        graph.add_edge(["vector", "bm25", "hyde"], "rrf")
        for left, right in [("rrf", "rerank"), ("rerank", "judge"), ("judge", "evaluate")]:
            graph.add_edge(left, right)
        graph.add_conditional_edges("evaluate", lambda s: "expand" if s.get("agent_stop_reason") else "replan", ["expand", "replan"])
        graph.add_conditional_edges("replan", lambda s: "expand" if s.get("agent_stop_reason") else ["vector", "bm25", "hyde"],
                                    ["expand", "vector", "bm25", "hyde"])
        graph.add_edge("expand", END)
        def retrieve(local):
            # 接近真实节点的路径：在单查询内实际调用预算代理。
            budgeted_retrieval_client(db).hybrid_search()
            return {"embedding_chunks": [chunk(local["rewritten_query"])]}
        db = MagicMock()
        db.search.side_effect = lambda **kw: [[chunk(kw["data"][0])]]
        reranker = MagicMock()
        reranker.compute_score.return_value = [5, 4]
        with patch.object(vector, "_process_ungrouped", side_effect=retrieve) as vector_calls, \
             patch("knowledge.processor.query_processor.nodes.bm25_search_node.StorageClients.get_milvus", return_value=db), \
             patch("knowledge.processor.query_processor.nodes.rerank_node.AIClients.get_bge_reranker_client", return_value=reranker), \
             patch.object(EvidenceEvaluatorNode, "_evaluate", side_effect=[
                 EvidenceEvaluation(sufficient=False, covered_sub_questions=[0], missing_subquestion_indices=[1],
                     followup_tasks=[RetrievalTask(subquestion_index=1, search_queries=["B追加证据"])], confidence=.95),
                 EvidenceEvaluation(sufficient=True, covered_sub_questions=[0, 1], confidence=.95)]):
            result = graph.compile().invoke(state)
        self.assertEqual(6, result["agent_tool_calls"])
        self.assertEqual(["Q0", "Q1", "B追加证据"], [c.args[0]["rewritten_query"] for c in vector_calls.call_args_list])
        self.assertEqual({"Q0", "Q1", "B追加证据"}, set(d["chunk_id"] for d in result["expanded_docs"]))
        self.assertFalse(result["conflict_judge_triggered"])
        prompt = AnswerOutputNode(self.config)._build_answer_prompt(result, 1200)
        self.assertIn("子问题0：Q0", prompt)
        self.assertEqual({"Q0", "Q1", "B追加证据"}, set(result["final_context_chunk_ids"]))


if __name__ == "__main__":
    unittest.main()
