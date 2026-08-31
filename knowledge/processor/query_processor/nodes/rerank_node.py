import math
import time
from typing import Any, Dict, List

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.ai_clients import AIClients


class RerankNode(BaseNode):
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取可独立理解的查询。
        query = (
            state.get("rewritten_query")
            or state.get("original_query")
            or ""
        ).strip()

        # 2. RRF 已经融合当前轮检索结果。第二轮开始时，把历史证据与
        # 当前候选合并后统一评分，避免旧证据天然排在新证据前面。
        rrf_chunks = state.get("rrf_chunks") or []

        # 3. 保留文件、完整章节路径和父块摘要，一起交给重排模型。
        if state.get("agentic_active"):
            merge_docs = self._merge_agentic_candidates(state, rrf_chunks)
        else:
            merge_docs = self._format_local_docs(rrf_chunks)

        # 4. 同时评估完整问题，以及真正召回每个候选的聚焦检索式。
        retrieval_queries = [
            item.strip()
            for item in (state.get("retrieval_queries") or [])
            if isinstance(item, str) and item.strip()
        ]
        started_at = time.perf_counter()
        reranked_docs_scores = self._rerank_merge_docs(
            query,
            merge_docs,
            retrieval_queries,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self.logger.info(
            "检索式感知重排完成: candidates=%s, focused_queries=%s, elapsed_ms=%.1f",
            len(merge_docs),
            max(len(retrieval_queries) - 1, 0),
            elapsed_ms,
        )
        state["rerank_candidates"] = reranked_docs_scores

        if state.get("agentic_active"):
            pool_limit = min(
                self.config.agent_evidence_pool_max,
                len(reranked_docs_scores),
            )
            evidence_pool = self._select_with_subquery_coverage(
                state,
                reranked_docs_scores,
                reranked_docs_scores[:pool_limit],
                max_results_override=pool_limit,
            )
            state["agent_evidence_pool"] = evidence_pool
            state["reranked_docs"] = evidence_pool[:self.config.rerank_max_top_k]
        else:
            # 非 Agentic 链路保持原有断崖检测行为。
            state["reranked_docs"] = self._truncate_at_score_cliff(
                reranked_docs_scores,
                self.config.rerank_min_top_k,
                self.config.rerank_max_top_k,
                self.config.rerank_gap_abs,
            )
        return state

    def _merge_agentic_candidates(
            self,
            state: QueryGraphState,
            current: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        iteration = max(int(state.get("agent_iteration") or 1), 1)
        previous = state.get("agent_evidence_pool") or []
        previous_ids = {
            self._document_id(doc)
            for doc in previous
            if isinstance(doc, dict)
        }
        merged: Dict[str, Dict[str, Any]] = {}

        for document in previous:
            if not isinstance(document, dict):
                continue
            identifier = self._document_id(document)
            merged[identifier] = {
                **document,
                "agent_first_seen_iteration": int(
                    document.get("agent_first_seen_iteration") or 1
                ),
                "agent_latest_seen_iteration": int(
                    document.get("agent_latest_seen_iteration") or max(iteration - 1, 1)
                ),
            }

        new_chunk_ids = []
        for document in current:
            if not isinstance(document, dict):
                continue
            identifier = self._document_id(document)
            if identifier not in previous_ids:
                new_chunk_ids.append(identifier)
            existing = merged.get(identifier, {})
            first_seen = int(
                existing.get("agent_first_seen_iteration") or iteration
            )
            retrieval_queries_hit = list(
                existing.get("retrieval_queries_hit") or []
            )
            merged[identifier] = {
                **existing,
                **document,
                "retrieval_queries_hit": list(dict.fromkeys(
                    retrieval_queries_hit
                )),
                "agent_first_seen_iteration": min(first_seen, iteration),
                "agent_latest_seen_iteration": iteration,
            }

        state["agent_new_chunk_ids"] = new_chunk_ids
        return self._format_local_docs(list(merged.values()))

    @staticmethod
    def _document_id(document: Dict[str, Any]) -> str:
        return str(
            document.get("chunk_id")
            or document.get("id")
            or f"{document.get('doc_id')}:{document.get('title')}:{document.get('content')}"
        )

    def _format_local_docs(
            self,
            rrf_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将 RRF 结果规整为重排文档，不丢失层级元数据。"""
        merge_docs: List[Dict[str, Any]] = []
        for rrf_chunk in rrf_chunks:
            if not isinstance(rrf_chunk, dict):
                continue
            content = (rrf_chunk.get("content") or "").strip()
            if not content:
                continue
            merge_docs.append({
                **rrf_chunk,
                "content": content,
                "title": (rrf_chunk.get("title") or "").strip(),
                "source": "local",
            })
        return merge_docs

    #分数归一化，传进来一个分数将其转换到0到1的范围
    def _score_normalize(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))



    def _rerank_merge_docs(
            self,
            query: str,
            merge_docs: List[Dict[str, Any]],
            retrieval_queries: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        if not query or not merge_docs:
            return []

        # 第一次调用时加载模型，后续调用复用AIClients中的单例。
        try:
            bge_reranker_client = AIClients.get_bge_reranker_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to BGE reranker,{e}")
            return []

        # 第一组 pair 计算所有候选对完整问题的相关性；后续只为确实由
        # 某条 focused search query 召回的候选追加 pair，避免无效笛卡尔积。
        rerank_texts = [self._build_rerank_text(doc) for doc in merge_docs]
        query_doc_content_pairs = [
            (query, rerank_text)
            for rerank_text in rerank_texts
        ]
        focused_pair_specs: List[tuple[int, str, str]] = []
        for doc_index, doc in enumerate(merge_docs):
            rrf_rank = doc.get("rrf_rank")
            if (
                    isinstance(rrf_rank, int)
                    and rrf_rank > self.config.rerank_coverage_query_rank_limit
            ):
                continue
            for query_name, retrieval_query in self._focused_queries_for_doc(
                    doc,
                    query,
                    retrieval_queries or [],
            ):
                query_doc_content_pairs.append(
                    (retrieval_query, rerank_texts[doc_index])
                )
                focused_pair_specs.append(
                    (doc_index, query_name, retrieval_query)
                )

        try:
            # 计算 rerank 分数
            rerank_scores = bge_reranker_client.compute_score(query_doc_content_pairs)

            # ===== compute_score 单文档防护 =====
            if isinstance(rerank_scores, (float, int)):
                rerank_scores = [rerank_scores]

            if len(rerank_scores) != len(query_doc_content_pairs):
                raise RuntimeError(
                    "重排分数数量与查询文档对数量不一致: "
                    f"scores={len(rerank_scores)}, "
                    f"pairs={len(query_doc_content_pairs)}"
                )

            normalized_scores = [
                float(self._score_normalize(score))
                for score in rerank_scores
            ]
            full_query_scores = normalized_scores[:len(merge_docs)]
            retrieval_query_scores: List[Dict[str, float]] = [
                {} for _ in merge_docs
            ]
            retrieval_query_texts: List[Dict[str, str]] = [
                {} for _ in merge_docs
            ]
            for score, (doc_index, query_name, retrieval_query) in zip(
                    normalized_scores[len(merge_docs):],
                    focused_pair_specs,
            ):
                retrieval_query_scores[doc_index][query_name] = score
                retrieval_query_texts[doc_index][query_name] = retrieval_query

            full_weight, retrieval_weight = self._normalized_rerank_weights()
            score_doc = []
            for doc_index, doc in enumerate(merge_docs):
                full_score = full_query_scores[doc_index]
                focused_scores = retrieval_query_scores[doc_index]
                if focused_scores:
                    best_query_name, best_focused_score = max(
                        focused_scores.items(),
                        key=lambda item: item[1],
                    )
                    final_score = (
                        full_weight * full_score
                        + retrieval_weight * best_focused_score
                    )
                else:
                    best_query_name = None
                    best_focused_score = None
                    final_score = full_score

                score_doc.append({
                    **doc,
                    "full_query_score": full_score,
                    "rerank_full_query_text": query,
                    "retrieval_query_scores": focused_scores,
                    "retrieval_query_texts": retrieval_query_texts[doc_index],
                    "best_retrieval_query_name": best_query_name,
                    "best_retrieval_query_score": best_focused_score,
                    "retrieval_queries_hit": list(dict.fromkeys(
                        [
                            *(doc.get("retrieval_queries_hit") or []),
                            *retrieval_query_texts[doc_index].values(),
                        ]
                    )),
                    "final_rerank_score": final_score,
                    "score": final_score,
                    "rerank_score": final_score,
                })

            #  排序并返回
            sorted_score_docs = sorted(score_doc, key=lambda x: x["score"], reverse=True)
            return [
                {**doc, "bge_rank": rank}
                for rank, doc in enumerate(sorted_score_docs, start=1)
            ]

        except Exception as e:
            self.logger.error(f"Rerank 重排序失败: {str(e)}")
            return [
                {
                    **doc,
                    "score": None,
                    "rerank_score": None,
                    "full_query_score": None,
                    "retrieval_query_scores": {},
                    "best_retrieval_query_score": None,
                    "final_rerank_score": None,
                    "bge_rank": rank,
                }
                for rank, doc in enumerate(merge_docs, start=1)
            ]

    def _normalized_rerank_weights(self) -> tuple[float, float]:
        full_weight = float(self.config.rerank_full_query_weight or 0.0)
        retrieval_weight = float(
            self.config.rerank_retrieval_query_weight or 0.0
        )
        total = full_weight + retrieval_weight
        if total <= 0:
            return 1.0, 0.0
        return full_weight / total, retrieval_weight / total

    @staticmethod
    def _same_query(left: str, right: str) -> bool:
        normalize = lambda value: "".join(value.lower().split())
        return normalize(left) == normalize(right)

    @classmethod
    def _focused_queries_for_doc(
            cls,
            doc: Dict[str, Any],
            full_query: str,
            retrieval_queries: List[str],
    ) -> List[tuple[str, str]]:
        focused: List[tuple[tuple[int, int, int], str, str]] = []
        seen: set[str] = set()

        # 历史证据保留最初命中它的真实检索表达，第二轮统一重排时继续
        # 使用，避免 query_0/query_1 在不同轮次发生语义错位。
        historical_queries = [
            *(doc.get("retrieval_queries_hit") or []),
            *(doc.get("retrieval_query_texts") or {}).values(),
        ]
        for index, retrieval_query in enumerate(historical_queries):
            text = str(retrieval_query or "").strip()
            key = "".join(text.lower().split())
            if not text or key in seen or cls._same_query(full_query, text):
                continue
            seen.add(key)
            focused.append(((0, index, 0), f"history_{index}", text))

        for query_index, retrieval_query in enumerate(retrieval_queries):
            query_name = f"query_{query_index}"
            text = str(retrieval_query or "").strip()
            key = "".join(text.lower().split())
            if (
                    not text
                    or key in seen
                    or cls._same_query(full_query, text)
                    or not cls._was_recalled_by_query(doc, query_name)
            ):
                continue
            seen.add(key)
            branch_ranks = (doc.get("subquery_ranks") or {}).get(
                query_name,
                {},
            )
            best_branch_rank = min(
                (
                    rank
                    for rank in branch_ranks.values()
                    if isinstance(rank, int) and rank > 0
                ),
                default=10 ** 9,
            )
            focused.append(
                ((1, best_branch_rank, query_index), query_name, text)
            )

        # 每个候选最多追加一个 focused query。历史证据优先沿用最初
        # 命中它的检索表达；本轮候选选择分支名次最靠前的检索式。
        focused.sort(key=lambda item: item[0])
        return [
            (query_name, text)
            for _priority, query_name, text in focused[:1]
        ]

    @staticmethod
    def _was_recalled_by_query(
            doc: Dict[str, Any],
            query_name: str,
    ) -> bool:
        subquery_ranks = doc.get("subquery_ranks") or {}
        if not isinstance(subquery_ranks, dict):
            return False
        branch_ranks = subquery_ranks.get(query_name) or {}
        if not isinstance(branch_ranks, dict):
            return False
        return any(
            isinstance(rank, int) and rank > 0
            for rank in branch_ranks.values()
        )

    @staticmethod
    def _build_rerank_text(doc: Dict[str, Any]) -> str:
        """构建带章节语义的重排文本。"""
        file_title = (
            doc.get("canonical_title")
            or doc.get("file_title")
            or ""
        ).strip()
        primary_subject = (doc.get("primary_subject") or "").strip()
        section_path = (doc.get("section_path") or "").strip()
        parent_title = (doc.get("parent_title") or "").strip()
        parent_summary = (doc.get("parent_summary") or "").strip()
        title = (doc.get("title") or "").strip()
        content = (doc.get("content") or "").strip()

        parts = []
        if file_title:
            parts.append(f"文档：{file_title[:80]}")
        if primary_subject and primary_subject != file_title:
            parts.append(f"主要对象：{primary_subject[:80]}")
        if section_path:
            parts.append(f"章节路径：{section_path[:160]}")
        if parent_title:
            parts.append(f"父标题：{parent_title[:80]}")
        if parent_summary:
            # BGE-Reranker 的输入有 token 上限，摘要只用作语义辅助，
            # 避免过长摘要把真正的命中切片挤出截断窗口。
            parts.append(f"父块摘要：{parent_summary[:180]}")
        if title and not content.startswith(title):
            parts.append(f"切片标题：{title[:80]}")
        if content:
            parts.append(f"切片内容：{content}")
        return "\n".join(parts)

    def _truncate_at_score_cliff(self, reranked_docs_scores:List[Dict[str,Any]], rerank_min_top_k:int, rerank_max_top_k:int, rerank_gap_abs:float):

        #1 定义截取的上下边界
        upper_bound = min(rerank_max_top_k, len(reranked_docs_scores))
        lower_bound = min(rerank_min_top_k ,upper_bound)

        # 重排模型不可用时，_rerank_merge_docs 会保留候选并将 score 设为 None。
        # 此时沿用 RRF/原始合并顺序，不能继续做分数差值计算。
        if any(
                not isinstance(document.get("score"), (int, float))
                for document in reranked_docs_scores[:upper_bound]
        ):
            return reranked_docs_scores[:upper_bound]

        #2 遍历reranked_docs
        max_score_gap = 0
        max_score_index = upper_bound
        #在扫描范围内寻找最大截断点   lower_bound - 1，
        for i in range(lower_bound - 1,upper_bound-1):
            #当前文档的分数
            current_document_score = reranked_docs_scores[i].get("score")
            next_document_score = reranked_docs_scores[i+1].get("score")

            current_gap = current_document_score-next_document_score
            if current_gap  > rerank_gap_abs and current_gap > max_score_gap:
                max_score_gap = current_gap
                max_score_index = i + 1

        #对文档进行截取
        return reranked_docs_scores[0:max_score_index]

    def _select_with_subquery_coverage(
            self,
            state: QueryGraphState,
            ranked_docs: List[Dict[str, Any]],
            truncated_docs: List[Dict[str, Any]],
            max_results_override: int | None = None,
    ) -> List[Dict[str, Any]]:
        """基于答案事实维度评分，而不是基于检索分支名次保护证据。"""
        if (
                not self.config.rerank_coverage_enabled
                or not state.get("agentic_active")
                or not ranked_docs
        ):
            return truncated_docs

        plan = state.get("agent_plan") or {}
        if not isinstance(plan, dict):
            return truncated_docs
        intent = str(plan.get("intent") or "").strip().lower()
        if intent not in {"comparison", "multi_fact", "multi_hop"}:
            return truncated_docs
        sub_questions = []
        for item in plan.get("sub_questions") or []:
            question = str(item or "").strip()
            if question and question not in sub_questions:
                sub_questions.append(question)
            if len(sub_questions) >= self.config.agent_max_subquestions:
                break
        if not sub_questions:
            return truncated_docs

        max_results = min(
            max(int(
                max_results_override
                if max_results_override is not None
                else self.config.rerank_max_top_k
            ), 0),
            len(ranked_docs),
        )
        if max_results <= 0:
            return []

        candidate_limit = min(
            max(int(self.config.rerank_coverage_candidate_top_k or 1), 1),
            len(ranked_docs),
        )
        candidates = self._coverage_candidate_pool(
            ranked_docs,
            candidate_limit,
        )
        min_score = float(self.config.rerank_coverage_min_score or 0.0)
        coverage_scores = self._score_subquestion_coverage(
            sub_questions,
            candidates,
        )
        if not coverage_scores:
            return truncated_docs

        protected: Dict[Any, Dict[str, Any]] = {}
        missing_subquestions = []
        for subquestion_index, _question in enumerate(sub_questions):
            best_candidate_index = max(
                range(len(candidates)),
                key=lambda index: coverage_scores[index][subquestion_index],
            )
            best_score = coverage_scores[best_candidate_index][subquestion_index]
            if best_score < min_score:
                missing_subquestions.append(subquestion_index)
                continue
            best = candidates[best_candidate_index]
            chunk_id = best.get("chunk_id") or best.get("id")
            if chunk_id is None:
                missing_subquestions.append(subquestion_index)
                continue
            protected_doc = protected.setdefault(chunk_id, dict(best))
            protected_doc.setdefault(
                "coverage_subquestion_indices", []
            ).append(subquestion_index)

        for candidate_index, candidate in enumerate(candidates):
            candidate["subquestion_coverage_scores"] = {
                f"subquestion_{index}": score
                for index, score in enumerate(coverage_scores[candidate_index])
            }
            chunk_id = candidate.get("chunk_id") or candidate.get("id")
            if chunk_id in protected:
                protected[chunk_id].update(candidate)

        state["coverage_missing_subquestion_indices"] = missing_subquestions
        state["coverage_protected_chunk_ids"] = list(protected)

        # 先放覆盖保护证据，再按综合重排分补齐剩余位置。一个文档可以
        # 覆盖多个事实维度，但只占一个 Top-K 位置。
        protected_docs = [
            {
                **doc,
                "coverage_protected": True,
                "coverage_subquestion_indices": sorted(set(
                    doc.get("coverage_subquestion_indices") or []
                )),
            }
            for doc in protected.values()
        ]
        protected_docs.sort(key=self._document_score, reverse=True)
        selected = protected_docs[:max_results]
        selected_ids = {
            doc.get("chunk_id") or doc.get("id") for doc in selected
        }
        candidate_by_id = {
            doc.get("chunk_id") or doc.get("id"): doc for doc in candidates
        }
        for ranked_doc in ranked_docs:
            if len(selected) >= max_results:
                break
            chunk_id = ranked_doc.get("chunk_id") or ranked_doc.get("id")
            if chunk_id in selected_ids:
                continue
            selected.append(dict(candidate_by_id.get(chunk_id, ranked_doc)))
            selected_ids.add(chunk_id)

        self.logger.info(
            "子问题覆盖保护完成: subquestions=%s, protected=%s, missing=%s, output=%s",
            len(sub_questions),
            len(protected),
            len(missing_subquestions),
            len(selected),
        )
        return selected

    def _coverage_candidate_pool(
            self,
            ranked_docs: List[Dict[str, Any]],
            candidate_limit: int,
    ) -> List[Dict[str, Any]]:
        """Reserve coverage slots for RRF leaders, then fill by BGE rank."""
        if candidate_limit <= 0:
            return []

        rrf_seed_limit = min(
            max(int(self.config.rerank_max_top_k or 1), 1),
            candidate_limit,
        )
        rrf_ranked = sorted(
            (
                document
                for document in ranked_docs
                if isinstance(document.get("rrf_rank"), int)
                and document["rrf_rank"] > 0
            ),
            key=lambda document: document["rrf_rank"],
        )

        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        for document in rrf_ranked[:rrf_seed_limit]:
            identifier = self._document_id(document)
            if identifier in selected_ids:
                continue
            selected.append(dict(document))
            selected_ids.add(identifier)

        for document in ranked_docs:
            if len(selected) >= candidate_limit:
                break
            identifier = self._document_id(document)
            if identifier in selected_ids:
                continue
            selected.append(dict(document))
            selected_ids.add(identifier)
        return selected

    def _score_subquestion_coverage(
            self,
            sub_questions: List[str],
            candidates: List[Dict[str, Any]],
    ) -> List[List[float]]:
        try:
            matrix: List[List[float | None]] = [
                [None] * len(sub_questions) for _ in candidates
            ]
            pairs: List[tuple[str, str]] = []
            pair_specs: List[tuple[int, int]] = []
            for candidate_index, candidate in enumerate(candidates):
                rerank_text = self._build_rerank_text(candidate)
                for question_index, question in enumerate(sub_questions):
                    reused = self._reuse_existing_query_score(
                        candidate,
                        question,
                    )
                    if reused is not None:
                        matrix[candidate_index][question_index] = reused
                        continue
                    pairs.append((question, rerank_text))
                    pair_specs.append((candidate_index, question_index))

            if pairs:
                client = AIClients.get_bge_reranker_client()
                scores = client.compute_score(pairs)
                if isinstance(scores, (float, int)):
                    scores = [scores]
                if len(scores) != len(pairs):
                    raise RuntimeError(
                        "覆盖评分数量与查询文档对数量不一致: "
                        f"scores={len(scores)}, pairs={len(pairs)}"
                    )
                for raw_score, (candidate_index, question_index) in zip(
                        scores,
                        pair_specs,
                ):
                    matrix[candidate_index][question_index] = (
                        self._score_normalize(float(raw_score))
                    )

            return [
                [float(score) for score in row]
                for row in matrix
            ]
        except Exception as exc:
            self.logger.error("子问题覆盖评分失败: %s", exc)
            return []

    @classmethod
    def _reuse_existing_query_score(
            cls,
            candidate: Dict[str, Any],
            query: str,
    ) -> float | None:
        if cls._same_query(
                str(candidate.get("rerank_full_query_text") or ""),
                query,
        ):
            score = candidate.get("full_query_score")
            if isinstance(score, (int, float)):
                return float(score)

        query_texts = candidate.get("retrieval_query_texts") or {}
        query_scores = candidate.get("retrieval_query_scores") or {}
        if not isinstance(query_texts, dict) or not isinstance(
                query_scores,
                dict,
        ):
            return None
        for query_name, query_text in query_texts.items():
            if not cls._same_query(str(query_text or ""), query):
                continue
            score = query_scores.get(query_name)
            if isinstance(score, (int, float)):
                return float(score)
        return None

    @staticmethod
    def _document_score(doc: Dict[str, Any]) -> float:
        score = doc.get("final_rerank_score", doc.get("score"))
        return float(score) if isinstance(score, (int, float)) else float("-inf")
