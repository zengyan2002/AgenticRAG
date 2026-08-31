"""LLM-based semantic enrichment for formulas extracted from documents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import Any, Dict, Iterable, List

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.prompts.import_prompt import (
    FORMULA_SEMANTIC_SYSTEM_PROMPT,
    FORMULA_SEMANTIC_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.formula_util import FormulaProcessor


class FormulaSemanticNode(BaseNode):
    """Enrich useful formulas in batches without blocking deterministic indexing."""

    name = "formula_semantic_node"

    _SEMANTIC_MARKERS = re.compile(
        r"[=<>+\-*/]|"
        r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|iint|iiint|lim|log|ln|"
        r"sin|cos|tan|exp|det|tr|trace|nabla|partial|begin|cases|matrix|"
        r"leq?|geq?|neq|approx|propto|sim|equiv|in|times|cdot|pm)\b",
        re.IGNORECASE,
    )

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks = state.get("chunks") or []
        if not isinstance(chunks, list) or not self.config.formula_semantic_enabled:
            return state

        formula_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("formulas")
        ]
        if not formula_chunks:
            return state

        work_items, formula_references = self._collect_work_items(formula_chunks)
        if not work_items:
            self._refresh_formula_search_text(formula_chunks)
            self.logger.info("没有需要调用 LLM 的复杂公式，已保留确定性公式检索信息")
            return state

        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except Exception as exc:
            self.logger.warning("公式语义模型不可用，保留确定性检索信息: %s", exc)
            self._refresh_formula_search_text(formula_chunks)
            return state

        batch_size = max(1, self.config.formula_semantic_batch_size)
        batches = [
            work_items[index:index + batch_size]
            for index in range(0, len(work_items), batch_size)
        ]
        max_workers = min(
            max(1, self.config.formula_semantic_max_workers),
            len(batches),
        )
        self.logger.info(
            "公式语义批处理开始: formula_chunks=%s, unique_formulas=%s, "
            "batches=%s, workers=%s",
            len(formula_chunks),
            len(work_items),
            len(batches),
            max_workers,
        )

        enrichment_by_id: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="formula-semantic",
        ) as executor:
            futures = {
                executor.submit(
                    self._analyze_batch,
                    batch,
                    llm_client,
                    str(state.get("theme_name") or state.get("file_title") or ""),
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    for item in future.result():
                        formula_id = item.get("formula_id")
                        if formula_id:
                            enrichment_by_id[formula_id] = item
                except Exception as exc:
                    # 公式语义说明属于召回增强，不应阻断正文入库。
                    self.logger.warning(
                        "公式语义批次增强失败，使用原始 LaTeX 继续入库: "
                        "formula_count=%s, reason=%s",
                        len(batch),
                        exc,
                    )

        self._apply_enrichments(formula_references, enrichment_by_id)
        self._refresh_formula_search_text(formula_chunks)
        self.logger.info(
            "公式语义批处理完成: requested=%s, enriched=%s, batches=%s",
            len(work_items),
            len(enrichment_by_id),
            len(batches),
        )
        return state

    def _collect_work_items(
        self,
        formula_chunks: Iterable[Dict[str, Any]],
    ) -> tuple[List[Dict[str, str]], Dict[str, List[Dict[str, Any]]]]:
        """Select useful formulas, deduplicate them and keep all update targets."""
        work_by_id: Dict[str, Dict[str, str]] = {}
        formula_references: Dict[str, List[Dict[str, Any]]] = {}
        max_per_chunk = max(1, self.config.formula_semantic_max_formulas)

        for chunk in formula_chunks:
            formulas = [
                formula
                for formula in (chunk.get("formulas") or [])
                if isinstance(formula, dict) and formula.get("formula_id")
            ]
            for formula in formulas:
                formula_references.setdefault(
                    str(formula["formula_id"]), []
                ).append(formula)

            selected = [
                formula for formula in formulas if self._needs_llm_enrichment(formula)
            ][:max_per_chunk]
            for formula in selected:
                formula_id = str(formula["formula_id"])
                if formula_id in work_by_id:
                    continue
                work_by_id[formula_id] = {
                    "formula_id": formula_id,
                    "latex": str(formula.get("latex") or ""),
                    "section_title": str(chunk.get("title") or ""),
                    "context": self._formula_context(chunk, formula),
                }

        return list(work_by_id.values()), formula_references

    @classmethod
    def _needs_llm_enrichment(cls, formula: Dict[str, Any]) -> bool:
        """Skip trivial inline symbols while retaining real equations."""
        if formula.get("display"):
            return True
        latex = re.sub(r"\s+", "", str(formula.get("latex") or ""))
        if not latex:
            return False
        if cls._SEMANTIC_MARKERS.search(latex):
            return True
        # Long inline expressions can still be meaningful even without an
        # explicit equality sign (for example, vector or set definitions).
        return len(latex) >= 24

    def _formula_context(
        self,
        chunk: Dict[str, Any],
        formula: Dict[str, Any],
    ) -> str:
        content = str(chunk.get("content") or "")
        limit = max(100, self.config.formula_semantic_context_chars_per_formula)
        if len(content) <= limit:
            return content

        raw = str(formula.get("raw") or "")
        position = content.find(raw) if raw else -1
        if position < 0:
            latex = str(formula.get("latex") or "")
            position = content.find(latex) if latex else -1
        if position < 0:
            return content[:limit]

        formula_length = max(len(raw), len(str(formula.get("latex") or "")))
        padding = max(0, limit - formula_length)
        start = max(0, position - padding // 2)
        end = min(len(content), start + limit)
        start = max(0, end - limit)
        return content[start:end]

    def _analyze_batch(
        self,
        batch: List[Dict[str, str]],
        llm_client: Any,
        theme_name: str,
    ) -> List[Dict[str, Any]]:
        user_prompt = FORMULA_SEMANTIC_USER_PROMPT_TEMPLATE.format(
            theme_name=theme_name,
            formulas_json=json.dumps(batch, ensure_ascii=False),
        )
        response = llm_client.invoke(
            [
                ("system", FORMULA_SEMANTIC_SYSTEM_PROMPT),
                ("user", user_prompt),
            ]
        )
        return self._parse_response(getattr(response, "content", response))

    @staticmethod
    def _apply_enrichments(
        formula_references: Dict[str, List[Dict[str, Any]]],
        enrichment_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        for formula_id, enrichment in enrichment_by_id.items():
            for formula in formula_references.get(formula_id, []):
                formula.update(
                    {
                        "description": enrichment.get("description", ""),
                        "variables": enrichment.get("variables", []),
                        "conditions": enrichment.get("conditions", []),
                    }
                )

    @staticmethod
    def _refresh_formula_search_text(
        formula_chunks: Iterable[Dict[str, Any]],
    ) -> None:
        for chunk in formula_chunks:
            chunk["formula_search_text"] = FormulaProcessor.build_search_text(
                chunk.get("formulas") or []
            )

    def _parse_response(self, content: Any) -> List[Dict[str, Any]]:
        if not isinstance(content, str):
            raise ValueError("公式语义模型返回内容不是字符串")
        value = content.strip()
        if value.startswith("```"):
            value = "\n".join(value.splitlines()[1:-1]).strip()
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end < start:
            raise ValueError("公式语义模型未返回完整 JSON")
        payload = json.loads(value[start:end + 1])
        items = payload.get("formulas") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("公式语义模型缺少 formulas 列表")

        result: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            formula_id = str(item.get("formula_id") or "").strip()
            if not formula_id:
                continue
            variables = item.get("variables")
            if not isinstance(variables, list):
                variables = []
            variables = [
                {
                    "symbol": str(variable.get("symbol") or "").strip(),
                    "meaning": str(variable.get("meaning") or "").strip(),
                    "unit": str(variable.get("unit") or "").strip(),
                }
                for variable in variables
                if isinstance(variable, dict) and variable.get("symbol")
            ]
            conditions = item.get("conditions")
            if not isinstance(conditions, list):
                conditions = []
            result.append(
                {
                    "formula_id": formula_id,
                    "description": str(item.get("description") or "").strip(),
                    "variables": variables,
                    "conditions": [
                        str(condition).strip()
                        for condition in conditions
                        if str(condition).strip()
                    ],
                }
            )
        return result
