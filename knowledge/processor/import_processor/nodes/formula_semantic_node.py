"""基于大语言模型，对从文档中提取的公式进行语义增强。"""

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
    """分批增强有价值公式的语义信息，同时不阻塞基于确定性规则的索引流程"""

    name = "formula_semantic_node"

    # 用来匹配是否值得进行语义解析的公式
    # 包含分数、根号、求和与连乘、积分、极限、对数、三角函数、矩阵、偏导与梯度、比较关系、近似关系、乘法关系、正负号
    _SEMANTIC_MARKERS = re.compile(
        r"[=<>+\-*/]|"
        r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|iint|iiint|lim|log|ln|"
        r"sin|cos|tan|exp|det|tr|trace|nabla|partial|begin|cases|matrix|"
        r"leq?|geq?|neq|approx|propto|sim|equiv|in|times|cdot|pm)\b",
        re.IGNORECASE,
    )

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """筛选并批量增强文档切片中的公式语义。

        该节点从状态中读取 ``chunks``，筛选包含公式且值得调用 LLM
        增强的内容，按 ``formula_id`` 去重后分批并发处理。成功返回的
        公式含义、变量说明和适用条件会回填到所有对应公式引用中，随后
        重新生成每个公式切片的 ``formula_search_text``。

        当公式语义增强被禁用、没有可处理公式、LLM 客户端不可用或某个
        批次处理失败时，节点保留原始 LaTeX 与确定性检索信息，避免公式
        语义增强故障阻断文档入库主流程。

        Args:
            state: 入库图状态，应包含待处理的 ``chunks``，并可提供
                ``primary_subject``、``canonical_title`` 或 ``file_title``
                作为公式分析的文档语境。

        Returns:
            更新后的原状态对象。公式增强成功时，其中的公式元数据与
            ``formula_search_text`` 会被原地更新。
        """
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
        # 以 formula_id 为键的临时结果表，用于收集各个并发批次成功返回的公式解释，并在全部处理完成后统一回填到公式引用中
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
                    str( state.get("primary_subject") or state.get("canonical_title") or state.get("file_title") or ""),
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
        """
        收集需要进行 LLM 语义增强的公式及其全部更新目标。

        遍历包含公式的文档切片，过滤缺少 ``formula_id`` 的非法公式，并记录每个公式在原始切片中的所有出现位置。随后筛选需要进行
        LLM 语义增强的公式，限制每个切片的处理数量，并根据``formula_id`` 对待处理公式进行全局去重。对每个唯一公式，组装 LaTeX、章节标题和公式附近的正文上下文，形成后续批量调用 LLM 所需的工作项。同一公式即使在多个切片中
        重复出现，也只调用一次 LLM；处理结果可通过引用映射回写到所有原始公式对象。

        Args:
            formula_chunks: 包含公式信息的文档切片集合。每个切片可包含``title``、``content`` 和 ``formulas`` 字段。

        Returns:
            一个二元组，包含：
            - 待处理工作项列表。每个工作项包含 ``formula_id``、``latex``、``section_title`` 和 ``context``。
            - 公式引用映射。键为 ``formula_id``，值为该公式在所有原始切片中的公式字典列表，用于批量解析后统一回写结果。
        """
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
        """
        判断公式是需要LLM进行语义增强

        块级公式通常具有相对完整的数学语义，因此直接进行增强。对于行内公式，先去除 LaTeX 中的空白字符，再检查其是否包含等号、运算符或常见数学命令等语义标志。
        未包含这些标志但长度较长的行内表达式，也可能是向量、集合定义等有意义的公式，因此同样需要进行增强。简单变量、空公式和较短且不含数学语义标志的行内表达式将被跳过，以减少不必要的 LLM 调用。

        Args:
            formula: 待判断的公式信息，可包含 ``display`` 和 ``latex``字段。``display`` 表示是否为块级公式，``latex`` 表示去除公式定界符后的 LaTeX 内容。

        Returns:
            如果公式需要进行 LLM 语义增强，返回 ``True``；否则返回``False``。
        """

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
        """截取目标公式附近的文档上下文，供大模型理解公式语义。

        当切片正文不超过配置长度时，直接返回完整正文；正文过长时，优先使用公式原文定位，定位失败后使用去除定界符的 LaTeX 内容定位，
        并以公式所在位置为中心截取指定长度的上下文。若两种方式均无法定位公式，则降级返回正文开头的指定长度内容。

        Args:
            chunk: 公式所在的文档切片，应包含 ``content`` 字段。
            formula: 目标公式信息，应包含 ``raw``、``latex`` 等字段。

        Returns:
            目标公式附近的上下文文本，长度原则上不超过``formula_semantic_context_chars_per_formula``，且最少按
            100 个字符的窗口进行截取。
        """
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
        """调用 LLM 批量分析公式，并解析为统一的增强结果。

        每个工作项包含公式 ID、LaTeX、章节标题和局部上下文。方法将
        整批公式连同文档主题组装进提示词，调用模型后交由
        ``_parse_response`` 完成 JSON 提取、校验和字段规范化。

        Args:
            batch: 当前批次的公式工作项列表。
            llm_client: 支持 ``invoke`` 方法的同步大语言模型客户端。
            theme_name: 用于辅助理解公式的文档主题、规范标题或文件名。

        Returns:
            规范化后的公式语义增强结果列表。

        Raises:
            Exception: 模型调用失败时透传客户端异常；返回内容无法解析
                时透传 ``_parse_response`` 抛出的解析或校验异常。异常会
                由上层并发任务收集逻辑捕获并执行降级处理。
        """
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
        """按公式 ID 将 LLM 语义增强结果回填到所有公式引用中。

        ``enrichment_by_id`` 为每个公式保存一份增强结果，
        ``formula_references`` 则保存同一公式在不同 chunk 中的所有引用。
        本方法会原地更新这些公式对象的含义、变量和适用条件，不返回新对象。
        """
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
        """根据最新公式元数据重新生成各 chunk 的公式检索文本。

        公式经过 LLM 语义增强后，``formulas`` 中可能新增含义、变量说明
        和适用条件。本方法会重新调用 ``FormulaProcessor.build_search_text``，
        将这些信息整理为适合向量化和 BM25 索引的文本，并原地写入每个
        chunk 的 ``formula_search_text`` 字段。

        Args:
            formula_chunks: 包含 ``formulas`` 字段的 chunk 可迭代对象。

        Returns:
            None。传入的 chunk 会被原地更新。
        """
        for chunk in formula_chunks:
            chunk["formula_search_text"] = FormulaProcessor.build_search_text(
                chunk.get("formulas") or []
            )

    def _parse_response(self, content: Any) -> List[Dict[str, Any]]:
        """解析并规范化公式语义模型返回的 JSON 内容。

        该方法会去除模型可能附加的 Markdown 代码围栏，截取最外层
        JSON 对象，校验其中的 ``formulas`` 列表，并过滤缺少
        ``formula_id`` 的无效记录。变量说明和适用条件会被整理为下游
        回填逻辑所需的统一结构。

        Args:
            content: 公式语义模型返回的原始文本。

        Returns:
            规范化后的公式增强结果列表。每项包含 ``formula_id``、
            ``description``、``variables`` 和 ``conditions``。

        Raises:
            ValueError: 返回值不是字符串、缺少完整 JSON 对象，或 JSON
                中不存在合法的 ``formulas`` 列表。
            json.JSONDecodeError: 截取出的内容不是合法 JSON。
        """
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
