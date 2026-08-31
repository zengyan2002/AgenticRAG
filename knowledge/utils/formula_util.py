"""Utilities for preserving and indexing LaTeX formulas in Markdown."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re
from typing import Dict, Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class FormulaMatch:
    """A non-overlapping LaTeX formula found in Markdown text."""

    start: int
    end: int
    raw: str
    latex: str
    display: bool


class FormulaProcessor:
    """Extract formulas and split Markdown without cutting formula bodies."""

    _BLOCK_PATTERNS = (
        re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL),
        re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
        re.compile(
            r"\\begin\{(?P<env>equation\*?|align\*?|aligned|gather\*?|multline\*?|cases|"
            r"matrix|pmatrix|bmatrix|vmatrix|Vmatrix)\}.*?\\end\{(?P=env)\}",
            re.DOTALL,
        ),
    )
    _INLINE_PATTERNS = (
        re.compile(r"\\\((.+?)\\\)", re.DOTALL),
        re.compile(r"(?<![\\$])\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)"),
    )

    _COMMAND_KEYWORDS: Dict[str, str] = {
        "frac": "fraction division 分数 除法",
        "sqrt": "square root 平方根",
        "sum": "summation 求和",
        "prod": "product 连乘",
        "int": "integral 积分",
        "iint": "double integral 二重积分",
        "iiint": "triple integral 三重积分",
        "partial": "partial derivative 偏导数",
        "nabla": "gradient 梯度",
        "lim": "limit 极限",
        "log": "logarithm 对数",
        "ln": "natural logarithm 自然对数",
        "sin": "sine 正弦",
        "cos": "cosine 余弦",
        "tan": "tangent 正切",
        "exp": "exponential 指数",
        "times": "multiplication 乘法",
        "cdot": "multiplication 点乘",
        "le": "less than or equal 小于等于",
        "ge": "greater than or equal 大于等于",
        "neq": "not equal 不等于",
        "approx": "approximately equal 约等于",
        "infty": "infinity 无穷",
        "alpha": "alpha 阿尔法",
        "beta": "beta 贝塔",
        "gamma": "gamma 伽马",
        "delta": "delta 德尔塔",
        "theta": "theta 西塔",
        "lambda": "lambda 拉姆达",
        "mu": "mu 缪",
        "sigma": "sigma 西格玛",
        "phi": "phi 斐",
        "omega": "omega 欧米伽",
    }

    @classmethod
    def find(cls, text: str) -> List[FormulaMatch]:
        """Return non-overlapping formulas in source order."""
        if not text:
            return []

        candidates: List[FormulaMatch] = []
        for pattern in cls._BLOCK_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                candidates.append(
                    FormulaMatch(
                        start=match.start(),
                        end=match.end(),
                        raw=raw,
                        latex=cls._strip_delimiters(raw),
                        display=True,
                    )
                )

        for pattern in cls._INLINE_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                candidates.append(
                    FormulaMatch(
                        start=match.start(),
                        end=match.end(),
                        raw=raw,
                        latex=cls._strip_delimiters(raw),
                        display=False,
                    )
                )

        # Prefer the widest match at the same position. This keeps an outer
        # $$...$$ block intact when it contains an aligned environment.
        candidates.sort(key=lambda item: (item.start, -(item.end - item.start)))
        formulas: List[FormulaMatch] = []
        occupied_until = -1
        for candidate in candidates:
            if candidate.start < occupied_until:
                continue
            formulas.append(candidate)
            occupied_until = candidate.end
        return formulas

    @classmethod
    def extract(cls, text: str) -> List[Dict[str, object]]:
        """Build serializable formula metadata for a document chunk."""
        result: List[Dict[str, object]] = []
        for formula in cls.find(text):
            formula_id = sha1(formula.raw.encode("utf-8")).hexdigest()[:16]
            result.append(
                {
                    "formula_id": formula_id,
                    "latex": formula.latex,
                    "raw": formula.raw,
                    "display": formula.display,
                    "keywords": cls._formula_keywords(formula.latex),
                }
            )
        return result

    @classmethod
    def build_search_text(cls, formulas: Iterable[Dict[str, object]]) -> str:
        """Create embedding-friendly text while retaining the original LaTeX."""
        lines: List[str] = []
        for index, formula in enumerate(formulas, start=1):
            latex = str(formula.get("latex") or "").strip()
            if not latex:
                continue
            keywords = str(formula.get("keywords") or "").strip()
            formula_type = "块级公式" if formula.get("display") else "行内公式"
            line = f"公式{index}；类型：{formula_type}；LaTeX：{latex}"
            if keywords:
                line += f"；检索语义：{keywords}"
            description = str(formula.get("description") or "").strip()
            if description:
                line += f"；含义：{description}"
            variables = formula.get("variables") or []
            variable_texts: List[str] = []
            if isinstance(variables, list):
                for variable in variables:
                    if not isinstance(variable, dict):
                        continue
                    symbol = str(variable.get("symbol") or "").strip()
                    meaning = str(variable.get("meaning") or "").strip()
                    unit = str(variable.get("unit") or "").strip()
                    if not symbol:
                        continue
                    variable_text = symbol
                    if meaning:
                        variable_text += f"表示{meaning}"
                    if unit:
                        variable_text += f"，单位{unit}"
                    variable_texts.append(variable_text)
            if variable_texts:
                line += "；变量说明：" + "；".join(variable_texts)
            conditions = formula.get("conditions") or []
            if isinstance(conditions, list):
                condition_text = "；".join(
                    str(item).strip() for item in conditions if str(item).strip()
                )
                if condition_text:
                    line += f"；适用条件：{condition_text}"
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    def protect(cls, text: str) -> tuple[str, Dict[str, str]]:
        """Replace formulas with stable tokens for non-formula transforms."""
        formulas = cls.find(text)
        if not formulas:
            return text, {}

        parts: List[str] = []
        mapping: Dict[str, str] = {}
        cursor = 0
        for index, formula in enumerate(formulas):
            token = f"FORMULAANCHOR{index:06d}X"
            parts.append(text[cursor:formula.start])
            parts.append(token)
            mapping[token] = formula.raw
            cursor = formula.end
        parts.append(text[cursor:])
        return "".join(parts), mapping

    @staticmethod
    def restore(text: str, mapping: Dict[str, str]) -> str:
        """Restore formulas previously replaced by :meth:`protect`."""
        restored = text
        for token, formula in mapping.items():
            restored = restored.replace(token, formula)
        return restored

    @classmethod
    def split_text_preserving_formulas(
        cls,
        text: str,
        chunk_size: int,
        separators: List[str] | None = None,
    ) -> List[str]:
        """Split text while treating every formula as an indivisible segment.

        A formula longer than ``chunk_size`` is returned as one oversized
        chunk. Preserving mathematical correctness is more important than the
        configured soft length limit in that edge case.
        """
        if not text:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        formulas = cls.find(text)
        if not formulas:
            return cls._split_plain_text(text, chunk_size, separators)

        segments: List[tuple[str, bool]] = []
        cursor = 0
        for formula in formulas:
            if formula.start > cursor:
                segments.append((text[cursor:formula.start], False))
            segments.append((formula.raw, True))
            cursor = formula.end
        if cursor < len(text):
            segments.append((text[cursor:], False))

        chunks: List[str] = []
        current = ""

        def flush_current() -> None:
            nonlocal current
            value = current.strip()
            if value:
                chunks.append(value)
            current = ""

        def append_piece(piece: str, is_formula: bool = False) -> None:
            nonlocal current
            if not piece:
                return
            if current and len(current) + len(piece) > chunk_size:
                flush_current()
            if is_formula and len(piece) > chunk_size:
                flush_current()
                chunks.append(piece.strip())
                return
            current += piece

        for segment, is_formula in segments:
            if is_formula:
                append_piece(segment, is_formula=True)
                continue
            for piece in cls._split_plain_text(segment, chunk_size, separators):
                append_piece(piece)

        flush_current()
        return chunks

    @classmethod
    def _split_plain_text(
        cls,
        text: str,
        chunk_size: int,
        separators: List[str] | None,
    ) -> List[str]:
        if not text or not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=separators
            or ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            strip_whitespace=False,
        )
        return splitter.split_text(text)

    @classmethod
    def _strip_delimiters(cls, raw: str) -> str:
        value = raw.strip()
        if value.startswith("$$") and value.endswith("$$"):
            return value[2:-2].strip()
        if value.startswith(r"\[") and value.endswith(r"\]"):
            return value[2:-2].strip()
        if value.startswith(r"\(") and value.endswith(r"\)"):
            return value[2:-2].strip()
        if value.startswith("$") and value.endswith("$"):
            return value[1:-1].strip()
        return value

    @classmethod
    def _formula_keywords(cls, latex: str) -> str:
        commands = re.findall(r"\\([A-Za-z]+)", latex)
        command_keywords: List[str] = []
        for command in commands:
            keyword = cls._COMMAND_KEYWORDS.get(command.lower())
            if keyword and keyword not in command_keywords:
                command_keywords.append(keyword)

        without_commands = re.sub(r"\\[A-Za-z]+", " ", latex)
        variables = []
        for variable in re.findall(r"(?<![A-Za-z])[A-Za-z](?:_[A-Za-z0-9]+)?", without_commands):
            if variable not in variables:
                variables.append(variable)

        parts = command_keywords
        if variables:
            parts = parts + ["变量符号 " + " ".join(variables[:20])]
        return "；".join(parts)
