"""Agentic RAG 结构化输出模型与解析辅助函数。"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


AgentIntent = Literal[
    "fact",
    "comparison",
    "multi_fact",
    "multi_hop",
    "formula",
    "image",
]
RetrievalProfile = Literal["fast", "deep", "comparison"]
RetrievalTool = Literal["vector", "hyde", "bm25"]


class AgentPlan(BaseModel):
    """Planner 生成的受控执行计划。"""

    intent: AgentIntent = "fact"
    objective: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    document_hints: list[str] = Field(default_factory=list)
    retrieval_profile: RetrievalProfile = "deep"
    retrieval_tools: list[RetrievalTool] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class EvidenceEvaluation(BaseModel):
    """Evidence Evaluator 对当前证据覆盖度的结构化判断。"""

    sufficient: bool = False
    covered_sub_questions: list[int] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    next_action: Literal[
        "answer",
        "rewrite_missing",
        "expand_scope",
        "ask_user",
    ] = "rewrite_missing"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("missing_aspects", "contradictions", mode="before")
    @classmethod
    def normalize_text_items(cls, value: Any) -> list[str]:
        """容忍模型把文本列表项输出成对象，避免整次证据评估回退。"""
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = "；".join(
                    str(part).strip()
                    for part in item.values()
                    if part is not None and str(part).strip()
                )
                if not text:
                    text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            else:
                text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized


def parse_json_object(content: Any) -> dict[str, Any]:
    """兼容模型返回字符串、Markdown JSON 代码块或字典。"""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
