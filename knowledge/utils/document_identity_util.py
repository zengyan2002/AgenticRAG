"""文档身份的标准化、型号提取与序列化工具。"""

import hashlib
import json
import re
from typing import Any, Iterable


def normalize_document_name(value: Any) -> str:
    """用于别名精确对齐的保守标准化。"""
    if not isinstance(value, str):
        return ""
    normalized = value.strip().casefold()
    return re.sub(r"[\s\-_/／·•—–_.,，。:：;；()（）\[\]【】]+", "", normalized)


def extract_model_codes(value: Any) -> list[str]:
    """提取同时包含字母和数字的型号，例如 RS-12、B5-440。"""
    if not isinstance(value, str):
        return []
    raw_codes = re.findall(
        r"(?i)(?<![a-z0-9])"
        r"(?=[a-z0-9_-]*[a-z])"
        r"(?=[a-z0-9_-]*\d)"
        r"[a-z0-9]+(?:[-_][a-z0-9]+)*"
        r"(?![a-z0-9])",
        value,
    )
    result = []
    seen = set()
    for code in raw_codes:
        normalized = re.sub(r"[-_]", "", code).casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def unique_strings(values: Iterable[Any], max_items: int | None = None) -> list[str]:
    """按标准化后的值去重，同时保留首次出现的原文。"""
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = re.sub(r"\s+", " ", value).strip()
        key = normalize_document_name(cleaned)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if max_items is not None and len(result) >= max_items:
            break
    return result


def parse_string_list(value: Any) -> list[str]:
    """兼容 Milvus 中 JSON 字符串与内存列表两种形式。"""
    if isinstance(value, list):
        return unique_strings(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part for part in re.split(r"[,，;；|]", value) if part.strip()]
    return unique_strings(parsed if isinstance(parsed, list) else [])


def stable_document_hash(chunks: list[dict[str, Any]]) -> str:
    """根据切片正文生成稳定内容哈希。"""
    digest = hashlib.sha256()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        digest.update(str(chunk.get("content") or "").encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def build_document_profile_text(identity: dict[str, Any]) -> str:
    """构建文档注册表的稠密/稀疏向量文本。"""
    aliases = identity.get("aliases") or []
    model_codes = identity.get("model_codes") or []
    parts = [
        f"文档名：{identity.get('canonical_title') or ''}",
        f"主要对象：{identity.get('primary_subject') or ''}",
        f"别名：{'、'.join(aliases)}",
        f"型号：{'、'.join(model_codes)}",
        f"类型：{identity.get('document_type') or ''}",
        f"摘要：{identity.get('summary') or ''}",
    ]
    return "\n".join(part for part in parts if not part.endswith("："))


def build_chunk_retrieval_text(chunk: dict[str, Any]) -> str:
    """构建Dense/Sparse/BM25共用的可检索文本。"""
    fields = (
        ("文档", chunk.get("canonical_title") or chunk.get("file_title")),
        ("研究对象", chunk.get("primary_subject") or chunk.get("theme_name")),
        ("章节路径", chunk.get("section_path")),
        ("父标题", chunk.get("parent_title")),
        ("切片标题", chunk.get("title")),
        ("正文", chunk.get("content")),
        ("公式检索信息", chunk.get("formula_search_text")),
    )
    parts = []
    for label, value in fields:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text:
            if label == "公式检索信息":
                parts.append(f"【公式检索信息】\n{text}")
            else:
                parts.append(f"{label}：{text}")
    return "\n".join(parts)
