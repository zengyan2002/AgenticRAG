"""入库流程的本地状态快照与 MongoDB 断点元数据。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
from typing import Any

from knowledge.core.settings import get_settings
from knowledge.utils.mongo_import_registry_util import _get_collection


CHECKPOINT_DIRECTORY = ".checkpoints"
CHECKPOINT_FILE_SUFFIX = ".state.json.gz"


def _json_value(value: Any) -> Any:
    """将图状态递归转换为 JSON 可序列化的数据。"""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def _restore_state_types(state: dict[str, Any]) -> dict[str, Any]:
    """恢复 JSON 会丢失的稀疏向量整数键。"""
    for chunk in state.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        sparse = chunk.get("sparse_vector")
        if isinstance(sparse, dict):
            chunk["sparse_vector"] = {
                int(key): float(value) for key, value in sparse.items()
            }
    batches = state.get("embedding_completed_batches")
    if isinstance(batches, list):
        state["embedding_completed_batches"] = [
            int(value) for value in batches
        ]
    return state


def save_import_checkpoint(
        state: dict[str, Any],
        *,
        checkpoint_node: str,
        resume_after_node: str | None = None,
        completed: bool = True,
) -> str:
    """原子保存状态文件，再更新 MongoDB 中的断点指针。

    大体积 Markdown、Chunk 和向量保存在 gzip 文件中，MongoDB 只保存
    路径、最后完成节点和任务进度，避免把完整图状态写进单条文档。
    """
    task_id = str(state.get("task_id") or "").strip()
    source_hash = str(state.get("source_hash") or "").strip()
    file_dir = str(state.get("file_dir") or "").strip()
    if not task_id or not source_hash or not file_dir:
        raise ValueError("保存入库断点缺少 task_id/source_hash/file_dir")

    task_dir = Path(file_dir).resolve()
    checkpoint_dir = task_dir / CHECKPOINT_DIRECTORY
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = Path(task_id).name
    if safe_task_id != task_id or safe_task_id in {"", ".", ".."}:
        raise ValueError("task_id 不能用于生成安全的断点文件名")
    checkpoint_path = checkpoint_dir / (
        f"{safe_task_id}{CHECKPOINT_FILE_SUFFIX}"
    )
    temporary_path = checkpoint_dir / f"{checkpoint_path.name}.tmp"

    snapshot = dict(state)
    snapshot["resume_after_node"] = (
        resume_after_node if resume_after_node is not None else checkpoint_node
    )
    payload = json.dumps(
        _json_value(snapshot), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    with gzip.open(temporary_path, "wb") as stream:
        stream.write(payload)
    os.replace(temporary_path, checkpoint_path)

    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(
        seconds=max(60, get_settings().import_claim_lease_seconds)
    )
    update: dict[str, Any] = {
        "$set": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_node": checkpoint_node,
            "resume_after_node": snapshot["resume_after_node"],
            "checkpoint_at": now,
            "file_dir": file_dir,
            "import_file_path": str(state.get("import_file_path") or ""),
            "version_id": str(state.get("version_id") or ""),
            "logical_document_id": str(
                state.get("logical_document_id") or ""
            ),
            "resource_objects": list(state.get("resource_objects") or []),
            "embedding_completed_batches": list(
                state.get("embedding_completed_batches") or []
            ),
            "updated_at": now,
            # 每完成一个节点就顺便续租，长文档处理时间超过初始租约时
            # 也不会被另一个上传请求误接管。
            "lease_until": lease_until,
        },
    }
    if completed:
        update["$addToSet"] = {"completed_nodes": checkpoint_node}
    result = _get_collection().update_one(
        {"source_hash": source_hash, "task_id": task_id},
        update,
    )
    if result.matched_count != 1:
        raise RuntimeError("任务已失去执行权，无法提交入库断点")
    return str(checkpoint_path)


def load_import_checkpoint(record: dict[str, Any]) -> dict[str, Any]:
    """从入库记录指向的压缩文件恢复并校验图状态。"""
    checkpoint_value = str(record.get("checkpoint_path") or "").strip()
    file_dir_value = str(record.get("file_dir") or "").strip()
    if not checkpoint_value or not file_dir_value:
        raise FileNotFoundError("该任务还没有可恢复的断点")

    checkpoint_path = Path(checkpoint_value).resolve()
    checkpoint_root = (
        Path(file_dir_value).resolve() / CHECKPOINT_DIRECTORY
    ).resolve()
    if checkpoint_path.parent != checkpoint_root:
        raise ValueError("断点文件不在任务专属目录中")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"断点文件不存在: {checkpoint_path}")

    with gzip.open(checkpoint_path, "rb") as stream:
        state = json.loads(stream.read().decode("utf-8"))
    if not isinstance(state, dict):
        raise ValueError("断点文件不是有效的图状态")
    return _restore_state_types(state)
