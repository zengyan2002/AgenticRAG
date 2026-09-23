"""使用 MongoDB 记录文档入库状态并提供文件级幂等控制。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict

from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from knowledge.core.settings import get_settings
from knowledge.utils.clients.storage_clients import StorageClients


IMPORT_STATUS_PROCESSING = "processing"
IMPORT_STATUS_COMPLETED = "completed"
IMPORT_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class ImportClaim:
    """一次文件入库占位的结果。"""

    acquired: bool
    record: Dict[str, Any]


def _get_collection() -> Collection:
    """获取入库幂等记录集合。

    Returns:
        MongoDB 入库记录 Collection。
    """
    settings = get_settings()
    return StorageClients.get_mongo_db()[settings.import_registry_collection]


@lru_cache(maxsize=1)
def ensure_import_registry_indexes() -> None:
    """创建文件哈希唯一索引以及状态查询索引。"""
    collection = _get_collection()
    collection.create_index("source_hash", unique=True, name="source_hash_unique")
    collection.create_index(
        [("status", 1), ("lease_until", 1)],
        name="status_lease_until",
    )
    collection.create_index(
        [("logical_document_id", 1), ("version_id", 1)],
        name="logical_document_version",
    )


def claim_import(
        source_hash: str,
        task_id: str,
        file_name: str,
        logical_document_id: str = "",
) -> ImportClaim:
    """原子占用文件哈希；失败任务或过期任务允许被新任务接管。

    MongoDB 的 ``source_hash`` 唯一索引负责处理并发竞争。首次上传会
    插入 ``processing`` 记录；若记录已存在，仅当旧任务失败或租约过期
    时才通过条件更新接管，否则返回现有记录供接口直接复用。

    Args:
        source_hash: 原始上传文件的 SHA-256。
        task_id: 当前上传请求生成的任务 ID。
        file_name: 清理路径信息后的原始文件名。
        logical_document_id: 调用方显式指定的逻辑文档标识。

    Returns:
        是否获得执行权以及对应的持久化记录。
    """
    if not source_hash:
        raise ValueError("source_hash 不能为空")

    ensure_import_registry_indexes()
    collection = _get_collection()
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(
        seconds=max(60, get_settings().import_claim_lease_seconds)
    )
    record = {
        "source_hash": source_hash,
        "task_id": task_id,
        "file_name": file_name,
        "status": IMPORT_STATUS_PROCESSING,
        "lease_until": lease_until,
        "created_at": now,
        "updated_at": now,
        "attempt": 1,
        "requested_logical_document_id": logical_document_id,
        "version_id": f"ver_{source_hash[:24]}",
        "completed_nodes": [],
    }
    try:
        collection.insert_one(record)
        return ImportClaim(acquired=True, record=record)
    except DuplicateKeyError:
        pass

    # 失败任务可以重试；进程异常退出后，超过租约的 processing 任务也
    # 可以被接管。条件更新保证并发请求中仍只有一个请求获得执行权。
    set_fields = {
        "task_id": task_id,
        "file_name": file_name,
        "status": IMPORT_STATUS_PROCESSING,
        "lease_until": lease_until,
        "updated_at": now,
        "error": "",
    }
    if logical_document_id:
        set_fields["requested_logical_document_id"] = logical_document_id
    claimed = collection.find_one_and_update(
        {
            "source_hash": source_hash,
            "$or": [
                {"status": IMPORT_STATUS_FAILED},
                {
                    "status": IMPORT_STATUS_PROCESSING,
                    "lease_until": {"$lte": now},
                },
            ],
        },
        {
            "$set": set_fields,
            "$inc": {"attempt": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if claimed:
        return ImportClaim(acquired=True, record=claimed)

    existing = collection.find_one({"source_hash": source_hash}) or {}
    return ImportClaim(acquired=False, record=existing)


def get_import_record_by_task_id(task_id: str) -> Dict[str, Any]:
    """读取一个任务的持久化入库记录。"""
    if not task_id:
        return {}
    return _get_collection().find_one({"task_id": task_id}) or {}


def reclaim_import_for_retry(task_id: str) -> Dict[str, Any]:
    """原子领取失败或租约过期的任务，用于从断点恢复。"""
    if not task_id:
        raise ValueError("task_id 不能为空")
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(
        seconds=max(60, get_settings().import_claim_lease_seconds)
    )
    record = _get_collection().find_one_and_update(
        {
            "task_id": task_id,
            "$or": [
                {"status": IMPORT_STATUS_FAILED},
                {
                    "status": IMPORT_STATUS_PROCESSING,
                    "lease_until": {"$lte": now},
                },
            ],
        },
        {
            "$set": {
                "status": IMPORT_STATUS_PROCESSING,
                "lease_until": lease_until,
                "updated_at": now,
                "error": "",
            },
            "$inc": {"attempt": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if record:
        return record
    existing = get_import_record_by_task_id(task_id)
    if not existing:
        raise KeyError("入库任务不存在")
    raise RuntimeError(
        f"任务当前状态为 {existing.get('status') or 'unknown'}，不能重试"
    )


def mark_import_completed(
        source_hash: str,
        task_id: str,
        *,
        doc_id: str,
        content_hash: str,
        version_id: str,
        logical_document_id: str,
) -> None:
    """把当前任务持有的入库记录标记为成功。

    Args:
        source_hash: 原始上传文件的 SHA-256。
        task_id: 当前持有占位的任务 ID。
        doc_id: 解析后生成的稳定文档 ID。
        content_hash: 规范化切片内容的哈希。
        version_id: 当前内容版本 ID。
        logical_document_id: 跨版本稳定的逻辑文档 ID。
    """
    now = datetime.now(timezone.utc)
    result = _get_collection().update_one(
        {
            "source_hash": source_hash,
            "task_id": task_id,
            "status": IMPORT_STATUS_PROCESSING,
        },
        {
            "$set": {
                "status": IMPORT_STATUS_COMPLETED,
                "doc_id": doc_id,
                "content_hash": content_hash,
                "version_id": version_id,
                "logical_document_id": logical_document_id,
                "completed_at": now,
                "updated_at": now,
            },
            "$unset": {
                "lease_until": "",
                "error": "",
                # 成功后本地任务目录可能被删除，不能继续保留一个失效的
                # 断点文件指针。completed_nodes 仍保留用于状态展示。
                "checkpoint_path": "",
                "resume_after_node": "",
                "embedding_completed_batches": "",
            },
        },
    )
    if result.matched_count != 1:
        raise RuntimeError("入库任务已失去幂等占位，无法标记为成功")


def mark_import_failed(source_hash: str, task_id: str, error: str) -> None:
    """把当前任务持有的入库记录标记为失败，以允许后续重试。

    Args:
        source_hash: 原始上传文件的 SHA-256。
        task_id: 当前持有占位的任务 ID。
        error: 入库流程的失败原因。
    """
    now = datetime.now(timezone.utc)
    _get_collection().update_one(
        {
            "source_hash": source_hash,
            "task_id": task_id,
            "status": IMPORT_STATUS_PROCESSING,
        },
        {
            "$set": {
                "status": IMPORT_STATUS_FAILED,
                "error": str(error)[:2000],
                "updated_at": now,
            },
            "$unset": {"lease_until": ""},
        },
    )
