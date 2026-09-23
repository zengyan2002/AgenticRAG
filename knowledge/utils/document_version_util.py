"""协调 MongoDB 活动版本指针与 Milvus 版本可见性。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import logging
from typing import Any, Iterable

from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from knowledge.core.settings import get_settings
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.utils.clients.storage_clients import StorageClients


logger = logging.getLogger(__name__)

VERSION_STATUS_BUILDING = "building"
VERSION_STATUS_READY = "ready"
VERSION_STATUS_RETIRED = "retired"
VERSION_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class VersionActivation:
    """一次版本切换的结果。"""

    logical_document_id: str
    version_id: str
    previous_version_id: str = ""
    previous_doc_id: str = ""
    retired_chunk_ids: tuple[int, ...] = ()
    retired_resource_objects: tuple[str, ...] = ()


def _get_version_collection() -> Collection:
    """获取保存逻辑文档活动版本指针的 MongoDB Collection。"""
    settings = get_settings()
    return StorageClients.get_mongo_db()[
        settings.document_version_registry_collection
    ]


@lru_cache(maxsize=1)
def ensure_document_version_indexes() -> None:
    """保证一个逻辑文档最多只有一条活动版本指针记录。"""
    collection = _get_version_collection()
    collection.create_index(
        "logical_document_id",
        unique=True,
        name="logical_document_id_unique",
    )
    collection.create_index("active_version_id", name="active_version_id")
    collection.create_index("source_hash", name="source_hash")


def switch_active_version(
        *,
        logical_document_id: str,
        version_id: str,
        doc_id: str,
        source_hash: str,
        content_hash: str = "",
        task_id: str,
        resource_objects: Iterable[str] = (),
) -> dict[str, Any]:
    """原子替换逻辑文档的活动版本，并返回切换前的记录。

    MongoDB 单文档更新具备原子性。首次创建指针遇到并发插入竞争时，
    会退化为一次普通更新，从而仍以最后完成切换的版本为准。
    """
    ensure_document_version_indexes()
    collection = _get_version_collection()
    now = datetime.now(timezone.utc)
    update = {
        "$set": {
            "logical_document_id": logical_document_id,
            "active_version_id": version_id,
            "active_doc_id": doc_id,
            "source_hash": source_hash,
            "content_hash": content_hash,
            "task_id": task_id,
            "status": VERSION_STATUS_READY,
            "resource_objects": list(dict.fromkeys(
                str(item) for item in resource_objects if str(item).strip()
            )),
            "updated_at": now,
        },
        "$setOnInsert": {"created_at": now},
    }
    try:
        previous = collection.find_one_and_update(
            {"logical_document_id": logical_document_id},
            update,
            upsert=True,
            return_document=ReturnDocument.BEFORE,
        )
    except DuplicateKeyError:
        previous = collection.find_one_and_update(
            {"logical_document_id": logical_document_id},
            update,
            return_document=ReturnDocument.BEFORE,
        )
    return previous or {}


def find_active_version_by_source_hash(source_hash: str) -> dict[str, Any]:
    """按原文件哈希查找已经成功发布的活动版本。"""
    if not source_hash:
        return {}
    ensure_document_version_indexes()
    return _get_version_collection().find_one({
        "source_hash": source_hash,
        "status": VERSION_STATUS_READY,
    }) or {}


def _chunk_ids(chunks: Iterable[dict[str, Any]]) -> list[int]:
    """提取并去重 Milvus Chunk 主键。"""
    result: list[int] = []
    seen: set[int] = set()
    for chunk in chunks or []:
        value = chunk.get("id", chunk.get("chunk_id"))
        if isinstance(value, bool):
            continue
        try:
            chunk_id = int(value)
        except (TypeError, ValueError):
            continue
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append(chunk_id)
    return result


def _set_chunk_status(
        client,
        collection_name: str,
        chunk_ids: Iterable[int],
        *,
        status: str,
        is_active: bool,
) -> None:
    """利用主键部分更新一批 Chunk 的版本状态。"""
    rows = [
        {"id": chunk_id, "version_status": status, "is_active": is_active}
        for chunk_id in chunk_ids
    ]
    if rows:
        client.upsert(
            collection_name=collection_name,
            data=rows,
            partial_update=True,
        )


def _set_profile_status(
        client,
        collection_name: str,
        doc_id: str,
        *,
        status: str,
        is_active: bool,
) -> None:
    """部分更新文档注册表中一个版本的可见状态。"""
    if not doc_id:
        return
    client.upsert(
        collection_name=collection_name,
        data=[{
            "doc_id": doc_id,
            "version_status": status,
            "is_active": is_active,
        }],
        partial_update=True,
    )


def _find_version_chunk_ids(
        client,
        collection_name: str,
        logical_document_id: str,
        version_id: str,
) -> list[int]:
    """查询指定逻辑文档版本对应的全部 Chunk 主键。"""
    if not logical_document_id or not version_id:
        return []
    rows = client.query(
        collection_name=collection_name,
        filter=(
            f'logical_document_id == "{logical_document_id}" and '
            f'version_id == "{version_id}"'
        ),
        output_fields=["id"],
        limit=16384,
    )
    return _chunk_ids(rows or [])


def activate_document_version(
        *,
        identity: dict[str, Any],
        chunks: list[dict[str, Any]],
        resource_objects: Iterable[str] = (),
        task_id: str,
) -> VersionActivation:
    """发布完整构建的新版本，并安全下线此前的活动版本。

    新版本在入库阶段始终为 ``building/is_active=false``。这里先使新版
    可检索，再原子更新 MongoDB 活动指针，最后停用旧版。这个顺序允许
    极短暂的新旧重叠，但不会产生两个版本都不可用的时间窗口。

    若活动指针切换失败，会把新版本恢复为不可见，旧版不受影响。
    """
    logical_id = str(identity.get("logical_document_id") or "").strip()
    version_id = str(identity.get("version_id") or "").strip()
    doc_id = str(identity.get("doc_id") or "").strip()
    source_hash = str(identity.get("source_hash") or "").strip()
    if not logical_id or not version_id or not doc_id:
        raise ValueError("版本发布缺少 logical_document_id/version_id/doc_id")

    config = ImportConfig.from_env()
    client = StorageClients.get_milvus()
    new_chunk_ids = _chunk_ids(chunks)
    if not new_chunk_ids:
        raise ValueError("版本发布时没有可用的 Chunk 主键")

    try:
        _set_chunk_status(
            client,
            config.chunks_collection,
            new_chunk_ids,
            status=VERSION_STATUS_READY,
            is_active=True,
        )
        _set_profile_status(
            client,
            config.document_registry_collection,
            doc_id,
            status=VERSION_STATUS_READY,
            is_active=True,
        )
        previous = switch_active_version(
            logical_document_id=logical_id,
            version_id=version_id,
            doc_id=doc_id,
            source_hash=source_hash,
            content_hash=str(identity.get("content_hash") or ""),
            task_id=task_id,
            resource_objects=resource_objects,
        )
    except Exception:
        # Mongo 指针未切换成功时，新版本不得继续暴露。
        try:
            _set_chunk_status(
                client,
                config.chunks_collection,
                new_chunk_ids,
                status=VERSION_STATUS_FAILED,
                is_active=False,
            )
            _set_profile_status(
                client,
                config.document_registry_collection,
                doc_id,
                status=VERSION_STATUS_FAILED,
                is_active=False,
            )
        except Exception as rollback_exc:
            logger.error("回滚未发布版本失败: %s", rollback_exc)
        raise

    previous_version_id = str(previous.get("active_version_id") or "")
    previous_doc_id = str(previous.get("active_doc_id") or "")
    previous_resources = tuple(
        str(item)
        for item in (previous.get("resource_objects") or [])
        if str(item).strip()
    )
    old_ids: list[int] = []
    retirement_resolved = False
    if previous_version_id and previous_version_id != version_id:
        try:
            old_ids = _find_version_chunk_ids(
                client,
                config.chunks_collection,
                logical_id,
                previous_version_id,
            )
            _set_chunk_status(
                client,
                config.chunks_collection,
                old_ids,
                status=VERSION_STATUS_RETIRED,
                is_active=False,
            )
            _set_profile_status(
                client,
                config.document_registry_collection,
                previous_doc_id,
                status=VERSION_STATUS_RETIRED,
                is_active=False,
            )
            retirement_resolved = True
        except Exception as retire_exc:
            # 活动指针已成功切到新版，旧版下线失败只能降级为待清理，
            # 不能反过来把一个已经发布成功的版本判为失败。
            logger.warning(
                "新版已发布，但旧版本暂未完成下线: logical_id=%s, version=%s, error=%s",
                logical_id,
                previous_version_id,
                retire_exc,
            )

    return VersionActivation(
        logical_document_id=logical_id,
        version_id=version_id,
        previous_version_id=previous_version_id,
        previous_doc_id=previous_doc_id,
        retired_chunk_ids=tuple(old_ids),
        retired_resource_objects=(
            previous_resources if retirement_resolved else ()
        ),
    )


def cleanup_retired_version(activation: VersionActivation) -> None:
    """尽力物理删除已停用 Chunk；失败不会影响当前活动版本。"""
    if not activation.retired_chunk_ids and not activation.retired_resource_objects:
        return
    config = ImportConfig.from_env()
    if activation.retired_chunk_ids:
        try:
            StorageClients.get_milvus().delete(
                collection_name=config.chunks_collection,
                ids=list(activation.retired_chunk_ids),
            )
        except Exception as exc:
            logger.warning(
                "旧版本 Chunk 物理清理失败，可稍后重试: logical_id=%s, version=%s, error=%s",
                activation.logical_document_id,
                activation.previous_version_id,
                exc,
            )
    if activation.retired_resource_objects:
        try:
            minio = StorageClients.get_minio()
            bucket = get_settings().require("minio_bucket_name")[0]
            for object_name in activation.retired_resource_objects:
                minio.remove_object(bucket, object_name)
        except Exception as exc:
            logger.warning(
                "旧版本对象存储清理失败，可稍后重试: logical_id=%s, version=%s, error=%s",
                activation.logical_document_id,
                activation.previous_version_id,
                exc,
            )


def discard_unpublished_version(
        identity: dict[str, Any],
        resource_objects: Iterable[str] = (),
) -> None:
    """尽力删除构建失败且从未发布的 Milvus 数据。"""
    version_id = str(identity.get("version_id") or "").strip()
    doc_id = str(identity.get("doc_id") or "").strip()
    if not version_id:
        return
    config = ImportConfig.from_env()
    try:
        client = StorageClients.get_milvus()
        client.delete(
            collection_name=config.chunks_collection,
            filter=f'version_id == "{version_id}" and is_active == false',
        )
        if doc_id:
            client.delete(
                collection_name=config.document_registry_collection,
                ids=[doc_id],
            )
    except Exception as exc:
        logger.warning("清理未发布版本的 Milvus 数据失败，可稍后重试: %s", exc)
    objects = [str(item) for item in resource_objects if str(item).strip()]
    if objects:
        try:
            minio = StorageClients.get_minio()
            bucket = get_settings().require("minio_bucket_name")[0]
            for object_name in objects:
                minio.remove_object(bucket, object_name)
        except Exception as exc:
            logger.warning("清理未发布版本的对象存储资源失败，可稍后重试: %s", exc)
