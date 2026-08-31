import logging
from datetime import datetime
from typing import Any, Dict, List

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.collection import Collection

from knowledge.utils.clients.storage_clients import StorageClients

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TITLE = "新会话"


def _get_collection() -> Collection:
    """获取 chat_session 集合。"""
    return StorageClients.get_mongo_db()["chat_session"]


def ensure_session_indexes() -> None:
    """创建会话列表查询需要的索引。"""
    collection = _get_collection()
    collection.create_index(
        [("session_id", ASCENDING)],
        unique=True,
    )
    collection.create_index([
        ("owner_id", ASCENDING),
        ("updated_at", DESCENDING),
    ])


def _serialize_session(document: Dict[str, Any]) -> Dict[str, Any]:
    if not document:
        return {}

    return {
        "session_id": document.get("session_id", ""),
        "title": document.get("title", DEFAULT_SESSION_TITLE),
        "owner_id": document.get("owner_id", ""),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
        "last_message": document.get("last_message", ""),
    }


def create_chat_session(
        session_id: str,
        owner_id: str,
        title: str = DEFAULT_SESSION_TITLE,
) -> Dict[str, Any]:
    """登记一个由前端生成的会话；重复登记时直接返回已有会话。"""
    now = datetime.now().timestamp()
    clean_title = title.strip() or DEFAULT_SESSION_TITLE

    document = {
        "session_id": session_id,
        "title": clean_title,
        "owner_id": owner_id,
        "created_at": now,
        "updated_at": now,
        "last_message": "",
    }

    result = _get_collection().find_one_and_update(
        {"session_id": session_id},
        {"$setOnInsert": document},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if result.get("owner_id") != owner_id:
        raise ValueError("该 session_id 已属于其他客户端")

    return _serialize_session(result)


def list_chat_sessions(owner_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """按最近使用时间返回某个客户端的会话列表。"""
    cursor = (
        _get_collection()
        .find({"owner_id": owner_id})
        .sort("updated_at", DESCENDING)
        .limit(limit)
    )
    return [_serialize_session(item) for item in cursor]


def touch_chat_session(session_id: str, message: str) -> Dict[str, Any]:
    """发送消息时更新会话标题、摘要和最近使用时间。"""
    collection = _get_collection()
    session = collection.find_one({"session_id": session_id})
    if not session:
        return {}

    message = " ".join(str(message or "").split())
    update_fields = {
        "updated_at": datetime.now().timestamp(),
        "last_message": message,
    }

    if session.get("title") in (None, "", DEFAULT_SESSION_TITLE) and message:
        update_fields["title"] = message if len(message) <= 20 else f"{message[:20]}…"

    result = collection.find_one_and_update(
        {"session_id": session_id},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize_session(result)


def rename_chat_session(session_id: str, owner_id: str, title: str) -> Dict[str, Any]:
    """修改会话标题。"""
    clean_title = str(title or "").strip()
    if not clean_title:
        raise ValueError("会话标题不能为空")

    result = _get_collection().find_one_and_update(
        {"session_id": session_id, "owner_id": owner_id},
        {
            "$set": {
                "title": clean_title,
                "updated_at": datetime.now().timestamp(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    return _serialize_session(result)


def clear_chat_session_preview(session_id: str) -> None:
    """清空消息后同步清除左侧会话摘要。"""
    _get_collection().update_one(
        {"session_id": session_id},
        {
            "$set": {
                "last_message": "",
                "updated_at": datetime.now().timestamp(),
            }
        },
    )


def delete_chat_session(session_id: str, owner_id: str) -> int:
    """删除会话目录记录，消息记录由 service 层一并删除。"""
    result = _get_collection().delete_one({
        "session_id": session_id,
        "owner_id": owner_id,
    })
    return result.deleted_count
