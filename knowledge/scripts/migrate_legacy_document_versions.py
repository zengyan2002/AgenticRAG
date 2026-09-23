"""把版本管理上线前的 Milvus 数据补齐为首个活动版本。"""

from __future__ import annotations

import argparse

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.document_identity_util import stable_logical_document_id
from knowledge.utils.document_version_util import switch_active_version


def parse_args():
    """解析迁移脚本参数。"""
    parser = argparse.ArgumentParser(
        description="为旧文档和Chunk补齐logical_document_id及活动版本字段"
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _legacy_version_id(row: dict) -> str:
    """优先复用旧版本 ID，缺失时从内容标识派生。"""
    existing = str(row.get("version_id") or "").strip()
    if existing:
        return existing
    seed = str(
        row.get("source_hash") or row.get("content_hash") or row.get("doc_id") or ""
    ).strip()
    if not seed:
        raise ValueError("旧文档缺少 version_id/source_hash/content_hash/doc_id")
    return f"ver_{seed[:24]}"


def main() -> int:
    """校验旧数据无歧义后，补齐版本字段和 MongoDB 活动指针。"""
    args = parse_args()
    config = ImportConfig.from_env()
    client = StorageClients.get_milvus()
    documents = client.query(
        collection_name=config.document_registry_collection,
        filter="",
        output_fields=["*"],
        limit=16384,
    ) or []

    planned: dict[str, dict] = {}
    by_doc_id: dict[str, dict] = {}
    for row in documents:
        logical_id, source = stable_logical_document_id(row)
        if logical_id in planned and planned[logical_id].get("doc_id") != row.get("doc_id"):
            raise RuntimeError(
                "发现多个旧版本具有同一规范标题，无法可靠判断哪个是最新版："
                f"{planned[logical_id].get('doc_id')}、{row.get('doc_id')}。"
                "请人工指定活动版本后再迁移。"
            )
        item = {
            **row,
            "logical_document_id": logical_id,
            "logical_id_source": source,
            "version_id": _legacy_version_id(row),
        }
        planned[logical_id] = item
        by_doc_id[str(row.get("doc_id") or "")] = item

    print(f"待迁移文档数: {len(planned)}")
    if args.dry_run:
        return 0

    for item in planned.values():
        client.upsert(
            collection_name=config.document_registry_collection,
            data=[{
                "doc_id": item["doc_id"],
                "logical_document_id": item["logical_document_id"],
                "logical_id_source": item["logical_id_source"],
                "version_id": item["version_id"],
                "version_status": "ready",
                "is_active": True,
            }],
            partial_update=True,
        )

    iterator = client.query_iterator(
        collection_name=config.chunks_collection,
        batch_size=args.batch_size,
        filter="",
        output_fields=["id", "doc_id"],
    )
    migrated_chunks = 0
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            updates = []
            for row in batch:
                document = by_doc_id.get(str(row.get("doc_id") or ""))
                if document is None:
                    raise RuntimeError(
                        f"Chunk {row.get('id')} 的 doc_id 不在文档注册表中"
                    )
                updates.append({
                    "id": row["id"],
                    "logical_document_id": document["logical_document_id"],
                    "version_id": document["version_id"],
                    "version_status": "ready",
                    "is_active": True,
                })
            if updates:
                client.upsert(
                    collection_name=config.chunks_collection,
                    data=updates,
                    partial_update=True,
                )
                migrated_chunks += len(updates)
    finally:
        iterator.close()

    for item in planned.values():
        switch_active_version(
            logical_document_id=item["logical_document_id"],
            version_id=item["version_id"],
            doc_id=item["doc_id"],
            source_hash=str(item.get("source_hash") or ""),
            content_hash=str(item.get("content_hash") or ""),
            task_id="legacy-migration",
        )

    print(f"迁移完成: documents={len(planned)}, chunks={migrated_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
