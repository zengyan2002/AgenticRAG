"""将现有切片无损复制到带Milvus原生BM25索引的新Collection。"""

from __future__ import annotations

import argparse
import os
import re

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.nodes.milvus_import_node import (
    _MilvusIndexBuilder,
    _MilvusSchemaBuilder,
)
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.document_identity_util import build_chunk_retrieval_text


def parse_args():
    parser = argparse.ArgumentParser(description="迁移切片并创建BM25索引")
    parser.add_argument(
        "--source",
        default=os.getenv("CHUNKS_COLLECTION", "kb_chunks_v1"),
    )
    parser.add_argument(
        "--target",
        default=os.getenv("BM25_COLLECTION", "kb_chunks_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collection_count(client, collection_name: str) -> int:
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count", 0))


def create_target(client, target: str, config: ImportConfig) -> None:
    schema = _MilvusSchemaBuilder.build_schema(
        client,
        config.embedding_dim,
        enable_bm25=True,
        analyzer_type=config.bm25_analyzer_type,
    )
    indexes = _MilvusIndexBuilder.build_index(client, enable_bm25=True)
    client.create_collection(
        collection_name=target,
        schema=schema,
        index_params=indexes,
    )


def ensure_analyzer_compatibility(server_version: str, analyzer_type: str) -> None:
    """Milvus内置中文/Jieba Analyzer从2.5.11起可用。"""
    if analyzer_type.lower() != "chinese":
        return
    numbers = [int(value) for value in re.findall(r"\d+", server_version)[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    if tuple(numbers) < (2, 5, 11):
        raise RuntimeError(
            "当前Milvus服务端为"
            f"{server_version}，中文BM25 Analyzer要求2.5.11或更高版本。"
            "为避免建立无效中文索引，本次没有创建目标Collection；"
            "请先升级Milvus后重新执行迁移。"
        )


def clean_row(row: dict) -> dict:
    result = dict(row)
    result.pop("bm25_sparse_vector", None)
    result["retrieval_text"] = build_chunk_retrieval_text(result)
    if result.get("id") is None:
        raise ValueError("源切片缺少主键id，无法保持跨Collection的chunk_id")
    return result


def main() -> int:
    args = parse_args()
    if args.source == args.target:
        raise ValueError("源Collection和目标Collection不能相同")

    config = ImportConfig.from_env()
    client = StorageClients.get_milvus()
    if not client.has_collection(args.source):
        raise ValueError(f"源Collection不存在: {args.source}")

    source_count = collection_count(client, args.source)
    server_version = client.get_server_version()
    print(f"Milvus服务端版本: {server_version}")
    print(f"源Collection: {args.source}, rows={source_count}")
    print(f"目标Collection: {args.target}")
    if args.dry_run:
        return 0

    ensure_analyzer_compatibility(
        server_version,
        config.bm25_analyzer_type,
    )

    if not client.has_collection(args.target):
        create_target(client, args.target, config)
    else:
        existing_count = collection_count(client, args.target)
        if existing_count not in (0, source_count):
            print(
                "检测到未完成的目标Collection，使用upsert从断点状态恢复: "
                f"{existing_count}/{source_count}"
            )

    iterator = client.query_iterator(
        collection_name=args.source,
        batch_size=args.batch_size,
        filter="",
        output_fields=["*"],
    )
    migrated = 0
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows = [clean_row(row) for row in batch]
            # 手动保留旧id，upsert使中断后的再次执行可恢复且不会重复。
            client.upsert(collection_name=args.target, data=rows)
            migrated += len(rows)
            print(f"已迁移 {migrated}/{source_count}", flush=True)
    finally:
        iterator.close()

    # Milvus可能仍有批次停留在写入缓冲区；不flush会读到过期row_count，
    # 从而把成功迁移误判成数据缺失。
    client.flush(args.target)
    target_count = collection_count(client, args.target)
    if target_count != source_count:
        raise RuntimeError(
            f"迁移数量校验失败: source={source_count}, target={target_count}"
        )
    print(f"迁移完成，目标行数={target_count}")
    print(
        "验证后设置 CHUNKS_COLLECTION、BM25_COLLECTION 为目标名称，"
        "并设置 BM25_ENABLED=true。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
