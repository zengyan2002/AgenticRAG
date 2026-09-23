# 旧知识库版本字段修复与复测

2026-09-06 排查发现：查询默认启用 `ACTIVE_VERSION_FILTER_ENABLED=true`，但已有 Milvus 文档和切片没有活动版本字段，导致所有旧资料被过滤掉。

本次保留活动版本过滤，迁移旧数据，并为同名论文选择正文较完整的物理记录作为迁移基线。没有按不可靠的时间信息推断最新版。其余副本仅设为非活动，未删除。

## 实际迁移结果

| 集合 | 总记录 | 活动记录 | 非活动记录 |
| --- | ---: | ---: | ---: |
| kb_document_registry_v1 | 34 | 31 | 3 |
| kb_chunks_bm25_v1 | 2835 | 2552 | 283 |

MongoDB 已补齐 31 个逻辑文档的活动版本指针。原文、向量和主键均在迁移后逐条与完整快照核对。

## 本地审计材料

- `temp_data/legacy_migration_20260906/before.json`：完整迁移前快照，含原始向量和动态字段。
- `inspection.json`、`selections.json`、`selection_rationale.md`：同名文档比较和选择依据。
- `migration_plan.json`、`migration_result.json`：计划及迁移后校验结果。
- `evaluation/results/quality_20260906/`：修复前评测，保留不覆盖。
- `evaluation/results/quality_20260906_repaired/`：修复后独立评测。

## 执行方式

原脚本对重复标题会拒绝迁移；当前 Milvus 还不支持仅提供部分字段的 upsert。本次使用 `scripts/migrate_legacy_snapshot.py`，先检查快照一致性和版本冲突，再以完整记录更新兼容旧服务。它不删除任何记录，也不重算原始稠密或稀疏向量；BM25 由数据库维护。

从 knowledge 目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/migrate_legacy_snapshot.py --snapshot temp_data/legacy_migration_20260906/before.json --selection-file temp_data/legacy_migration_20260906/selections.json --dry-run
.\.venv\Scripts\python.exe scripts/migrate_legacy_snapshot.py --snapshot temp_data/legacy_migration_20260906/before.json --selection-file temp_data/legacy_migration_20260906/selections.json --full-row-upsert
.\.venv\Scripts\python.exe -u evaluation/run_quality_eval.py --output evaluation/results/quality_20260906_repaired/results.json --resume
.\.venv\Scripts\python.exe -u evaluation/run_quality_eval.py --output evaluation/results/quality_20260906_repaired/results.json --finalize
```

历史快照只适用于对应数据状态；以后数据发生变化时需重新检查并建立新快照，不能强制覆盖一致性检查。复测沿用原始 67 题、原有模型和查询参数，指标定义与修复前一致；自动评审仍有偏差，应结合逐题证据复核。
