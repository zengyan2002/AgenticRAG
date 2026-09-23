"""Migrate a verified legacy snapshot; preserve inactive duplicates and all content."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import get_config
from knowledge.core.settings import get_settings
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.document_identity_util import stable_logical_document_id
from knowledge.utils.document_version_util import switch_active_version

VERSION_FIELDS = {'logical_document_id', 'logical_id_source', 'version_id', 'version_status', 'is_active'}


def build_plan(documents, chunks, selections):
    groups, by_doc = defaultdict(list), {}
    for row in documents:
        logical_id, source = stable_logical_document_id(row)
        doc_id = row.get('doc_id')
        if not doc_id or doc_id in by_doc:
            raise ValueError('文档主键缺失或重复')
        seed = str(row.get('source_hash') or row.get('content_hash') or doc_id)
        item = {'doc_id': doc_id, 'logical_document_id': logical_id,
                'logical_id_source': source, 'version_id': row.get('version_id') or f'ver_{seed[:24]}'}
        groups[logical_id].append(item)
        by_doc[doc_id] = item
    if set(selections) - set(groups):
        raise ValueError('活动版本选择包含未知逻辑文档')
    for logical_id, items in groups.items():
        selected = selections.get(logical_id)
        if len(items) > 1 and not selected:
            raise ValueError(f'同名文档需要显式选择活动记录: {logical_id}')
        selected = selected or items[0]['doc_id']
        if selected not in {x['doc_id'] for x in items}:
            raise ValueError('活动记录不属于对应逻辑文档')
        for item in items:
            item['is_active'] = item['doc_id'] == selected
            item['version_status'] = 'ready' if item['is_active'] else 'retired'
    updates = []
    for row in chunks:
        item = by_doc.get(row.get('doc_id'))
        if item is None:
            raise ValueError(f"Chunk {row.get('id')} 的 doc_id 不在文档注册表中")
        updates.append({'id': row['id'], **{k: item[k] for k in VERSION_FIELDS - {'logical_id_source'}}})
    return list(by_doc.values()), updates


def read_rows(client, name):
    rows = []
    iterator = client.query_iterator(collection_name=name, filter='', output_fields=['*'], batch_size=200, consistency_level='Strong')
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    return json.loads(json.dumps(rows, default=str))


def verify_unchanged(before, current, primary_key):
    originals = {r[primary_key]: r for r in before}
    actual = {r[primary_key]: r for r in current}
    if originals.keys() != actual.keys() or len(before) != len(current):
        raise RuntimeError('数据库主键集合与备份不一致')
    for key, row in originals.items():
        for field, value in row.items():
            if field not in VERSION_FIELDS and actual[key].get(field) != value:
                raise RuntimeError(f'正文或向量与备份不一致: {key}/{field}')


def full_rows(originals, updates, primary_key):
    original_map = {r[primary_key]: r for r in originals}
    result = []
    for update in updates:
        row = {**original_map[update[primary_key]], **update}
        # JSON snapshots stringify sparse vector indices; Milvus expects integers.
        if isinstance(row.get('sparse_vector'), dict):
            row['sparse_vector'] = {int(k):v for k,v in row['sparse_vector'].items()}
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--full-row-upsert', action='store_true', help='Compatibility for servers without partial update support')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
    selections = json.loads(args.selection_file.read_text(encoding='utf-8')) if args.selection_file else {}
    config, qc = ImportConfig.from_env(), get_config()
    names = {config.chunks_collection, qc.chunks_collection, qc.bm25_collection}
    registry_name = config.document_registry_collection
    documents = snapshot['collections'][registry_name]['rows']
    plans = {}
    for name in names:
        profiles, plans[name] = build_plan(documents, snapshot['collections'][name]['rows'], selections)
    client = StorageClients.get_milvus()
    for name in names | {registry_name}:
        current = read_rows(client, name)
        verify_unchanged(snapshot['collections'][name]['rows'], current, 'doc_id' if name == registry_name else 'id')
        expected = {x['doc_id' if name == registry_name else 'id']: x for x in (profiles if name == registry_name else plans[name])}
        for row in current:
            target = expected[row['doc_id' if name == registry_name else 'id']]
            for field in VERSION_FIELDS:
                if row.get(field) is not None and field in target and row[field] != target[field]:
                    raise RuntimeError(f'已有版本状态与计划冲突: {name}/{field}')
    pointers = StorageClients.get_mongo_db()[get_settings().document_version_registry_collection]
    expected_active = {p['logical_document_id']: p for p in profiles if p['is_active']}
    for pointer in pointers.find():
        expected = expected_active.get(pointer['logical_document_id'])
        if not expected or pointer.get('active_doc_id') != expected['doc_id'] or pointer.get('active_version_id') != expected['version_id']:
            raise RuntimeError('已有 MongoDB 活动指针与迁移计划冲突')
    manifest = {'profiles':profiles, 'chunk_counts':{name:{'total':len(rows), 'active':sum(r['is_active'] for r in rows)} for name,rows in plans.items()},
                'active_documents':len(expected_active), 'retained_inactive_documents':len(profiles)-len(expected_active)}
    (args.snapshot.parent/'migration_plan.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='profiles'}, ensure_ascii=False),flush=True)
    if args.dry_run:
        return 0
    for name, rows in plans.items():
        if args.full_row_upsert:
            rows = full_rows(snapshot['collections'][name]['rows'], rows, 'id')
        for offset in range(0,len(rows),args.batch_size):
            client.upsert(collection_name=name, data=rows[offset:offset+args.batch_size], partial_update=not args.full_row_upsert, timeout=60)
        print('updated',name,len(rows),flush=True)
    registry_rows = full_rows(documents, profiles, 'doc_id') if args.full_row_upsert else profiles
    client.upsert(collection_name=registry_name, data=registry_rows, partial_update=not args.full_row_upsert, timeout=60)
    originals = {d['doc_id']:d for d in documents}
    for profile in expected_active.values():
        original = originals[profile['doc_id']]
        switch_active_version(logical_document_id=profile['logical_document_id'], version_id=profile['version_id'],
            doc_id=profile['doc_id'], source_hash=original.get('source_hash') or '',
            content_hash=original.get('content_hash') or '', task_id='legacy-migration')
    for name in names | {registry_name}:
        current = read_rows(client,name)
        verify_unchanged(snapshot['collections'][name]['rows'],current,'doc_id' if name==registry_name else 'id')
        expected = {r['doc_id' if name==registry_name else 'id']:r for r in (profiles if name==registry_name else plans[name])}
        for row in current:
            for field,value in expected[row['doc_id' if name==registry_name else 'id']].items():
                if row.get(field)!=value:
                    raise RuntimeError(f'迁移字段校验失败: {name}/{field}')
    manifest['verified_original_content_and_vectors_unchanged'] = True
    (args.snapshot.parent/'migration_result.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Migration verified; all original content and vectors unchanged.',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
