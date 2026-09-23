"""Read-only snapshot and content comparison for legacy version migration."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import get_config
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.document_identity_util import stable_logical_document_id
from knowledge.core.settings import get_settings

OUT = ROOT / 'temp_data/legacy_migration_20260906'
OUT.mkdir(parents=True, exist_ok=True)
client = StorageClients.get_milvus()
config = ImportConfig.from_env()
query_config = get_config()
collections = sorted({config.chunks_collection, query_config.chunks_collection, query_config.bm25_collection, config.document_registry_collection})
snapshot = {'collections': {}, 'mongo_pointers': list(StorageClients.get_mongo_db()[get_settings().document_version_registry_collection].find())}
for name in collections:
    rows = []
    iterator = client.query_iterator(collection_name=name, filter='', output_fields=['*'], batch_size=200, consistency_level='Strong')
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    snapshot['collections'][name] = {'schema': client.describe_collection(name), 'rows': rows}
    print(name, len(rows), flush=True)
path = OUT / 'before.json'
if path.exists():
    raise RuntimeError('Refusing to overwrite original snapshot')
path.write_text(json.dumps(snapshot, ensure_ascii=False, default=str), encoding='utf-8')
groups = defaultdict(list)
documents = snapshot['collections'][config.document_registry_collection]['rows']
chunks = snapshot['collections'][config.chunks_collection]['rows']
by_doc = defaultdict(list)
for chunk in chunks:
    by_doc[chunk.get('doc_id')].append(chunk)
for row in documents:
    owned = by_doc[row['doc_id']]
    signature = hashlib.sha256(json.dumps(sorted((str(c.get('title','')),str(c.get('content',''))) for c in owned), ensure_ascii=False).encode()).hexdigest()
    groups[stable_logical_document_id(row)[0]].append({
        'doc_id': row['doc_id'], 'title': row.get('canonical_title'), 'content_hash': row.get('content_hash'),
        'source_hash': row.get('source_hash'), 'chunks':len(owned), 'chunk_signature':signature,
        'version_id':row.get('version_id'), 'is_active':row.get('is_active')})
result = {'collection_names':collections, 'groups':dict(groups), 'orphan_doc_ids':sorted(set(by_doc)-{r['doc_id'] for r in documents}), 'mongo_pointers':snapshot['mongo_pointers']}
(OUT / 'inspection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print('snapshot_bytes', path.stat().st_size)
print('duplicates', json.dumps([v for v in groups.values() if len(v)>1], ensure_ascii=False, indent=2))
print('orphans', result['orphan_doc_ids'])
print('mongo_pointers',len(snapshot['mongo_pointers']))
