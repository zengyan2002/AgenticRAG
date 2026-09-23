"""Read-only collection counts to diagnose an empty retrieval evaluation."""
import json
from pathlib import Path
import sys
import re
import unicodedata
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'evaluation/results/quality_20260906/knowledge_base_diagnostic.json')
args = parser.parse_args()
from knowledge.processor.query_processor.config import get_config
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.core.settings import get_settings

config = get_config()
client = StorageClients.get_milvus()
report = {"active_version_filter_enabled": config.active_version_filter_enabled, "collections": {}}
for name in sorted({config.chunks_collection, config.bm25_collection, config.document_registry_collection}):
    entry = {}
    for label, expression in (("total", ""), ("active", "is_active == true"), ("inactive", "is_active == false")):
        try:
            entry[label] = client.query(collection_name=name, filter=expression,
                                         output_fields=["count(*)"], timeout=20)
        except Exception as exc:
            entry[label] = {"error": type(exc).__name__}
    entry["schema_fields"] = [f["name"] for f in client.describe_collection(name, timeout=20)["fields"]]
    entry["samples"] = client.query(collection_name=name, filter="", limit=3,
        output_fields=["doc_id", "canonical_title", "is_active", "version_id", "version_status"], timeout=20)
    report["collections"][name] = entry
registry = client.query(collection_name=config.document_registry_collection, filter="", limit=1000,
                        output_fields=["canonical_title", "aliases_json"], timeout=20)
benchmark = json.loads((ROOT / "evaluation/benchmark.json").read_text(encoding="utf-8"))
def normalize(value):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", value).casefold())
available = {normalize(row.get("canonical_title") or "") for row in registry}
report["benchmark_title_coverage"] = {doc["id"]: normalize(doc["canonical_title"]) in available
                                       for doc in benchmark["documents"]}
active_registry = client.query(collection_name=config.document_registry_collection,filter='is_active == true',limit=1000,
                               output_fields=['canonical_title'],timeout=20)
active_titles = {normalize(r.get('canonical_title') or '') for r in active_registry}
report['benchmark_active_title_coverage'] = {doc['id']:normalize(doc['canonical_title']) in active_titles for doc in benchmark['documents']}
report['mongo_active_version_pointers'] = StorageClients.get_mongo_db()[get_settings().document_version_registry_collection].count_documents({'status':'ready'})
path = args.output
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
