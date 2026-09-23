import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import UploadFile
from pymongo.errors import DuplicateKeyError

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.main_graph import route_resume
from knowledge.processor.import_processor.nodes.bge_embedding_chunks_node import (
    BGEEmbeddingChunksNode,
)
from knowledge.processor.import_processor.nodes.milvus_import_node import (
    MilvusImportNode,
    _MilvusSchemaBuilder,
)
from knowledge.service.file_process_service import FileProcessService
from knowledge.utils.mongo_import_registry_util import (
    IMPORT_STATUS_COMPLETED,
    ImportClaim,
    claim_import,
)
from knowledge.utils.import_checkpoint_util import (
    load_import_checkpoint,
    save_import_checkpoint,
)
from knowledge.utils.task_util import task_info_from_import_record
from knowledge.utils.document_version_util import activate_document_version
from knowledge.utils.milvus_util import _doc_ids_filter


class FileHashIdempotencyTest(unittest.TestCase):
    def test_save_file_calculates_sha256_while_copying(self):
        payload = b"same scientific paper\x00content"
        upload = UploadFile(filename="paper.pdf", file=io.BytesIO(payload))
        service = FileProcessService()

        with tempfile.TemporaryDirectory() as directory:
            path, source_hash = service._save_file_to_local(upload, directory)

            self.assertEqual(payload, path.read_bytes())
            self.assertEqual(hashlib.sha256(payload).hexdigest(), source_hash)

    def test_completed_duplicate_skips_expensive_import_steps(self):
        payload = b"duplicate paper"
        upload = UploadFile(filename="paper.pdf", file=io.BytesIO(payload))
        service = FileProcessService()
        source_hash = hashlib.sha256(payload).hexdigest()
        claim = ImportClaim(
            acquired=False,
            record={
                "source_hash": source_hash,
                "task_id": "existing-task",
                "status": IMPORT_STATUS_COMPLETED,
                "version_id": f"ver_{source_hash[:24]}",
                "doc_id": "doc_existing",
            },
        )

        with tempfile.TemporaryDirectory() as directory, patch(
            "knowledge.service.file_process_service.get_local_base_dir",
            return_value=directory,
        ), patch(
            "knowledge.service.file_process_service.claim_import",
            return_value=claim,
        ), patch.object(service, "_save_file_to_minio") as save_to_minio:
            result = service.process_upload_file(upload)

        self.assertFalse(result.should_process)
        self.assertEqual("existing-task", result.task_id)
        self.assertEqual("doc_existing", result.doc_id)
        self.assertEqual(IMPORT_STATUS_COMPLETED, result.duplicate_status)
        save_to_minio.assert_not_called()

    def test_first_upload_continues_into_import_pipeline(self):
        payload = b"new paper"
        upload = UploadFile(filename="paper.pdf", file=io.BytesIO(payload))
        service = FileProcessService()
        source_hash = hashlib.sha256(payload).hexdigest()
        claim = ImportClaim(
            acquired=True,
            record={"source_hash": source_hash, "status": "processing"},
        )

        with tempfile.TemporaryDirectory() as directory, patch(
            "knowledge.service.file_process_service.get_local_base_dir",
            return_value=directory,
        ), patch(
            "knowledge.service.file_process_service.claim_import",
            return_value=claim,
        ), patch.object(service, "_save_file_to_minio") as save_to_minio:
            result = service.process_upload_file(upload)

            self.assertTrue(result.should_process)
            self.assertEqual(source_hash, result.source_hash)
            self.assertEqual(f"ver_{source_hash[:24]}", result.version_id)
            self.assertTrue(result.import_file_path.exists())
            save_to_minio.assert_called_once()

    def test_reuploaded_failed_file_automatically_uses_checkpoint(self):
        payload = b"retry paper"
        upload = UploadFile(filename="paper.pdf", file=io.BytesIO(payload))
        service = FileProcessService()
        source_hash = hashlib.sha256(payload).hexdigest()
        claim = ImportClaim(
            acquired=True,
            record={
                "source_hash": source_hash,
                "task_id": "new-task",
                "status": "processing",
                "attempt": 2,
                "checkpoint_path": "old/.checkpoints/old.state.json.gz",
                "completed_nodes": ["entry_node", "pdf_to_md_node"],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            recovered_path = Path(directory) / "old" / "paper.pdf"
            recovered_path.parent.mkdir()
            recovered_path.write_bytes(payload)
            checkpoint_state = {
                "task_id": "new-task",
                "source_hash": source_hash,
                "file_dir": str(recovered_path.parent),
                "import_file_path": str(recovered_path),
                "resume_after_node": "pdf_to_md_node",
                "resource_objects": ["document_versions/ver/origin/paper.pdf"],
            }
            with patch(
                "knowledge.service.file_process_service.get_local_base_dir",
                return_value=directory,
            ), patch(
                "knowledge.service.file_process_service.claim_import",
                return_value=claim,
            ), patch(
                "knowledge.service.file_process_service.find_active_version_by_source_hash",
                return_value={},
            ), patch(
                "knowledge.service.file_process_service.get_settings",
                return_value=SimpleNamespace(import_checkpoint_enabled=True),
            ), patch.object(
                service, "_load_retry_state", return_value=checkpoint_state
            ), patch.object(
                service, "_save_file_to_minio"
            ) as save_to_minio, patch.object(
                service, "_remove_task_directory"
            ):
                result = service.process_upload_file(upload)

        self.assertTrue(result.should_process)
        self.assertEqual(recovered_path, result.import_file_path)
        self.assertEqual("pdf_to_md_node", result.checkpoint_state["resume_after_node"])
        save_to_minio.assert_not_called()

    def test_successful_graph_marks_hash_record_completed(self):
        service = FileProcessService()
        graph = Mock()
        graph.stream.return_value = [{
            "document_registry_node": {
                "doc_id": "doc_1",
                "chunks": [{"id": 101}],
                "document_identity": {
                    "doc_id": "doc_1",
                    "content_hash": "content-hash",
                    "logical_document_id": "ldoc_1",
                    "version_id": "ver_source",
                    "source_hash": "source-hash",
                },
            }
        }]
        settings = SimpleNamespace(
            keep_import_artifacts=True,
            keep_failed_artifacts=True,
        )

        with patch(
            "knowledge.service.file_process_service.create_import_graph",
            return_value=graph,
        ), patch(
            "knowledge.service.file_process_service.mark_import_completed",
        ) as mark_completed, patch(
            "knowledge.service.file_process_service.mark_import_failed",
        ) as mark_failed, patch(
            "knowledge.service.file_process_service.get_settings",
            return_value=settings,
        ), patch(
            "knowledge.service.file_process_service.activate_document_version",
        ) as activate, patch(
            "knowledge.service.file_process_service.cleanup_retired_version",
        ):
            service.run_main_graph(
                Path("paper.pdf"),
                "task-dir",
                "task-1",
                "source-hash",
                "ver_source",
            )

        mark_completed.assert_called_once_with(
            "source-hash",
            "task-1",
            doc_id="doc_1",
            content_hash="content-hash",
            version_id="ver_source",
            logical_document_id="ldoc_1",
        )
        activate.assert_called_once()
        mark_failed.assert_not_called()

    def test_failed_graph_releases_hash_for_retry(self):
        service = FileProcessService()
        graph = Mock()
        graph.stream.side_effect = RuntimeError("pipeline failed")
        settings = SimpleNamespace(
            keep_import_artifacts=True,
            keep_failed_artifacts=True,
        )

        with patch(
            "knowledge.service.file_process_service.create_import_graph",
            return_value=graph,
        ), patch(
            "knowledge.service.file_process_service.mark_import_completed",
        ) as mark_completed, patch(
            "knowledge.service.file_process_service.mark_import_failed",
        ) as mark_failed, patch(
            "knowledge.service.file_process_service.get_settings",
            return_value=settings,
        ), patch(
            "knowledge.service.file_process_service.discard_unpublished_version",
        ):
            service.run_main_graph(
                Path("paper.pdf"),
                "task-dir",
                "task-1",
                "source-hash",
                "ver_source",
            )

        mark_completed.assert_not_called()
        mark_failed.assert_called_once_with(
            "source-hash",
            "task-1",
            "pipeline failed",
        )

    def test_failed_graph_keeps_external_artifacts_when_resume_is_enabled(self):
        service = FileProcessService()
        graph = Mock()
        graph.stream.side_effect = RuntimeError("pipeline failed")
        settings = SimpleNamespace(
            keep_import_artifacts=False,
            keep_failed_artifacts=True,
            import_checkpoint_enabled=True,
        )

        with patch(
            "knowledge.service.file_process_service.create_import_graph",
            return_value=graph,
        ), patch(
            "knowledge.service.file_process_service.mark_import_failed",
        ), patch(
            "knowledge.service.file_process_service.get_settings",
            return_value=settings,
        ), patch(
            "knowledge.service.file_process_service.discard_unpublished_version",
        ) as discard:
            service.run_main_graph(
                Path("paper.pdf"),
                "task-dir",
                "task-1",
                "source-hash",
                "ver_source",
            )

        discard.assert_not_called()

    def test_prepare_retry_restores_checkpoint_state(self):
        service = FileProcessService()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "paper.pdf"
            source_path.write_bytes(b"paper")
            record = {
                "task_id": "task-1",
                "source_hash": "a" * 64,
                "version_id": "ver_a",
                "status": "processing",
                "completed_nodes": ["entry_node", "pdf_to_md_node"],
            }
            state = {
                "task_id": "task-1",
                "source_hash": "a" * 64,
                "version_id": "ver_a",
                "file_dir": directory,
                "import_file_path": str(source_path),
                "resume_after_node": "pdf_to_md_node",
                "resource_objects": ["documents/ver_a/paper.pdf"],
            }
            with patch(
                "knowledge.service.file_process_service.reclaim_import_for_retry",
                return_value=record,
            ), patch(
                "knowledge.service.file_process_service.load_import_checkpoint",
                return_value=state,
            ):
                preparation = service.prepare_retry("task-1")

        self.assertTrue(preparation.should_process)
        self.assertEqual("pdf_to_md_node", preparation.checkpoint_state["resume_after_node"])
        self.assertEqual(("documents/ver_a/paper.pdf",), preparation.resource_objects)


class ImportCheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_restores_sparse_vector_keys(self):
        collection = Mock()
        collection.update_one.return_value = SimpleNamespace(matched_count=1)
        with tempfile.TemporaryDirectory() as directory, patch(
            "knowledge.utils.import_checkpoint_util._get_collection",
            return_value=collection,
        ), patch(
            "knowledge.utils.import_checkpoint_util.get_settings",
            return_value=SimpleNamespace(import_claim_lease_seconds=3600),
        ):
            source_path = Path(directory) / "paper.pdf"
            source_path.write_bytes(b"paper")
            state = {
                "task_id": "task-1",
                "source_hash": "a" * 64,
                "file_dir": directory,
                "import_file_path": str(source_path),
                "chunks": [{"sparse_vector": {7: 0.5}}],
                "embedding_completed_batches": [0],
            }
            checkpoint_path = save_import_checkpoint(
                state, checkpoint_node="document_split_node"
            )
            loaded = load_import_checkpoint({
                "checkpoint_path": checkpoint_path,
                "file_dir": directory,
            })

        self.assertEqual("document_split_node", loaded["resume_after_node"])
        self.assertEqual({7: 0.5}, loaded["chunks"][0]["sparse_vector"])
        collection.update_one.assert_called_once()

    def test_resume_router_selects_next_node(self):
        self.assertEqual(
            "md_image_node",
            route_resume({"resume_after_node": "pdf_to_md_node"}),
        )
        self.assertEqual(
            "__end__",
            route_resume({"resume_after_node": "document_registry_node"}),
        )

    def test_embedding_resume_skips_completed_batches(self):
        node = BGEEmbeddingChunksNode(
            config=ImportConfig(embedding_batch_size=2)
        )
        model = Mock()
        model.encode.return_value = {
            "dense_vecs": [SimpleNamespace(tolist=lambda: [0.3, 0.4])],
            "lexical_weights": [{9: 0.8}],
        }
        chunks = [
            {"content": "a", "dense_vector": [0.1], "sparse_vector": {1: 1.0}},
            {"content": "b", "dense_vector": [0.2], "sparse_vector": {2: 1.0}},
            {"content": "c"},
        ]
        state = {
            "task_id": "task-1",
            "source_hash": "a" * 64,
            "file_dir": "task-dir",
            "chunks": chunks,
            "embedding_completed_batches": [0],
        }
        with patch(
            "knowledge.processor.import_processor.nodes.bge_embedding_chunks_node.AIClients.get_bge_m3_client",
            return_value=model,
        ), patch(
            "knowledge.processor.import_processor.nodes.bge_embedding_chunks_node.get_settings",
            return_value=SimpleNamespace(
                import_checkpoint_enabled=True,
                import_debug_artifacts=False,
            ),
        ), patch(
            "knowledge.processor.import_processor.nodes.bge_embedding_chunks_node.save_import_checkpoint",
        ) as save_checkpoint:
            result = node.process(state)

        model.encode.assert_called_once()
        self.assertEqual([0, 1], result["embedding_completed_batches"])
        self.assertEqual([0.3, 0.4], result["chunks"][2]["dense_vector"])
        save_checkpoint.assert_called_once()

    def test_persisted_record_can_rebuild_status_after_restart(self):
        info = task_info_from_import_record({
            "status": "failed",
            "completed_nodes": ["entry_node", "pdf_to_md_node"],
            "source_hash": "a" * 64,
            "error": "embedding unavailable",
        })
        self.assertEqual("failed", info["status"])
        self.assertEqual(
            ["上传文件", "检查文件", "PDF转Markdown"],
            info["done_list"],
        )
        self.assertEqual("embedding unavailable", info["result"]["error"])


class ImportRegistryClaimTest(unittest.TestCase):
    def test_existing_completed_hash_is_not_acquired(self):
        existing = {
            "source_hash": "a" * 64,
            "task_id": "old-task",
            "status": IMPORT_STATUS_COMPLETED,
        }
        collection = Mock()
        collection.insert_one.side_effect = DuplicateKeyError("duplicate")
        collection.find_one_and_update.return_value = None
        collection.find_one.return_value = existing

        with patch(
            "knowledge.utils.mongo_import_registry_util.ensure_import_registry_indexes",
        ), patch(
            "knowledge.utils.mongo_import_registry_util._get_collection",
            return_value=collection,
        ), patch(
            "knowledge.utils.mongo_import_registry_util.get_settings",
            return_value=SimpleNamespace(import_claim_lease_seconds=3600),
        ):
            result = claim_import("a" * 64, "new-task", "paper.pdf")

        self.assertFalse(result.acquired)
        self.assertEqual(existing, result.record)

    def test_existing_processing_hash_keeps_original_owner(self):
        existing = {
            "source_hash": "c" * 64,
            "task_id": "running-task",
            "status": "processing",
        }
        collection = Mock()
        collection.insert_one.side_effect = DuplicateKeyError("duplicate")
        collection.find_one_and_update.return_value = None
        collection.find_one.return_value = existing

        with patch(
            "knowledge.utils.mongo_import_registry_util.ensure_import_registry_indexes",
        ), patch(
            "knowledge.utils.mongo_import_registry_util._get_collection",
            return_value=collection,
        ), patch(
            "knowledge.utils.mongo_import_registry_util.get_settings",
            return_value=SimpleNamespace(import_claim_lease_seconds=3600),
        ):
            result = claim_import("c" * 64, "duplicate-task", "paper.pdf")

        self.assertFalse(result.acquired)
        self.assertEqual("running-task", result.record["task_id"])

    def test_failed_hash_can_be_atomically_reclaimed(self):
        reclaimed = {
            "source_hash": "b" * 64,
            "task_id": "new-task",
            "status": "processing",
            "attempt": 2,
        }
        collection = Mock()
        collection.insert_one.side_effect = DuplicateKeyError("duplicate")
        collection.find_one_and_update.return_value = reclaimed

        with patch(
            "knowledge.utils.mongo_import_registry_util.ensure_import_registry_indexes",
        ), patch(
            "knowledge.utils.mongo_import_registry_util._get_collection",
            return_value=collection,
        ), patch(
            "knowledge.utils.mongo_import_registry_util.get_settings",
            return_value=SimpleNamespace(import_claim_lease_seconds=3600),
        ):
            result = claim_import("b" * 64, "new-task", "paper.pdf")

        self.assertTrue(result.acquired)
        self.assertEqual("new-task", result.record["task_id"])


class ChunkWriteIdempotencyTest(unittest.TestCase):
    def test_new_collection_schema_disables_auto_id(self):
        client = Mock()
        schema = Mock()
        client.create_schema.return_value = schema

        _MilvusSchemaBuilder.build_schema(client, dim=1024)

        id_calls = [
            call.kwargs
            for call in schema.add_field.call_args_list
            if call.kwargs.get("field_name") == "id"
        ]
        self.assertEqual(1, len(id_calls))
        self.assertFalse(id_calls[0]["auto_id"])
        field_names = {
            call.kwargs.get("field_name")
            for call in schema.add_field.call_args_list
        }
        self.assertTrue({
            "logical_document_id", "version_id", "version_status", "is_active"
        } <= field_names)

    def test_chunk_primary_key_is_isolated_by_version(self):
        base = {
            "doc_id": "doc_1",
            "section_id": "section_1",
            "chunk_index": 0,
            "content": "same content",
        }
        first = MilvusImportNode._stable_chunk_id(
            {**base, "version_id": "ver_1"}, 0
        )
        second = MilvusImportNode._stable_chunk_id(
            {**base, "version_id": "ver_2"}, 0
        )
        self.assertNotEqual(first, second)

    def test_active_version_filter_is_combined_with_document_scope(self):
        expression, params = _doc_ids_filter(
            ["doc_1"], active_only=True
        )
        self.assertEqual(
            "is_active == true and doc_id in {doc_ids}", expression
        )
        self.assertEqual({"doc_ids": ["doc_1"]}, params)

    def test_explicit_primary_key_collection_uses_upsert(self):
        client = Mock()
        client.has_collection.return_value = True
        client.describe_collection.return_value = {"auto_id": False}
        client.upsert.return_value = {"ids": [123]}
        node = MilvusImportNode(
            config=ImportConfig(
                chunks_collection="chunks",
                bm25_enabled=True,
            )
        )
        chunks = [{
            "doc_id": "doc_1",
            "section_id": "section_1",
            "chunk_index": 0,
            "content": "content",
        }]

        with patch(
            "knowledge.processor.import_processor.nodes.milvus_import_node.StorageClients.get_milvus",
            return_value=client,
        ):
            result = node._insert_chunks_to_milvus(chunks, 1024)

        client.upsert.assert_called_once()
        client.insert.assert_not_called()
        self.assertEqual(123, result[0]["chunk_id"])
        self.assertIsInstance(result[0]["id"], int)


class DocumentVersionActivationTest(unittest.TestCase):
    def test_new_version_is_published_before_previous_version_is_retired(self):
        client = Mock()
        client.query.return_value = [{"id": 11}, {"id": 12}]
        previous = {
            "active_version_id": "ver_old",
            "active_doc_id": "doc_old",
        }
        config = ImportConfig(
            chunks_collection="chunks",
            document_registry_collection="documents",
        )
        identity = {
            "logical_document_id": "ldoc_abc",
            "version_id": "ver_new",
            "doc_id": "doc_new",
            "source_hash": "a" * 64,
        }

        with patch(
            "knowledge.utils.document_version_util.ImportConfig.from_env",
            return_value=config,
        ), patch(
            "knowledge.utils.document_version_util.StorageClients.get_milvus",
            return_value=client,
        ), patch(
            "knowledge.utils.document_version_util.switch_active_version",
            return_value=previous,
        ) as switch:
            result = activate_document_version(
                identity=identity,
                chunks=[{"id": 21}, {"chunk_id": 22}],
                task_id="task-1",
            )

        switch.assert_called_once()
        self.assertEqual("ver_old", result.previous_version_id)
        self.assertEqual((11, 12), result.retired_chunk_ids)
        chunk_updates = [
            call.kwargs["data"]
            for call in client.upsert.call_args_list
            if call.kwargs["collection_name"] == "chunks"
        ]
        self.assertTrue(chunk_updates[0][0]["is_active"])
        self.assertEqual("ready", chunk_updates[0][0]["version_status"])
        self.assertFalse(chunk_updates[1][0]["is_active"])
        self.assertEqual("retired", chunk_updates[1][0]["version_status"])

    def test_pointer_failure_hides_the_unpublished_new_version(self):
        client = Mock()
        config = ImportConfig(
            chunks_collection="chunks",
            document_registry_collection="documents",
        )
        identity = {
            "logical_document_id": "ldoc_abc",
            "version_id": "ver_new",
            "doc_id": "doc_new",
            "source_hash": "a" * 64,
        }

        with patch(
            "knowledge.utils.document_version_util.ImportConfig.from_env",
            return_value=config,
        ), patch(
            "knowledge.utils.document_version_util.StorageClients.get_milvus",
            return_value=client,
        ), patch(
            "knowledge.utils.document_version_util.switch_active_version",
            side_effect=RuntimeError("mongo unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                activate_document_version(
                    identity=identity,
                    chunks=[{"id": 21}],
                    task_id="task-1",
                )

        chunk_updates = [
            call.kwargs["data"]
            for call in client.upsert.call_args_list
            if call.kwargs["collection_name"] == "chunks"
        ]
        self.assertFalse(chunk_updates[-1][0]["is_active"])
        self.assertEqual("failed", chunk_updates[-1][0]["version_status"])


if __name__ == "__main__":
    unittest.main()
