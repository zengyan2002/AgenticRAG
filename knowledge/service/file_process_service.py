import os
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path

from datetime import datetime

import uuid
import time
from typing import Any
from fastapi import UploadFile
import logging

from knowledge.core.paths import get_local_base_dir
from knowledge.core.settings import get_settings
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import create_import_graph
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.mongo_import_registry_util import (
    IMPORT_STATUS_COMPLETED,
    IMPORT_STATUS_PROCESSING,
    claim_import,
    reclaim_import_for_retry,
    mark_import_completed,
    mark_import_failed,
)
from knowledge.utils.document_version_util import (
    activate_document_version,
    cleanup_retired_version,
    discard_unpublished_version,
    find_active_version_by_source_hash,
)
from knowledge.utils.document_identity_util import stable_logical_document_id
from knowledge.utils.import_checkpoint_util import load_import_checkpoint
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED, add_running_task, add_done_task, add_node_duration, set_task_result

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadPreparation:
    """上传文件保存及幂等检查结果。"""

    import_file_path: Path | None
    file_dir: str | None
    task_id: str
    source_hash: str = ""
    version_id: str = ""
    logical_document_id: str = ""
    resource_objects: tuple[str, ...] = ()
    should_process: bool = False
    duplicate_status: str = ""
    doc_id: str = ""
    error: str = ""
    checkpoint_state: dict[str, Any] | None = None


class FileProcessService():
    def process_upload_file(
            self,
            file: UploadFile,
            logical_document_id: str = "",
    ) -> UploadPreparation:

        # 1. 生成任务id
        """保存上传文件并完成进入主流程前的幂等检查。

        Args:
            file: 用户上传的文件。
            logical_document_id: 同一篇文档跨版本复用的业务标识。

        Returns:
            处理结果。
        """
        task_id = str(uuid.uuid4().hex[:8])

        #把上传文件放进running列表
        add_running_task(task_id,"upload_file")

        # 整个任务开始了，应该设置整个任务的状态为 RUNNING
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        #开始时间
        start_time = time.perf_counter()

        # 生成临时目录
        temp_file_dir = os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))
        file_dir = os.path.join(temp_file_dir,task_id)

        #2 将上传的文件保存到本地临时文件目录
        try:
            import_file_path_obj, source_hash = self._save_file_to_local(file, file_dir)

        except Exception as e:
            logger.error(e)
            update_task_status(task_id,TASK_STATUS_FAILED)
            return UploadPreparation(None, None, task_id)

        version_id = f"ver_{source_hash[:24]}"
        requested_logical_id = ""
        if logical_document_id.strip():
            requested_logical_id, _ = stable_logical_document_id(
                {}, logical_document_id
            )
        set_task_result(task_id, "source_hash", source_hash)
        set_task_result(task_id, "version_id", version_id)
        if requested_logical_id:
            set_task_result(
                task_id, "logical_document_id", requested_logical_id
            )

        # MongoDB 唯一索引在这里完成原子抢占。只有抢占成功的请求才会
        # 继续执行 MinIO、MinerU、VLM 和 Embedding 等昂贵步骤。
        try:
            claim = claim_import(
                source_hash=source_hash,
                task_id=task_id,
                file_name=import_file_path_obj.name,
                logical_document_id=requested_logical_id,
            )
        except Exception as exc:
            logger.error("入库幂等检查失败: %s", exc)
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(exc))
            self._remove_task_directory(file_dir)
            return UploadPreparation(
                None,
                None,
                task_id,
                source_hash=source_hash,
                version_id=version_id,
                logical_document_id=requested_logical_id,
                error="入库幂等检查服务不可用",
            )

        if not claim.acquired:
            record = claim.record
            duplicate_status = str(record.get("status") or "")
            existing_task_id = str(record.get("task_id") or task_id)
            doc_id = str(record.get("doc_id") or "")
            resolved_logical_id = str(
                record.get("logical_document_id")
                or record.get("requested_logical_document_id")
                or requested_logical_id
                or ""
            )

            # 当前请求不再需要自己的临时文件；接口返回原任务 ID，调用方
            # 可以继续轮询正在执行的任务，或直接获得已完成结果。
            self._remove_task_directory(file_dir)
            add_done_task(task_id, "upload_file")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            update_task_status(
                existing_task_id,
                TASK_STATUS_COMPLETED
                if duplicate_status == IMPORT_STATUS_COMPLETED
                else TASK_STATUS_PROCESSING,
            )
            set_task_result(existing_task_id, "source_hash", source_hash)
            set_task_result(
                existing_task_id,
                "version_id",
                str(record.get("version_id") or version_id),
            )
            set_task_result(existing_task_id, "doc_id", doc_id)
            set_task_result(
                existing_task_id,
                "logical_document_id",
                resolved_logical_id,
            )
            set_task_result(existing_task_id, "duplicate", "true")
            if duplicate_status == IMPORT_STATUS_COMPLETED:
                add_done_task(existing_task_id, "upload_file")
            return UploadPreparation(
                None,
                None,
                existing_task_id,
                source_hash=source_hash,
                version_id=str(record.get("version_id") or version_id),
                logical_document_id=resolved_logical_id,
                should_process=False,
                duplicate_status=duplicate_status,
                doc_id=doc_id,
            )

        # 极端情况下进程可能在“活动版本已切换、幂等记录尚未完成”之间
        # 退出。租约到期后的重试先检查活动指针，避免把同一版本重新写成
        # building/inactive 而造成短暂不可用。
        claim_attempt = int(claim.record.get("attempt") or 1)
        active_version = (
            find_active_version_by_source_hash(source_hash)
            if claim_attempt > 1 else {}
        )
        if active_version:
            active_doc_id = str(active_version.get("active_doc_id") or "")
            active_logical_id = str(
                active_version.get("logical_document_id") or ""
            )
            try:
                mark_import_completed(
                    source_hash,
                    task_id,
                    doc_id=active_doc_id,
                    content_hash=str(active_version.get("content_hash") or ""),
                    version_id=str(
                        active_version.get("active_version_id") or version_id
                    ),
                    logical_document_id=active_logical_id,
                )
            finally:
                self._remove_task_directory(file_dir)
                previous_file_dir = str(claim.record.get("file_dir") or "")
                if (
                    previous_file_dir
                    and Path(previous_file_dir).resolve()
                    != Path(file_dir).resolve()
                ):
                    self._remove_task_directory(previous_file_dir)
            add_done_task(task_id, "upload_file")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            return UploadPreparation(
                None,
                None,
                task_id,
                source_hash=source_hash,
                version_id=str(
                    active_version.get("active_version_id") or version_id
                ),
                logical_document_id=active_logical_id,
                should_process=False,
                duplicate_status=IMPORT_STATUS_COMPLETED,
                doc_id=active_doc_id,
            )

        # 失败任务或租约过期任务被重新抢占后，优先复用旧任务目录中的
        # 最近断点。新上传的同一文件只是触发恢复，不再重复上传 MinIO，
        # 也不从入口节点重新执行。
        if (
            claim_attempt > 1
            and get_settings().import_checkpoint_enabled
            and claim.record.get("checkpoint_path")
        ):
            try:
                checkpoint_state = self._load_retry_state(
                    claim.record,
                    task_id=task_id,
                    source_hash=source_hash,
                )
                recovered_file_path = Path(
                    str(checkpoint_state["import_file_path"])
                )
                recovered_file_dir = str(checkpoint_state["file_dir"])
                recovered_logical_id = str(
                    checkpoint_state.get("logical_document_id")
                    or claim.record.get("logical_document_id")
                    or claim.record.get("requested_logical_document_id")
                    or requested_logical_id
                    or ""
                )
                checkpoint_state["logical_document_id"] = recovered_logical_id
                self._remove_task_directory(file_dir)
                add_done_task(task_id, "upload_file")
                for node_name in claim.record.get("completed_nodes") or []:
                    add_done_task(task_id, str(node_name))
                add_node_duration(
                    task_id, "upload_file", time.perf_counter() - start_time
                )
                return UploadPreparation(
                    recovered_file_path,
                    recovered_file_dir,
                    task_id,
                    source_hash=source_hash,
                    version_id=version_id,
                    logical_document_id=recovered_logical_id,
                    resource_objects=tuple(
                        checkpoint_state.get("resource_objects") or []
                    ),
                    should_process=True,
                    checkpoint_state=checkpoint_state,
                )
            except Exception as exc:
                # 断点文件丢失或损坏时仍可用这次上传的新文件完整重跑。
                # 下游 Milvus 使用稳定主键 upsert，因此重复副作用可收敛。
                logger.warning("断点恢复不可用，将从头重新入库: %s", exc)

        #3 将文件保存到Minio（备份，报错也没事）
        source_object = self._save_file_to_minio(
            import_file_path_obj,
            version_id=version_id,
        )

        # 把上传文件放进done列表
        add_done_task(task_id, "upload_file")
        #结束时间
        end_time = time.perf_counter()

        add_node_duration(task_id,"upload_file",end_time-start_time)

        return UploadPreparation(
            import_file_path_obj,
            file_dir,
            task_id,
            source_hash=source_hash,
            version_id=version_id,
            logical_document_id=requested_logical_id,
            resource_objects=(source_object,) if source_object else (),
            should_process=True,
        )

    def prepare_retry(self, task_id: str) -> UploadPreparation:
        """领取失败任务，并从最近一个持久化断点恢复执行参数。"""
        record: dict[str, Any] = {}
        try:
            record = reclaim_import_for_retry(task_id)
            state = self._load_retry_state(
                record,
                task_id=task_id,
                source_hash=str(record.get("source_hash") or ""),
            )
        except Exception as exc:
            # reclaim 成功而加载失败时要重新释放任务，避免它一直停留在
            # processing；reclaim 本身失败则该更新不会命中，也没有副作用。
            try:
                source_hash = str(record.get("source_hash") or "")
                if source_hash:
                    mark_import_failed(source_hash, task_id, str(exc))
            except Exception:
                logger.exception("恢复失败后回写任务状态失败")
            raise

        version_id = str(
            state.get("version_id")
            or record.get("version_id")
            or f"ver_{str(record.get('source_hash') or '')[:24]}"
        )
        logical_document_id = str(
            state.get("logical_document_id")
            or record.get("logical_document_id")
            or record.get("requested_logical_document_id")
            or ""
        )
        state["version_id"] = version_id
        state["logical_document_id"] = logical_document_id

        update_task_status(task_id, TASK_STATUS_PROCESSING)
        add_done_task(task_id, "upload_file")
        for node_name in record.get("completed_nodes") or []:
            add_done_task(task_id, str(node_name))
        set_task_result(task_id, "source_hash", str(record["source_hash"]))
        set_task_result(task_id, "version_id", version_id)
        set_task_result(task_id, "logical_document_id", logical_document_id)

        return UploadPreparation(
            Path(str(state["import_file_path"])),
            str(state["file_dir"]),
            task_id,
            source_hash=str(record["source_hash"]),
            version_id=version_id,
            logical_document_id=logical_document_id,
            resource_objects=tuple(state.get("resource_objects") or []),
            should_process=True,
            checkpoint_state=state,
        )

    @staticmethod
    def _load_retry_state(
            record: dict[str, Any],
            *,
            task_id: str,
            source_hash: str,
    ) -> dict[str, Any]:
        """加载断点并校验恢复所需的本地文件与任务身份。"""
        state = load_import_checkpoint(record)
        checkpoint_hash = str(state.get("source_hash") or "")
        if not source_hash or checkpoint_hash != source_hash:
            raise ValueError("断点与当前上传文件的 source_hash 不一致")

        file_dir = Path(str(state.get("file_dir") or "")).resolve()
        import_file_path = Path(
            str(state.get("import_file_path") or "")
        ).resolve()
        if not file_dir.is_dir():
            raise FileNotFoundError(f"断点任务目录不存在: {file_dir}")
        if not import_file_path.is_file() or file_dir not in import_file_path.parents:
            raise FileNotFoundError("断点对应的原始文件不存在或路径非法")

        state["task_id"] = task_id
        state["source_hash"] = source_hash
        state["file_dir"] = str(file_dir)
        state["import_file_path"] = str(import_file_path)
        return state


    def _save_file_to_local(self, file: UploadFile, file_dir: str) -> tuple[Path, str]:


        #创建临时文件目录
        """保存上传文件，并在复制过程中计算 SHA-256。

        Args:
            file: 用户上传的文件。
            file_dir: 本次文件处理使用的工作目录。

        Returns:
            处理结果。

        Raises:
            FileProcessingError: 输入无效或处理过程无法继续时抛出。
        """
        os.makedirs(file_dir, exist_ok=True)

        #导入文件的存放路径
        safe_file_name = Path(file.filename or "uploaded_file").name
        if safe_file_name in {"", ".", ".."}:
            raise FileProcessingError("上传文件名无效")
        import_file_path_obj = Path(file_dir) / safe_file_name

        # 保存文件时同步计算 SHA-256，避免为幂等检查再次读取大文件。
        try:
            digest = hashlib.sha256()
            with open(import_file_path_obj,"wb") as f:
                while True:
                    block = file.file.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    f.write(block)
        except Exception as e:
            logger.error(f"Failed to save the uploaded file to {import_file_path_obj}: {e}")
            raise FileProcessingError(f"Failed to save the uploaded file to {import_file_path_obj}: {e}")


        return import_file_path_obj, digest.hexdigest()


    def _save_file_to_minio(
            self,
            import_file_path_obj: Path,
            version_id: str = "",
    ) -> str:

        """尽力将原始上传文件备份到 MinIO。

        Args:
            import_file_path_obj: 待导入文件的 Path 对象。

        Returns:
            处理结果。
        """
        try:
            # 创建Minio客户端
            minio_client = StorageClients.get_minio()

            # 提取bucket名
            bucket_name = get_settings().require("minio_bucket_name")[0]

            # 设置存入的object名
            object_name = (
                f"document_versions/{version_id}/origin/{import_file_path_obj.name}"
                if version_id
                else f"origin_files/{datetime.now().strftime('%Y%m%d')}/{import_file_path_obj.name}"
            )
            #利用客户端进行存储
            minio_client.fput_object(bucket_name=bucket_name, object_name=object_name,file_path=str(import_file_path_obj))

            logger.info(f"Successfully save the file to minio")
            return object_name

        except Exception as e:
            logger.warning(f"Failed to save the uploaded file to Minio,{e}")
            return ""

    def run_main_graph(
            self,
            import_file_path_obj: Path,
            file_dir: str,
            task_id: str,
            source_hash: str,
            version_id: str,
            logical_document_id: str = "",
            resource_objects: tuple[str, ...] = (),
            checkpoint_state: dict[str, Any] | None = None,
    ):
        # 1 定义state
        """运行文档入库工作流并提交最终幂等状态。

        Args:
            import_file_path_obj: 待导入文件的 Path 对象。
            file_dir: 本次文件处理使用的工作目录。
            task_id: 异步任务唯一标识。
            source_hash: 原始上传文件的 SHA-256。
            version_id: 当前入库内容的版本标识。
            logical_document_id: 调用方显式指定的逻辑文档标识。
            resource_objects: 当前版本已经上传的对象存储资源清单。
            checkpoint_state: 最近一次成功断点保存的完整图状态；首次执行
                时为空。

        Returns:
            处理结果。
        """
        state = dict(checkpoint_state or {})
        state.update({
            "import_file_path": str(import_file_path_obj),
            "file_dir": file_dir,
            "task_id": task_id,
            "source_hash": source_hash,
            "version_id": version_id,
            "logical_document_id": logical_document_id,
            "resource_objects": list(resource_objects),
        })
        # 2 执行主流程
        succeeded = False
        final_state = state
        try:
            compiled_graph = create_import_graph()
            for event in compiled_graph.stream(state):
                for node_name, node_state in event.items():
                    logger.debug("导入节点完成: %s", node_name)
                    if isinstance(node_state, dict):
                        final_state = node_state

            identity = final_state.get("document_identity") or {}
            doc_id = str(final_state.get("doc_id") or identity.get("doc_id") or "")
            content_hash = str(identity.get("content_hash") or "")
            resolved_logical_id = str(
                identity.get("logical_document_id") or ""
            )
            if not doc_id or not content_hash or not resolved_logical_id:
                raise RuntimeError(
                    "入库完成但未生成有效的 doc_id/content_hash/logical_document_id"
                )
            activation = activate_document_version(
                identity=identity,
                chunks=final_state.get("chunks") or [],
                resource_objects=final_state.get("resource_objects") or [],
                task_id=task_id,
            )
            # 活动版本指针已经切换成功，后续幂等记录同步失败不能再把已发布
            # 的知识版本判为失败；租约过期后的同源重试仍会被稳定主键兜底。
            try:
                mark_import_completed(
                    source_hash,
                    task_id,
                    doc_id=doc_id,
                    content_hash=content_hash,
                    version_id=version_id,
                    logical_document_id=resolved_logical_id,
                )
            except Exception as registry_exc:
                logger.error("新版已发布，但同步入库幂等记录失败: %s", registry_exc)
                set_task_result(
                    task_id,
                    "registry_warning",
                    "活动版本已发布，但幂等记录同步失败",
                )
            set_task_result(task_id, "doc_id", doc_id)
            set_task_result(task_id, "source_hash", source_hash)
            set_task_result(task_id, "version_id", version_id)
            set_task_result(
                task_id,
                "logical_document_id",
                resolved_logical_id,
            )

            #代表整个任务执行完成
            update_task_status(task_id,TASK_STATUS_COMPLETED)
            succeeded = True
            # 状态切换完成后再做物理清理；清理失败不会影响新版服务。
            cleanup_retired_version(activation)
        except Exception as e:
            logger.error(f"Failed to run main graph,{e}")
            update_task_status(task_id,TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(e))
            try:
                mark_import_failed(source_hash, task_id, str(e))
            except Exception as registry_exc:
                logger.error("记录入库失败状态时发生异常: %s", registry_exc)
            settings = get_settings()
            if not (
                getattr(settings, "import_checkpoint_enabled", False)
                and settings.keep_failed_artifacts
            ):
                try:
                    failed_identity = dict(
                        final_state.get("document_identity") or {}
                    )
                    failed_identity.setdefault("version_id", version_id)
                    failed_identity.setdefault(
                        "doc_id", final_state.get("doc_id") or ""
                    )
                    discard_unpublished_version(
                        failed_identity,
                        resource_objects=(
                            final_state.get("resource_objects") or []
                        ),
                    )
                except Exception as cleanup_exc:
                    logger.warning("清理失败版本时发生异常: %s", cleanup_exc)
        finally:
            settings = get_settings()
            should_remove = (
                succeeded and not settings.keep_import_artifacts
            ) or (
                not succeeded and not settings.keep_failed_artifacts
            )
            if should_remove:
                self._remove_task_directory(file_dir)

    @staticmethod
    def _remove_task_directory(file_dir: str) -> None:
        """仅删除 temp_data 下的单个任务目录。"""
        task_dir = Path(file_dir).resolve()
        temp_root = Path(get_local_base_dir()).resolve()
        if task_dir == temp_root or temp_root not in task_dir.parents:
            logger.error("拒绝清理非任务目录: %s", task_dir)
            return
        try:
            shutil.rmtree(task_dir, ignore_errors=False)
            logger.info("已清理导入任务临时目录: %s", task_dir)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("清理导入任务临时目录失败: %s", exc)












