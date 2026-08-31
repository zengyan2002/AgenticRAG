import os
import shutil
from pathlib import Path

from datetime import datetime

import uuid
import time
from fastapi import UploadFile
import logging

from knowledge.core.paths import get_local_base_dir
from knowledge.core.settings import get_settings
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import create_import_graph
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED, add_running_task, add_done_task, add_node_duration

logger = logging.getLogger(__name__)
class FileProcessService():
    def process_upload_file(self,file: UploadFile):

        # 1. 生成任务id
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
            import_file_path_obj = self._save_file_to_local(file,file_dir)

        except Exception as e:
            logger.error(e)
            update_task_status(task_id,TASK_STATUS_FAILED)
            return None, None, task_id

        #3 将文件保存到Minio（备份，报错也没事）
        self._save_file_to_minio(import_file_path_obj)

        # 把上传文件放进done列表
        add_done_task(task_id, "upload_file")
        #结束时间
        end_time = time.perf_counter()

        add_node_duration(task_id,"upload_file",end_time-start_time)

        return import_file_path_obj, file_dir, task_id


    def _save_file_to_local(self, file: UploadFile,file_dir:str)->Path:


        #创建临时文件目录
        os.makedirs(file_dir, exist_ok=True)

        #导入文件的存放路径
        import_file_path_obj = Path(file_dir)/file.filename

        #写入文件
        try:
            with open(import_file_path_obj,"wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logger.error(f"Failed to save the uploaded file to {import_file_path_obj}: {e}")
            raise FileProcessingError(f"Failed to save the uploaded file to {import_file_path_obj}: {e}")


        return import_file_path_obj


    def _save_file_to_minio(self,import_file_path_obj: Path):

        try:
            # 创建Minio客户端
            minio_client = StorageClients.get_minio()

            # 提取bucket名
            bucket_name = get_settings().require("minio_bucket_name")[0]

            # 设置存入的object名
            object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{import_file_path_obj.name}"
            #利用客户端进行存储
            minio_client.fput_object(bucket_name=bucket_name, object_name=object_name,file_path=str(import_file_path_obj))

            logger.info(f"Successfully save the file to minio")

        except Exception as e:
            logger.warning(f"Failed to save the uploaded file to Minio,{e}")

    def run_main_graph(self,import_file_path_obj:Path,file_dir:str,task_id:str):
        # 1 定义state
        state = {
            "import_file_path": str(import_file_path_obj),
            "file_dir": file_dir,
            "task_id": task_id
        }
        # 2 执行主流程
        succeeded = False
        try:
            compiled_graph = create_import_graph()
            for event in compiled_graph.stream(state):
                for node_name, node_state in event.items():
                    logger.debug("导入节点完成: %s", node_name)

            #代表整个任务执行完成
            update_task_status(task_id,TASK_STATUS_COMPLETED)
            succeeded = True
        except Exception as e:
            logger.error(f"Failed to run main graph,{e}")
            update_task_status(task_id,TASK_STATUS_FAILED)
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












