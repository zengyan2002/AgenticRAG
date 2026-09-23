import uvicorn
import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.api.deps import get_file_process_service
from knowledge.core.paths import get_front_page_dir
from knowledge.core.settings import get_settings
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.file_process_service import FileProcessService
from knowledge.utils.mongo_import_registry_util import get_import_record_by_task_id
from knowledge.utils.task_util import get_task_info, task_info_from_import_record


logger = logging.getLogger(__name__)


def register_router(router: APIRouter) -> None:
    """注册路由。

    Args:
        router: 用于注册接口的 FastAPI 路由器。

    Returns:
        None。
    """
    @router.post("/upload", response_model=UploadResponse)
    def upload_file(background_tasks: BackgroundTasks,
                    file: UploadFile,
                    logical_document_id: str = Form(""),
                    file_process_service: FileProcessService = Depends(get_file_process_service)):
        # 1 对上传的文件进行处理（保存）
        """上传文件。

        Args:
            background_tasks: FastAPI 后台任务调度器。
            file: 用户上传的文件。
            file_process_service: 文档导入与处理服务。

        Returns:
            处理结果。

        Raises:
            HTTPException: 输入无效或处理过程无法继续时抛出。
        """
        preparation = file_process_service.process_upload_file(
            file,
            logical_document_id=logical_document_id,
        )
        if not preparation.source_hash:
            raise HTTPException(status_code=500, detail="上传文件保存失败")
        if preparation.error:
            raise HTTPException(status_code=503, detail=preparation.error)

        if not preparation.should_process:
            completed = preparation.duplicate_status == "completed"
            return UploadResponse(
                message=(
                    "文件已存在，无需重复入库"
                    if completed
                    else "相同文件正在入库，请查询原任务状态"
                ),
                task_id=preparation.task_id,
                duplicate=True,
                source_hash=preparation.source_hash,
                version_id=preparation.version_id,
                logical_document_id=preparation.logical_document_id,
                doc_id=preparation.doc_id or None,
            )

        # 2 执行导入的主流程，注册后台任务
        background_tasks.add_task(
            file_process_service.run_main_graph,
            preparation.import_file_path,
            preparation.file_dir,
            preparation.task_id,
            preparation.source_hash,
            preparation.version_id,
            preparation.logical_document_id,
            preparation.resource_objects,
            preparation.checkpoint_state,
        )

        return UploadResponse(
            message="上传成功，已提交入库任务",
            task_id=preparation.task_id,
            duplicate=False,
            source_hash=preparation.source_hash,
            version_id=preparation.version_id,
            logical_document_id=preparation.logical_document_id,
        )

    @router.post("/retry/{task_id}", response_model=UploadResponse)
    def retry_import(
            task_id: str,
            background_tasks: BackgroundTasks,
            file_process_service: FileProcessService = Depends(
                get_file_process_service
            ),
    ):
        """从失败任务最近一次成功断点继续入库。"""
        try:
            preparation = file_process_service.prepare_retry(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"领取重试任务失败: {exc}",
            ) from exc

        background_tasks.add_task(
            file_process_service.run_main_graph,
            preparation.import_file_path,
            preparation.file_dir,
            preparation.task_id,
            preparation.source_hash,
            preparation.version_id,
            preparation.logical_document_id,
            preparation.resource_objects,
            preparation.checkpoint_state,
        )
        return UploadResponse(
            message="已从最近断点重新提交入库任务",
            task_id=preparation.task_id,
            duplicate=True,
            source_hash=preparation.source_hash,
            version_id=preparation.version_id,
            logical_document_id=preparation.logical_document_id,
        )

    @router.get("/status/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(task_id: str):
        # 获取当前任务的信息
        """获取任务状态。

        Args:
            task_id: 异步任务唯一标识。

        Returns:
            处理结果。
        """
        task_info = get_task_info(task_id)

        # 进程重启后内存状态会清空；此时回退到 MongoDB 的持久化
        # 阶段记录，使原 task_id 仍然可以查询失败点和最终结果。
        if not task_info["status"]:
            try:
                record = get_import_record_by_task_id(task_id)
            except Exception as exc:
                logger.warning("从 MongoDB 恢复任务状态失败: %s", exc)
                record = {}
            if record:
                task_info = task_info_from_import_record(record)

        return TaskStatusResponse(**task_info)

router = APIRouter(tags=["文档入库"])
register_router(router)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    Returns:
        处理结果。
    """
    settings = get_settings()
    app = FastAPI(
        title="科研文档知识库入库服务",
        description="文档上传、解析与向量入库 API",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.mount(
        "/front",
        StaticFiles(directory=get_front_page_dir()),
        name="front",
    )

    @app.get("/health", tags=["运行状态"])
    def health() -> dict[str, str]:
        """返回服务健康状态。

        Returns:
            处理结果。
        """
        return {"service": "import", "status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18000)
