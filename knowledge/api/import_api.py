import uvicorn

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
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
from knowledge.utils.task_util import get_task_info


def register_router(router: APIRouter) -> None:
    @router.post("/upload", response_model=UploadResponse)
    def upload_file(background_tasks: BackgroundTasks,
                    file: UploadFile,
                    file_process_service: FileProcessService = Depends(get_file_process_service)):
        # 1 对上传的文件进行处理（保存）
        import_file_path_obj, file_dir, task_id = file_process_service.process_upload_file(file)
        if import_file_path_obj is None or file_dir is None:
            raise HTTPException(status_code=500, detail="上传文件保存失败")

        # 2 执行导入的主流程，注册后台任务
        background_tasks.add_task(file_process_service.run_main_graph,import_file_path_obj,file_dir,task_id)

        return UploadResponse(message="上传成功", task_id=task_id)

    @router.get("/status/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(task_id: str):
        # 获取当前任务的信息
        task_info = get_task_info(task_id)

        return TaskStatusResponse(**task_info)

router = APIRouter(tags=["文档入库"])
register_router(router)


def create_app() -> FastAPI:
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
        return {"service": "import", "status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18000)
