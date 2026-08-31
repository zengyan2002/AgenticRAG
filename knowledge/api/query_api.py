import asyncio
from contextlib import asynccontextmanager
from typing import Union

import uuid
import uvicorn
from fastapi import (
    APIRouter,
    Depends,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from knowledge.api.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.core.settings import get_settings
from knowledge.schema.query_schema import (
    HistoryResponse,
    QueryRequest,
    QueryResponse,
    SessionCreateRequest,
    SessionItem,
    SessionListResponse,
    SessionRenameRequest,
    StreamSubmitResponse,
)

from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.mongo_session_util import ensure_session_indexes
from knowledge.utils.task_util import get_task_result, get_done_task_list



def register_router(router: APIRouter) -> None:
    @router.post("/query",response_model=Union[QueryResponse,StreamSubmitResponse])
    async def query(request: QueryRequest,
                    background_tasks: BackgroundTasks,
                    query_service: QueryService = Depends(get_query_service))-> Union[QueryResponse,StreamSubmitResponse]:
        #获取session_id、task_id、query、is_stream
        session_id = request.session_id or f"sess-{uuid.uuid4().hex}"
        task_id = str(uuid.uuid4().hex[:12])
        user_query = request.query
        is_stream = request.is_stream

        # 更新左侧会话目录中的标题、摘要和最近使用时间。
        await asyncio.to_thread(
            query_service.touch_session,
            session_id=session_id,
            message=user_query,
        )

        # 调用service
        # 如果是流式调用
        if is_stream:

            #创建该任务对应的SSE队列
            create_sse_queue(task_id)

            #必须在另一个线程中异步执行
            background_tasks.add_task(query_service.run_query_graph,user_query,session_id,task_id,is_stream)

            #将task_id和session_id返回给前端
            return StreamSubmitResponse(message="查询请求已经提交",session_id=session_id,task_id=task_id)
        else:
            await asyncio.to_thread(
                query_service.run_query_graph,
                user_query, session_id, task_id, is_stream
            )
            #获取已完成的节点列表
            done_task_list = get_done_task_list(task_id)
            return QueryResponse(message="查询请求已经处理完毕",
                                 session_id=session_id,
                                 answer=query_service.get_query_result(task_id),
                                 done_list=done_task_list)

    @router.post(
        "/query/image",
        response_model=Union[QueryResponse, StreamSubmitResponse],
    )
    async def query_by_image(
            background_tasks: BackgroundTasks,
            image: UploadFile = File(...),
            query: str = Form(""),
            session_id: str | None = Form(None),
            is_stream: bool = Form(False),
            query_service: QueryService = Depends(get_query_service),
    ) -> Union[QueryResponse, StreamSubmitResponse]:
        safe_session_id = session_id or f"sess-{uuid.uuid4().hex}"
        task_id = uuid.uuid4().hex[:12]
        user_query = query.strip()
        display_query = user_query or "根据这张图片检索相关内容"

        max_bytes = query_service._get_max_query_image_bytes()
        image_bytes = await image.read(max_bytes + 1)
        try:
            image_mime_type = query_service.validate_query_image(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.to_thread(
            query_service.touch_session,
            session_id=safe_session_id,
            message=f"[图片检索] {display_query}",
        )

        graph_args = (
            user_query,
            safe_session_id,
            task_id,
            is_stream,
            image_bytes,
            image_mime_type,
            display_query,
        )
        if is_stream:
            create_sse_queue(task_id)
            background_tasks.add_task(query_service.run_query_graph, *graph_args)
            return StreamSubmitResponse(
                message="图片查询请求已经提交",
                session_id=safe_session_id,
                task_id=task_id,
            )

        await asyncio.to_thread(query_service.run_query_graph, *graph_args)
        return QueryResponse(
            message="图片查询请求已经处理完毕",
            session_id=safe_session_id,
            answer=query_service.get_query_result(task_id),
            done_list=get_done_task_list(task_id),
        )

    @router.get("/stream/{task_id}")
    async def stream(task_id:str,request:Request):
        #获取流式调用接口
        return StreamingResponse(content = sse_generator(task_id,request),
                                 media_type="text/event-stream")

    @router.delete("/history/{session_id}")
    def clear_history(session_id:str,
                      query_service: QueryService = Depends(get_query_service)):
        clear_num = query_service.clear_history(session_id)
        return {
            "message": "历史会话已清空",
            "session_id": session_id,
            "deleted_count": clear_num,
        }

    @router.get("/history/{session_id}",response_model=HistoryResponse)
    def get_history(session_id:str,
                  query_service: QueryService = Depends(get_query_service)):
        history_list = query_service.get_history(session_id)
        return HistoryResponse(
            session_id=session_id,
            items=history_list,
        )

    @router.post("/sessions", response_model=SessionItem)
    def add_session(
            request: SessionCreateRequest,
            query_service: QueryService = Depends(get_query_service),
    ):
        try:
            return query_service.create_session(
                session_id=request.session_id,
                owner_id=request.owner_id,
                title=request.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/sessions", response_model=SessionListResponse)
    def get_sessions(
            owner_id: str,
            limit: int = 50,
            query_service: QueryService = Depends(get_query_service),
    ):
        safe_limit = max(1, min(limit, 200))
        return SessionListResponse(
            items=query_service.get_sessions(owner_id=owner_id, limit=safe_limit),
        )

    @router.patch("/sessions/{session_id}", response_model=SessionItem)
    def rename_session(
            session_id: str,
            request: SessionRenameRequest,
            query_service: QueryService = Depends(get_query_service),
    ):
        try:
            session = query_service.rename_session(
                session_id=session_id,
                owner_id=request.owner_id,
                title=request.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        return session

    @router.delete("/sessions/{session_id}")
    def delete_session(
            session_id: str,
            owner_id: str,
            query_service: QueryService = Depends(get_query_service),
    ):
        result = query_service.delete_session(
            session_id=session_id,
            owner_id=owner_id,
        )
        if not result["deleted_count"]:
            raise HTTPException(status_code=404, detail="会话不存在")

        return {
            "message": "会话已删除",
            "session_id": session_id,
            **result,
        }
router = APIRouter(tags=["知识查询"])
register_router(router)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_session_indexes()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="科研文档知识库查询服务",
        description="文档检索、问答和会话管理 API",
        version="1.0.0",
        lifespan=lifespan,
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
        return {"service": "query", "status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18001)

