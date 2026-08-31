"""查询相关 Schema 定义"""

from typing import Optional, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")
    error:str = Field("",description="错误信息")
    done_list:List[str] = Field(default_factory=list,description="已经处理完成的节点列表")
    image_urls:List[str] = Field(default_factory=list,description="答案关联的图片地址")


class StreamSubmitResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID，前端用此 ID 建立 SSE 连接")


class HistoryItem(BaseModel):
    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    theme_names: List[str] = Field(default_factory=list)
    image_urls: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]


class SessionCreateRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="前端生成的会话ID")
    owner_id: str = Field(..., min_length=1, description="用户ID或浏览器客户端ID")
    title: str = Field("新会话", max_length=100, description="会话标题")


class SessionRenameRequest(BaseModel):
    owner_id: str = Field(..., min_length=1, description="用户ID或浏览器客户端ID")
    title: str = Field(..., min_length=1, max_length=100, description="新会话标题")


class SessionItem(BaseModel):
    session_id: str
    title: str = "新会话"
    owner_id: str
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    last_message: str = ""


class SessionListResponse(BaseModel):
    items: List[SessionItem] = Field(default_factory=list)
