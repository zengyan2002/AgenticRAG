from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """文件上传响应 —— POST /upload 返回"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")
    duplicate: bool = Field(False, description="是否命中相同文件的入库记录")
    source_hash: str = Field("", description="原始文件的 SHA-256")
    version_id: str = Field("", description="由原始文件哈希生成的版本ID")
    logical_document_id: str = Field("", description="跨内容版本稳定的逻辑文档ID")
    doc_id: Optional[str] = Field(None, description="已入库文档ID")


class TaskStatusResponse(BaseModel):
    """任务状态响应 —— GET /status/{task_id} 返回"""
    status: str = Field(..., description="任务状态")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    durations: Dict[str, float] = Field(
        default_factory=dict,
        description="各节点耗时(秒)",
    )
    result: Dict[str, str] = Field(
        default_factory=dict,
        description="任务产生的文档ID、文件哈希或错误信息",
    )
