from functools import cache

from knowledge.service.file_process_service import FileProcessService
from knowledge.service.query_service import QueryService

@cache
def get_file_process_service() -> FileProcessService:
    """获取文件processservice。

    Returns:
        处理结果。
    """
    return FileProcessService()

@cache
def get_query_service() -> QueryService:
    """获取查询service。

    Returns:
        处理结果。
    """
    return QueryService()
