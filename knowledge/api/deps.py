from functools import cache

from knowledge.service.file_process_service import FileProcessService
from knowledge.service.query_service import QueryService

@cache
def get_file_process_service() -> FileProcessService:
    return FileProcessService()

@cache
def get_query_service() -> QueryService:
    return QueryService()
