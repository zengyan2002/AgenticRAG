import re
from io import BytesIO

from PIL import Image

from knowledge.processor.query_processor.main_graph import query_app
import logging

from knowledge.schema.query_schema import HistoryItem
from knowledge.utils.mongo_history_util import clear_history as clear_chat_history, get_recent_messages
from knowledge.utils.mongo_session_util import (
    clear_chat_session_preview,
    create_chat_session,
    delete_chat_session,
    list_chat_sessions,
    rename_chat_session,
    touch_chat_session,
)
from knowledge.utils.task_util import get_task_result, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
from knowledge.processor.query_processor.state import create_default_state

logger = logging.getLogger(__name__)


class QueryService:
    _IMAGE_FORMAT_TO_MIME = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
    }

    def validate_query_image(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("上传的图片为空")

        max_bytes = self._get_max_query_image_bytes()
        if len(image_bytes) > max_bytes:
            raise ValueError(f"图片大小不能超过 {max_bytes // (1024 * 1024)} MB")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = (image.format or "").upper()
                image.verify()
        except Exception as exc:
            raise ValueError("上传文件不是有效图片或图片已经损坏") from exc

        mime_type = self._IMAGE_FORMAT_TO_MIME.get(image_format)
        if not mime_type:
            raise ValueError("仅支持 JPEG、PNG、WebP、GIF 和 BMP 图片")
        return mime_type

    @staticmethod
    def _get_max_query_image_bytes() -> int:
        from knowledge.processor.query_processor.config import get_config

        return get_config().max_query_image_bytes

    #提取文字中图片的url
    def _extract_image_urls(self,text:str):
        if not text:
            return []

        pattern = re.compile(
            r'https?://[^\s<>"\]\)]+'
            r'\.(?:png|jpe?g|gif|webp|bmp|svg)'
            r'(?:\?[^\s<>"\]\)]*)?',
            re.IGNORECASE,
        )

        return list(dict.fromkeys(pattern.findall(text)))

    # 执行完整的答案查询流程
    def run_query_graph(
            self,
            query: str,
            session_id: str,
            task_id: str,
            is_stream: bool,
            image_bytes: bytes = b"",
            image_mime_type: str = "",
            display_query: str = "",
    ):
        # 创建state
        state = create_default_state(
            session_id=session_id,
            task_id=task_id,
            original_query=query,
            display_query=display_query or query,
            is_stream=is_stream,
            query_image_bytes=image_bytes,
            query_image_mime_type=image_mime_type,
        )


        try:
            #设置整个检索流程的状态为正在进行
            update_task_status(task_id,TASK_STATUS_PROCESSING)

            result = query_app.invoke(state)

            #设置整个检索流程的状态为已经完成
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"Failed to run query graph,{e}")
            #设置整个检索流程的状态为失败
            update_task_status(task_id, TASK_STATUS_FAILED)

            raise e

    #获取答案
    def get_query_result(self,task_id):
        return get_task_result(task_id,"answer")

    # 清空历史会话内容
    def clear_history(self, session_id):
        clear_num = clear_chat_history(session_id)
        clear_chat_session_preview(session_id)
        return clear_num

    # 创建新会话
    def create_session(self, session_id: str, owner_id: str, title: str = "新会话"):
        return create_chat_session(
            session_id=session_id,
            owner_id=owner_id,
            title=title,
        )

    def get_sessions(self, owner_id: str, limit: int = 50):
        return list_chat_sessions(owner_id=owner_id, limit=limit)

    def touch_session(self, session_id: str, message: str):
        return touch_chat_session(session_id=session_id, message=message)

    def rename_session(self, session_id: str, owner_id: str, title: str):
        return rename_chat_session(
            session_id=session_id,
            owner_id=owner_id,
            title=title,
        )

    def delete_session(self, session_id: str, owner_id: str):
        deleted_count = delete_chat_session(session_id=session_id, owner_id=owner_id)
        deleted_messages = 0
        if deleted_count:
            deleted_messages = clear_chat_history(session_id)
        return {
            "deleted_count": deleted_count,
            "deleted_messages": deleted_messages,
        }

    #查询历史会话内容  返回items
    def get_history(self, session_id,limit:int=50):
        #通过session_id得到历史会话列表
        """
        [
            {
                "_id": ObjectId("696b00000000000000000002"),
                "session_id": "sess-001",
                "role": "assistant",
                "text": "请将功能转盘调整到电压档位……",
                "rewritten_query": "",
                "theme_names": ["万用表RS-12"],
                "ts": 1784253225.5
            },
            {
                "_id": ObjectId("696b00000000000000000001"),
                "session_id": "sess-001",
                "role": "user",
                "text": "万用表RS-12怎么测量电压？",
                "rewritten_query": "万用表RS-12如何测量电压？",
                "theme_names": ["万用表RS-12"],
                "ts": 1784253217.5
            }
        ]
        :param session_id:
        :return:
        """
        history_list = get_recent_messages(
            session_id=session_id,
            limit=limit,
        )
        history_list.reverse()
        items = []
        for history in history_list:
            role = history.get("role", "")

            # 只向前端返回正常的用户和助手消息
            if role not in ("user", "assistant"):
                continue

            items.append(
                HistoryItem(
                    session_id=session_id,
                    role=role,
                    text=history.get("text", ""),
                    ts=history.get("ts"),
                    image_urls=self._extract_image_urls(history.get("text", "")),
                )
            )

        return items





