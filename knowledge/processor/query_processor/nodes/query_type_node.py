from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class QueryTypeNode(BaseNode):
    """Classify the request before routing text and image inputs."""

    name = "query_type_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = (state.get("original_query") or "").strip()
        has_text = bool(original_query)
        has_image = bool(state.get("query_image_bytes"))

        if has_text and has_image:
            state["query_type"] = "multimodal"
        elif has_image:
            state["query_type"] = "image"
        elif has_text:
            state["query_type"] = "text"
            state["retrieval_query"] = original_query
        else:
            raise ValueError("查询内容不能为空，请输入文字或上传图片")

        return state
