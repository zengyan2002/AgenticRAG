import base64

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import (
    IMAGE_QUERY_SYSTEM_PROMPT,
    IMAGE_QUERY_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients


class ImageQueryNode(BaseNode):
    """Turn an uploaded image into a semantic query for the text RAG pipeline."""

    name = "image_query_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = (state.get("original_query") or "").strip()
        image_bytes = state.get("query_image_bytes") or b""

        if not image_bytes:
            state["retrieval_query"] = original_query
            return state

        mime_type = state.get("query_image_mime_type") or "image/jpeg"
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        user_prompt = IMAGE_QUERY_USER_PROMPT_TEMPLATE.format(
            query=original_query or "检索与图片内容相关的科研资料",
        )
        messages = [
            {"role": "system", "content": IMAGE_QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    },
                ],
            },
        ]

        vlm_client = AIClients.get_vlm_client()
        response = vlm_client.chat.completions.create(
            model=self.config.vl_model,
            messages=messages,
            temperature=0,
        )
        description = (response.choices[0].message.content or "").strip()
        if not description:
            raise ValueError("视觉模型未返回有效的图片检索描述")

        description = description[:4000]
        state["image_query_description"] = description
        state["retrieval_query"] = (
            f"用户检索要求：{original_query}\n图片语义描述：{description}"
            if original_query
            else f"图片语义描述：{description}"
        )
        # 图片完成语义化后不再需要在后续节点间传递原始二进制。
        state["query_image_bytes"] = b""
        return state
