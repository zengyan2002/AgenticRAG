"""在切片成功入库后，将文档档案 upsert 到独立注册表。"""

from knowledge.processor.import_processor.nodes.document_identity_node import (
    DocumentIdentityNode,
)
from knowledge.processor.import_processor.state import ImportGraphState


class DocumentRegistryNode(DocumentIdentityNode):
    name = "document_registry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """执行 DocumentRegistryNode 的核心处理流程。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。

        Raises:
            ValueError: 输入无效或处理过程无法继续时抛出。
        """
        identity = state.get("document_identity") or {}
        profile_text = str(identity.get("profile_text") or "").strip()
        if not identity.get("doc_id") or not profile_text:
            raise ValueError("文档档案不完整，无法写入注册表")

        dense_vector, sparse_vector = self._embed_profile(profile_text)
        self._upsert_document_profile(identity, dense_vector, sparse_vector)
        return state

