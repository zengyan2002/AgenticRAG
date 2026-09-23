import logging

import json
import os
from typing import List,Dict,Any

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.core.settings import get_settings
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.formula_util import FormulaProcessor
from knowledge.utils.document_identity_util import build_chunk_retrieval_text
from knowledge.utils.import_checkpoint_util import save_import_checkpoint


class BGEEmbeddingChunksNode(BaseNode):
    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1 参数校验
        """执行 BGEEmbeddingChunksNode 的核心处理流程。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。

        Raises:
            EmbeddingError: 输入无效或处理过程无法继续时抛出。
        """
        chunks = self._valid_state(state)

        # 2 将chunks向量化
        #2.1 获取嵌入式模型对象
        try:
            embedding_model_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"Failed to initialize the BGE-M3 embedding model: {e}")
            raise EmbeddingError(message=f"Failed to initialize the BGE-M3 embedding model: {e}", node_name=self.name)

        #2.2 从配置中拿去批量处理chunks的数量
        embedding_batch_size = self.config.embedding_batch_size

        # 断点恢复时 chunks 中可能已有部分批次的向量，完整批次直接跳过。
        final_chunks = list(chunks)
        completed_batches = {
            int(value)
            for value in (state.get("embedding_completed_batches") or [])
        }

        total_number_of_chunks = len(chunks)

        #按embedding_batch_size组装chunks
        for index in range(0,total_number_of_chunks,embedding_batch_size):
            batch_number = index // embedding_batch_size
            batch_chunks = final_chunks[index:index+embedding_batch_size]
            already_embedded = all(
                isinstance(chunk.get("dense_vector"), list)
                and bool(chunk.get("dense_vector"))
                and isinstance(chunk.get("sparse_vector"), dict)
                and bool(chunk.get("sparse_vector"))
                for chunk in batch_chunks
            )
            if batch_number in completed_batches and already_embedded:
                continue

            embedded = self._embedding_batch(
                batch_chunks, embedding_model_client
            )
            final_chunks[index:index + len(embedded)] = embedded
            completed_batches.add(batch_number)
            state["chunks"] = final_chunks
            state["embedding_completed_batches"] = sorted(completed_batches)
            if getattr(get_settings(), "import_checkpoint_enabled", False):
                # 批次断点恢复时需要重新进入 Embedding 节点，因此将恢复
                # 位置设为其上游已完成的公式增强节点。
                save_import_checkpoint(
                    state,
                    checkpoint_node=f"{self.name}:batch_{batch_number}",
                    resume_after_node="formula_semantic_node",
                    completed=False,
                )

        if get_settings().import_debug_artifacts:
            self._backup_chunks(final_chunks, state)

        #
        state["chunks"] = final_chunks
        state["embedding_completed_batches"] = sorted(completed_batches)

        return state



    #参数校验
    def _valid_state(self, state: ImportGraphState):
        #1 拿到chunks
        """校验状态。

        Args:
            state: 当前工作流状态。

        Returns:
            处理结果。

        Raises:
            StateFieldError: 输入无效或处理过程无法继续时抛出。
            ValidationError: 输入无效或处理过程无法继续时抛出。
        """
        chunks = state.get("chunks")

        #2 检查chunks是否为空，并且数据类型是否为List
        if not chunks or not isinstance(chunks, list):
            self.logger.error("Invalid chunks parameter")
            raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=list)

        #3 遍历每一个chunk看看是否为dict类型
        for chunk in chunks:
            if not isinstance(chunk, dict):
                self.logger.error("Invalid chunk parameter")
                raise ValidationError(node_name=self.name,message="Each item in chunks must be a dictionary.")

        return chunks

    #利用嵌入模型处理 batch_chunks
    def _embedding_batch(self,batch_chunks, embedding_model_client):

        # 组装嵌入文本。公式除了保留原始 LaTeX，还追加命令含义、
        # 运算类型和变量符号，改善自然语言问题对公式的召回效果。
        """批量生成切片的稠密与稀疏向量。

        Args:
            batch_chunks: 当前批次的文档切片。
            embedding_model_client: 用于生成向量的 Embedding 客户端。

        Returns:
            处理结果。

        Raises:
            EmbeddingError: 输入无效或处理过程无法继续时抛出。
        """
        texts = [self._build_embedding_text(chunk) for chunk in batch_chunks]

        #调用嵌入模型
        try:
            result = embedding_model_client.encode(texts,return_dense=True,return_sparse=True)
        except Exception as e:
            message = f"Failed to embed a batch of chunks: {e}"
            self.logger.exception(message)
            raise EmbeddingError(message=message,node_name=self.name,cause=e)

        #获取稠密向量和稀疏向量
        dense_vectors = result["dense_vecs"]
        sparse_vectors = result["lexical_weights"]

        #声明包含稀疏向量和稠密向量的新的chunks
        embedded_chunks = []

        for chunk,dense_vector,sparse_vector in zip(batch_chunks, dense_vectors, sparse_vectors):
            chunk["dense_vector"] = dense_vector.tolist()
            chunk["sparse_vector"] =  {
                int(token_id): float(weight)
                for token_id, weight in dict(sparse_vector).items()}

            embedded_chunks.append(chunk)

        return embedded_chunks

    def _build_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """构建向量表示文本。

        Args:
            chunk: 待处理的文档切片。

        Returns:
            处理后的字符串。
        """
        content = str(chunk.get("content") or "")
        formula_search_text = str(chunk.get("formula_search_text") or "").strip()

        # 兼容旧切片文件：如果导入数据还没有公式元数据，则在向量化前补齐。
        if not formula_search_text:
            formulas = FormulaProcessor.extract(content)
            if formulas:
                chunk["has_formula"] = True
                chunk["formulas"] = formulas
                formula_search_text = FormulaProcessor.build_search_text(formulas)
                chunk["formula_search_text"] = formula_search_text

        retrieval_text = build_chunk_retrieval_text(chunk)
        chunk["retrieval_text"] = retrieval_text
        return retrieval_text or content

    def _backup_chunks(self, chunks:List[Dict[str,Any]], state:ImportGraphState):
        #保存的文件目录
        """将切片中间结果备份到本地文件。

        Args:
            chunks: 待处理的文档切片列表。
            state: 当前工作流状态。

        Returns:
            处理结果。
        """
        file_dir = state.get("file_dir")
        if not file_dir:
            return

        #创建目录
        os.makedirs(file_dir, exist_ok=True)
        try:
            #文件路径
            backup_file_path = os.path.join(file_dir, "chunks_embedding.json")

            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks,f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to back up the embedded chunks; "
                f"the import process will continue: {e}")

