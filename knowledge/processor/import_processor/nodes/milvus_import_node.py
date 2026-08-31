import math
import hashlib
from typing import Any
from pymilvus import DataType, Function, FunctionType
import logging

import json

from pymilvus import MilvusClient

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import ValidationError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.storage_clients import StorageClients
from numbers import Real

class _MilvusSchemaBuilder:
    @classmethod
    def build_schema(
            cls,
            client: MilvusClient,
            dim: int,
            enable_bm25: bool = False,
            analyzer_type: str = "chinese",
    ):
        #1 创建一个schema
        schema = client.create_schema(enable_dynamic_field=True)

        #2 添加字段
        #标量
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            auto_id=not enable_bm25,
            is_primary=True
        )
        schema.add_field(
            field_name="theme_name",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="canonical_title",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="primary_subject",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="document_type",
            datatype=DataType.VARCHAR,
            max_length=255,
        )
        schema.add_field(
            field_name="document_summary",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="aliases_json",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="model_codes_json",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="section_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="section_path",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="parent_summary",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="section_chunk_index",
            datatype=DataType.INT64,
        )
        #向量
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        if enable_bm25:
            schema.add_field(
                field_name="retrieval_text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": analyzer_type},
            )
            schema.add_field(
                field_name="bm25_sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_function(Function(
                name="retrieval_text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["retrieval_text"],
                output_field_names=["bm25_sparse_vector"],
            ))
        return schema

class _MilvusIndexBuilder:
    @classmethod
    def build_index(cls, client: MilvusClient, enable_bm25: bool = False):
        index_params = client.prepare_index_params()
        index_params.add_index(
            index_name="dense_vector_index",
            field_name="dense_vector",  # 建立索引的字段
            index_type="AUTOINDEX",  # 索引类型
            metric_type="COSINE",  # 向量相似度度量方式
        )
        index_params.add_index(
            index_name="sparse_vector_index",
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        if enable_bm25:
            index_params.add_index(
                index_name="bm25_sparse_vector_index",
                field_name="bm25_sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        return index_params

class MilvusImportNode(BaseNode):
    name = "milvus_import_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        #1 参数校验  输出
        validated_chunks,expected_dim = self._validate_inputs(state)

        #2 向量入库
        validated_chunks = self._insert_chunks_to_milvus(validated_chunks,expected_dim)

        #3 用validated_chunks更新state中的chunk
        state["chunks"] = validated_chunks
        
        return state


    def _validate_inputs(self, state):
        self.log_step("validate", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("The chunks field must be a list", self.name)

        
        expected_dim = self.config.embedding_dim

        validated_chunks: list[dict[str, Any]] = []

        skipped_count = 0

        for i, chunk in enumerate(chunks):
            # 类型不对 → 抛异常（和上游 embedding 节点保持一致）
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunks[{i}] must be a dictionary", self.name
                )

            #单个chunk的正文无效时先跳过
            content = chunk.get("content")
            if not isinstance(content, str) or not content.strip():
                skipped_count+=1
                self.logger.warning(f"Skipping chunks[{i}] because its content is empty or invalid.")
                continue

            # 缺少向量字段 → 跳过（嵌入可能部分失败，属于数据级容错）
            dense_vector = chunk.get("dense_vector")
            sparse_vector = chunk.get("sparse_vector")
            # 缺少稠密向量
            if not isinstance(dense_vector, list) or not dense_vector:
                skipped_count += 1
                self.logger.warning(f"Skipping chunks[{i}] because its dense_vector is empty or invalid.")
                continue

            # 缺少稀疏向量
            if not isinstance(sparse_vector, dict) or not sparse_vector:
                skipped_count += 1
                self.logger.warning(f"Skipping chunks[{i}] because its sparse_vector is empty or invalid.")
                continue

            # 稠密向量的维度错误，直接抛错
            actual_dim = len(dense_vector)
            if actual_dim != expected_dim:
                raise ValidationError(
                    message=(
                        f"chunks[{i}].dense_vector has an invalid "
                        f"dimension: expected {expected_dim}, "
                        f"got {actual_dim}."
                    ),
                    node_name=self.name,
                )
            # 稠密向量的数据类型错误，直接抛错
            dense_vector_is_valid = all(isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in dense_vector
            )
            if not dense_vector_is_valid:
                raise ValidationError(
                    message=(
                        f"chunks[{i}].dense_vector contains "
                        f"invalid numeric values."
                    ),
                    node_name=self.name,
                )

            # 检查稀疏向量的数据类型
            for token_id, weight in sparse_vector.items():
                if (
                    not isinstance(token_id, int)
                    or isinstance(token_id, bool)
                    or token_id < 0
                ):
                    raise ValidationError(
                        message=(
                            f"chunks[{i}].sparse_vector contains "
                            f"an invalid token ID: {token_id!r}."
                        ),
                        node_name=self.name,
                    )

                if not isinstance(weight, Real) or isinstance(weight, bool) or not math.isfinite(float(weight)):
                    raise ValidationError(
                        message=(
                            f"chunks[{i}].sparse_vector contains "
                            f"an invalid weight for token {token_id}."
                        ),
                        node_name=self.name,
                    )

            validated_chunks.append(chunk)

        # 没有可用的chunks直接报错
        if not validated_chunks:
            raise ValidationError(
                message="No valid chunks are available for insertion into Milvus.",
                node_name=self.name,
            )

        self.logger.info(
            f"Chunk validation completed: valid={len(validated_chunks)}, "
            f"skipped={skipped_count}, dimension={expected_dim}."
        )

        return validated_chunks,expected_dim

    def _insert_chunks_to_milvus(self,validated_chunks,expected_dim):

        try:
            # 1创建Milvus客户端
            milvus_client = StorageClients.get_milvus()
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to the Milvus server. Reason: {e}")
            raise MilvusError(message=f"Failed to connect to the Milvus server. Reason: {e}", node_name=self.name)

        # 2从配置中获取存储chunk的Collection的name
        chunks_collection = self.config.chunks_collection

        # 3判断上述Collection是否存在，如果不存在则创建Collection
        if milvus_client.has_collection(chunks_collection):
            # 已经有相同名的collection了，不用创建了
            self.logger.info(f"Chunks collection {chunks_collection} already exists.")
        else:
            # 3.1创建schema
            schema = _MilvusSchemaBuilder.build_schema(
                milvus_client,
                expected_dim,
                enable_bm25=self.config.bm25_enabled,
                analyzer_type=self.config.bm25_analyzer_type,
            )

            # 3.2创建index
            index = _MilvusIndexBuilder.build_index(
                milvus_client,
                enable_bm25=self.config.bm25_enabled,
            )

            # 3.3创建Collection
            collection = milvus_client.create_collection(collection_name=chunks_collection,
                                                         schema=schema,
                                                         index_params=index)

        try:
            # 4 插入数据，并且获取自增长的id
            if self.config.bm25_enabled:
                for position, chunk in enumerate(validated_chunks):
                    chunk.setdefault("id", self._stable_chunk_id(chunk, position))
            result = milvus_client.insert(collection_name=chunks_collection,data=validated_chunks)
            generated_ids = result.get("ids", [])
        except Exception as e:
            self.logger.error(f"Failed to insert chunks into Milvus. Reason: {e}")
            raise MilvusError(message=f"Failed to insert chunks into Milvus. Reason: {e}",
                              node_name=self.name)

        # 5 将增长的id回填到chunks中
        if len(generated_ids) != len(validated_chunks):
            raise MilvusError(
                message=(
                    f"The number of generated IDs does not match "
                    f"the number of inserted chunks: "
                    f"ids={len(generated_ids)}, "
                    f"chunks={len(validated_chunks)}."
                ),
                node_name=self.name,
            )

        for chunk, chunk_id in zip(
                validated_chunks,
                generated_ids,
        ):
            chunk["chunk_id"] = int(chunk_id)

        return validated_chunks

    @staticmethod
    def _stable_chunk_id(chunk: dict[str, Any], position: int) -> int:
        raw = "\x1f".join([
            str(chunk.get("doc_id") or ""),
            str(chunk.get("section_id") or ""),
            str(chunk.get("chunk_index") if chunk.get("chunk_index") is not None else position),
            str(chunk.get("content") or "")[:200],
        ]).encode("utf-8")
        # 保持在Milvus有符号INT64正数范围内。
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)
