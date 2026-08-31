import threading
from typing import Optional
import logging
from pymongo import MongoClient
from pymongo.database import Database
logger = logging.getLogger(__name__)


from minio import Minio
from pymilvus import MilvusClient
from knowledge.core.settings import get_settings
from knowledge.utils.clients.base import BaseClientManager


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus、MongoDB"""

    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    _mongo_db: Optional[Database] = None
    _mongo_lock = threading.Lock()

    # ── MinIO ──

    @classmethod
    def get_minio(cls) -> Minio:
        return cls._get_or_create("_minio_client", cls._minio_lock, cls._create_minio)

    @classmethod
    def _create_minio(cls) -> Minio:
        try:
            settings = get_settings()
            endpoint, access_key, secret_key, bucket_name = settings.require(
                "minio_endpoint",
                "minio_access_key",
                "minio_secret_key",
                "minio_bucket_name",
            )

            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=settings.minio_secure,
            )

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' 已存在")

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e

    # ── Milvus ──

    @classmethod
    def get_milvus(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus)

    @classmethod
    def _create_milvus(cls) -> MilvusClient:
        try:
            uri = get_settings().require("milvus_url")[0]
            client = MilvusClient(uri=uri)
            logger.info(f"Milvus 客户端初始化成功 (uri={uri})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Milvus 客户端创建失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}") from e

    # ── MongoDB ──

    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create(
            "_mongo_db",
            cls._mongo_lock,
            cls._create_mongo_db,
        )

    @classmethod
    def _create_mongo_db(cls) -> Database:
        try:
            mongo_url, database_name = get_settings().require(
                "mongo_url",
                "mongo_db_name",
            )

            client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
            )

            # MongoClient默认延迟连接，使用ping提前验证连接
            client.admin.command("ping")

            database = client[database_name]

            logger.info(
                "MongoDB客户端初始化成功 "
                f"(database={database_name})"
            )
            return database

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MongoDB客户端创建失败: {e}")
            raise ConnectionError(f"MongoDB连接失败: {e}") from e
