"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from knowledge.core.settings import get_settings

_app_settings = get_settings()


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    theme_name_chunk_k: int = 3  # 识别文档主题的名时最多使用几个切片
    theme_name_chunk_size: int = 2500  # 文档主题名识别时使用的切片内容长度
    document_title_review_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("DOCUMENT_TITLE_REVIEW_THRESHOLD", "0.65")
        )
    )

    vl_model: str = field(
        default_factory=lambda: _app_settings.vl_model
    )

    # ==================== Milvus 配置 ====================
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    bm25_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "BM25_ENABLED", "false"
        ).lower() in ("true", "1", "yes")
    )
    bm25_analyzer_type: str = field(
        default_factory=lambda: os.getenv("BM25_ANALYZER_TYPE", "chinese")
    )
    document_registry_collection: str = field(
        default_factory=lambda: (
            os.getenv("DOCUMENT_REGISTRY_COLLECTION")
            or "kb_document_registry_v1"
        )
    )
    # ==================== Mineru 配置 ====================
    mineru_api_token: str = field(
        default_factory=lambda: _app_settings.mineru_api_token
    )

    mineru_base_url: str = field(
        default_factory=lambda: _app_settings.mineru_base_url
    )


    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = field(
        default_factory=lambda: _app_settings.minio_secure
    )

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    # ==================== 公式语义增强 ====================
    formula_semantic_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "FORMULA_SEMANTIC_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
    )

    # Word 文档先转成 PDF，再进入 MinerU。留空时自动发现 LibreOffice，
    # Windows 环境下可回退到本机 Microsoft Word。
    libreoffice_path: str = field(
        default_factory=lambda: os.getenv("LIBREOFFICE_PATH", "")
    )
    word_convert_timeout_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("WORD_CONVERT_TIMEOUT_SECONDS", "120")
        )
    )
    formula_semantic_max_formulas: int = field(
        default_factory=lambda: int(
            os.getenv("FORMULA_SEMANTIC_MAX_FORMULAS", "10")
        )
    )
    formula_semantic_batch_size: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("FORMULA_SEMANTIC_BATCH_SIZE", "20"))
        )
    )
    formula_semantic_max_workers: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("FORMULA_SEMANTIC_MAX_WORKERS", "3"))
        )
    )
    formula_semantic_context_chars_per_formula: int = field(
        default_factory=lambda: max(
            100,
            int(
                os.getenv(
                    "FORMULA_SEMANTIC_CONTEXT_CHARS_PER_FORMULA",
                    "600",
                )
            ),
        )
    )

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
