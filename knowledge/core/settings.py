"""应用级环境配置。

本模块是项目唯一的 ``.env`` 加载入口。客户端和 API 只从
``get_settings()`` 读取基础设施配置，入库/查询算法参数仍分别由各自的
Config 管理。
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values, load_dotenv

from knowledge.core.paths import KNOWLEDGE_ROOT


ENV_FILE = Path(KNOWLEDGE_ROOT) / ".env"
load_dotenv(ENV_FILE)
_DOTENV_VALUES = dotenv_values(ENV_FILE)


def dotenv_value(name: str) -> str:
    """只读取项目 ``.env``，不回退到操作系统同名变量。"""
    value = _DOTENV_VALUES.get(name)
    return str(value).strip() if value is not None else ""


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: Iterable[str] = ()) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """跨流程共享的基础设施与部署配置。"""

    dashscope_api_base: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_BASE", "")
    )
    dashscope_api_key: str = field(
        default_factory=lambda: dotenv_value("DASHSCOPE_API_KEY")
    )
    llm_default_model: str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "")
    )
    llm_fast_model: str = field(
        default_factory=lambda: os.getenv("LLM_FAST_MODEL", "")
    )
    llm_agent_model: str = field(
        default_factory=lambda: os.getenv("LLM_AGENT_MODEL", "")
    )
    llm_answer_model: str = field(
        default_factory=lambda: os.getenv("LLM_ANSWER_MODEL", "")
    )
    llm_light_thinking_budget: int = field(
        default_factory=lambda: int(os.getenv("LLM_LIGHT_THINKING_BUDGET", "4096"))
    )
    llm_deep_thinking_budget: int = field(
        default_factory=lambda: int(os.getenv("LLM_DEEP_THINKING_BUDGET", "16384"))
    )
    vl_model: str = field(default_factory=lambda: os.getenv("VL_MODEL", ""))
    mineru_api_token: str = field(
        default_factory=lambda: dotenv_value("MINERU_API_TOKEN")
    )
    mineru_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MINERU_BASE_URL",
            "https://mineru.net/api/v4",
        )
    )

    bge_m3_path: str = field(
        default_factory=lambda: os.getenv("BGE_M3_PATH", "")
    )
    bge_device: str = field(
        default_factory=lambda: os.getenv("BGE_DEVICE", "cpu")
    )
    bge_fp16: bool = field(
        default_factory=lambda: env_bool("BGE_FP16", False)
    )
    bge_reranker_path: str = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_LARGE", "")
    )
    bge_reranker_device: str = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_DEVICE", "cpu")
    )
    bge_reranker_fp16: bool = field(
        default_factory=lambda: env_bool("BGE_RERANKER_FP16", False)
    )

    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket_name: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = field(
        default_factory=lambda: env_bool("MINIO_SECURE", False)
    )
    mongo_url: str = field(
        default_factory=lambda: os.getenv("MONGO_URL", "")
    )
    mongo_db_name: str = field(
        default_factory=lambda: os.getenv("MONGO_DB_NAME", "")
    )

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: env_list("CORS_ORIGINS", ("*",))
    )

    import_debug_artifacts: bool = field(
        default_factory=lambda: env_bool("IMPORT_DEBUG_ARTIFACTS", False)
    )
    keep_import_artifacts: bool = field(
        default_factory=lambda: env_bool("KEEP_IMPORT_ARTIFACTS", False)
    )
    keep_failed_artifacts: bool = field(
        default_factory=lambda: env_bool("KEEP_FAILED_ARTIFACTS", True)
    )

    def require(self, *attribute_names: str) -> tuple[str, ...]:
        """返回必填配置；缺失时在启动/首次使用阶段给出明确错误。"""
        values: list[str] = []
        missing: list[str] = []
        for attribute_name in attribute_names:
            value = getattr(self, attribute_name)
            if value in (None, ""):
                missing.append(attribute_name.upper())
            else:
                values.append(str(value))
        if missing:
            raise EnvironmentError(
                "缺少必需的环境配置: " + ", ".join(missing)
            )
        return tuple(values)

    def llm_model_for_role(self, role: str) -> str:
        """按调用职责选择模型，并兼容旧的单模型配置。"""
        role_mapping = {
            "default": self.llm_default_model,
            "fast": self.llm_fast_model,
            "agent": self.llm_agent_model,
            "answer": self.llm_answer_model,
        }
        if role not in role_mapping:
            raise ValueError(f"不支持的 LLM role: {role}")
        return role_mapping[role] or self.llm_default_model


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
