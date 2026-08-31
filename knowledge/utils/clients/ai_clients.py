import threading
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple
import os

# 本项目中的BGE模型使用PyTorch，避免Transformers误加载不兼容的Keras 3后端。
os.environ.setdefault("USE_TF", "0")

from langchain_openai import ChatOpenAI
from openai import OpenAI
from knowledge.core.settings import get_settings
from knowledge.utils.clients.base import BaseClientManager, logger
from FlagEmbedding import BGEM3FlagModel, FlagReranker


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _llm_clients: Dict[Tuple[str, bool, str], ChatOpenAI] = {}
    _llm_clients_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3FlagModel] = None
    _bge_m3_lock = threading.Lock()

    _bge_reranker_client: Optional[FlagReranker] = None
    _bge_reranker_lock = threading.Lock()

    # ── VLM ──

    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            settings = get_settings()
            base_url, api_key = settings.require(
                "dashscope_api_base",
                "dashscope_api_key",
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("DashScope VLM 客户端初始化成功 (base_url=%s)", base_url)

            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── LLM ──
    @classmethod
    def get_llm_client(
        cls,
        response_format: bool = True,
        role: Literal["default", "fast", "agent", "answer"] = "default",
        thinking: Optional[Literal["none", "light", "deep"]] = None,
    ) -> ChatOpenAI:
        """按模型角色、响应格式和 Thinking 等级缓存客户端。"""
        settings = get_settings()
        model_name = settings.llm_model_for_role(role)
        if not model_name:
            raise EnvironmentError(
                f"缺少 LLM 模型配置: LLM_{role.upper()}_MODEL/LLM_DEFAULT_MODEL"
            )
        thinking_key = thinking or "default"
        cache_key = (model_name, bool(response_format), thinking_key)
        client = cls._llm_clients.get(cache_key)
        if client is not None:
            return client

        with cls._llm_clients_lock:
            client = cls._llm_clients.get(cache_key)
            if client is None:
                client = cls._create_llm_client(
                    response_format=response_format,
                    role=role,
                    thinking=thinking,
                )
                cls._llm_clients[cache_key] = client
            return client

    @classmethod
    def _create_llm_client(
        cls,
        response_format: bool,
        role: str = "default",
        thinking: Optional[str] = None,
    ) -> ChatOpenAI:
        try:
            settings = get_settings()
            base_url, api_key = settings.require(
                "dashscope_api_base",
                "dashscope_api_key",
            )
            model_name = settings.llm_model_for_role(role)
            if not model_name:
                raise EnvironmentError(
                    f"缺少 LLM 模型配置: role={role}"
                )

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            client_kwargs = {
                "model_name": model_name,
                "temperature": 0,
                "openai_api_key": api_key,
                "openai_api_base": base_url,
                "model_kwargs": model_kwargs,
            }
            extra_body = cls._build_thinking_extra_body(
                model_name=model_name,
                thinking=thinking,
                light_budget=settings.llm_light_thinking_budget,
                deep_budget=settings.llm_deep_thinking_budget,
            )
            if extra_body:
                client_kwargs["extra_body"] = extra_body

            llm_client = ChatOpenAI(
                **client_kwargs
            )
            logger.info(
                "DashScope LLM 客户端初始化成功 "
                "(role=%s, model=%s, thinking=%s, json=%s)",
                role,
                model_name,
                thinking or "service_default",
                response_format,
            )
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    @staticmethod
    def _build_thinking_extra_body(
        model_name: str,
        thinking: Optional[str],
        light_budget: int = 4096,
        deep_budget: int = 16384,
    ) -> Dict[str, object]:
        """把项目内 Thinking 等级映射为 DashScope Chat 参数。

        Qwen3.8 使用 reasoning_effort，避免与 thinking_budget 同传；
        其他 Qwen 混合思考模型使用 enable_thinking + budget。
        """
        if thinking is None:
            return {}
        if thinking == "none":
            return {"enable_thinking": False}
        if thinking not in {"light", "deep"}:
            raise ValueError(f"不支持的 Thinking 等级: {thinking}")

        normalized_model = model_name.lower()
        if normalized_model.startswith("qwen3.8-"):
            return {
                "reasoning_effort": "low" if thinking == "light" else "medium"
            }
        return {
            "enable_thinking": True,
            "thinking_budget": (
                int(light_budget) if thinking == "light" else int(deep_budget)
            ),
        }

    # ── BGE-M3嵌入模型客户端 ──
    @classmethod
    def get_bge_m3_client(cls) -> BGEM3FlagModel:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3FlagModel:
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            settings = get_settings()
            model_name = settings.require("bge_m3_path")[0]
            device = settings.bge_device
            fp16 = settings.bge_fp16 and not device.lower().startswith("cpu")
            # 2. 创建
            bge_m3_ef = BGEM3FlagModel(
                model_name_or_path=model_name,
                devices=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e

    # ── BGE重排模型客户端 ──
    @classmethod
    def get_bge_reranker_client(cls) -> FlagReranker:
        """获取BGE重排模型，同一进程内只创建一次。"""
        return cls._get_or_create(
            "_bge_reranker_client",
            cls._bge_reranker_lock,
            cls._create_bge_reranker_client,
        )

    @classmethod
    def _create_bge_reranker_client(cls) -> FlagReranker:
        """从本地目录加载BGE重排模型。"""
        try:
            settings = get_settings()
            model_path = settings.require("bge_reranker_path")[0]
            device = settings.bge_reranker_device

            if not Path(model_path).is_dir():
                raise EnvironmentError(f"BGE重排模型目录不存在: {model_path}")

            use_fp16 = settings.bge_reranker_fp16

            # CPU环境不使用FP16，避免模型初始化或推理失败。
            if device.lower().startswith("cpu"):
                use_fp16 = False

            reranker = FlagReranker(
                model_name_or_path=model_path,
                devices=device,
                use_fp16=use_fp16,
                max_length=512,
            )

            logger.info(
                "BGE重排模型初始化成功 "
                f"(path={model_path}, device={device}, fp16={use_fp16})"
            )
            return reranker

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE重排模型初始化失败: {e}")
            raise ConnectionError(f"BGE重排模型初始化失败: {e}") from e
