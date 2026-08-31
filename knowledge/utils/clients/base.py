import logging
import threading

logger = logging.getLogger(__name__)


class BaseClientManager:
    """
    客户端管理器基类，提供线程安全的客户端懒加载模板。

    子类只需要关注「怎么创建客户端」，不用重复写锁逻辑。
    """

    @classmethod
    def _get_or_create(cls, attr_name: str, lock: threading.Lock, factory):
        """
        双重检查锁的通用模板。

        Args:
            attr_name: 类属性名（如 "_minio_client"）
            lock: 对应的线程锁
            factory: 无参工厂函数，返回客户端实例

        Returns:
            缓存的或新创建的客户端实例
        """
        # 第一次检查（无锁，快速路径）
        instance = getattr(cls, attr_name, None)
        if instance is not None:
            return instance

        with lock:
            # 第二次检查（持锁，防并发重复创建）
            instance = getattr(cls, attr_name, None)
            if instance is not None:
                return instance

            instance = factory()
            setattr(cls, attr_name, instance)
            return instance
