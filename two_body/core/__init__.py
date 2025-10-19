"""
Componentes compartidos por todas las capas del sistema: configuraciones,
telemetria y caches.
"""

from .config import Config, set_global_seeds  # noqa: F401
from .telemetry import setup_logger, MetricsLogger, Reporter  # noqa: F401
from .cache import HierarchicalCache, LRUCache  # noqa: F401

__all__ = [
    "Config",
    "set_global_seeds",
    "setup_logger",
    "MetricsLogger",
    "Reporter",
    "HierarchicalCache",
    "LRUCache",
]


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Core module smoke test completado.")
