"""
Package raiz para el sistema modular de optimizacion de dos cuerpos.

Cada submodulo representa una capa de la arquitectura mostrada en el
diagrama: nucleo compartido, simulacion, logica de optimizacion y capa de
presentacion.
"""

from .core.config import Config, set_global_seeds  # noqa: F401
from .core.telemetry import setup_logger  # noqa: F401

__all__ = ["Config", "set_global_seeds", "setup_logger"]


if __name__ == "__main__":
    # Prueba rapida para verificar que los imports principales funcionan.
    cfg = Config()
    logger = setup_logger()
    logger.info("Inicializacion basica completada.")
    print(cfg)
