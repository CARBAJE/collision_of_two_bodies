"""
Logica para modificar parametros del sistema (m1, m2) durante la optimizacion.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..core.config import Config
from .ga import StreamingGA


class ParameterModifier:
    """
    Encapsula reglas sencillas para recentrar o perturbar parametros.

    Esta clase abstrae la logica que en el monolito estaba embebida en el
    controlador, alineando el diseno con el bloque 8 del diagrama.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def on_stagnation(
        self,
        ga: StreamingGA,
        best: Optional[Tuple[float, float]],
        radius: float,
    ) -> float:
        """
        Aplica una estrategia de reseed cuando la busqueda se estanca.

        Devuelve el nuevo radio de exploracion que debe utilizarse en
        iteraciones futuras.
        """
        if best is None:
            return radius
        ga.reseed_around(best, radius)
        return radius * self.cfg.radius_decay


if __name__ == "__main__":
    from ..core.config import Config

    cfg = Config()
    ga = StreamingGA(cfg)
    modifier = ParameterModifier(cfg)
    radius = modifier.on_stagnation(ga, (1.0, 1.0), cfg.local_radius)
    print("Nuevo radio tras estancamiento:", radius)
