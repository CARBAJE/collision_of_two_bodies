"""
Capa logica: motor de optimizacion y adaptacion de parametros.
"""

from .fitness import FitnessEvaluator  # noqa: F401
from .ga import StreamingGA  # noqa: F401
from .parameters import ParameterModifier  # noqa: F401
from .controller import ContinuousOptimizationController  # noqa: F401

__all__ = ["FitnessEvaluator", "StreamingGA", "ParameterModifier", "ContinuousOptimizationController"]


if __name__ == "__main__":
    from ..core.config import Config
    from ..core.cache import HierarchicalCache

    cfg = Config()
    evaluator = FitnessEvaluator(HierarchicalCache(), cfg)
    print("Evaluador preparado con poblacion vacia:", evaluator.evaluate_batch([]))
