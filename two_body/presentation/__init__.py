"""
Capa de presentacion: interfaz Qt y visualizacion Matplotlib.
"""

from .ui import launch_preview_window  # noqa: F401
from .visualization import Visualizer  # noqa: F401

__all__ = ["launch_preview_window", "Visualizer"]


if __name__ == "__main__":
    print("Ejecuta ui.launch_preview_window() o Visualizer.quick_view() para pruebas.")
