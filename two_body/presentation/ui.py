"""
Interfaz de usuario (placeholder) basada en PyQt5.
"""

from __future__ import annotations


def launch_preview_window() -> None:
    """
    Muestra una ventana sencilla siempre que PyQt5 este disponible.
    """

    try:
        from PyQt5 import QtWidgets  # type: ignore
    except Exception as exc:  # pragma: no cover - PyQt opcional
        print("PyQt5 no disponible. Instale PyQt5 para usar la interfaz.", exc)
        return

    app = QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.setWindowTitle("Sistema Visualizador del Modelo")
    layout = QtWidgets.QVBoxLayout(window)
    label = QtWidgets.QLabel("Placeholder de UI Qt para la visualizacion del modelo.")
    layout.addWidget(label)
    window.resize(480, 240)
    window.show()
    app.exec_()


if __name__ == "__main__":
    launch_preview_window()
