import customtkinter as ctk
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
import matplotlib.pyplot as plt # Necesario para cerrar figuras

# 1. Definir el directorio actual (donde está streamlit_app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subir un nivel para llegar al directorio raíz del proyecto
#    (El directorio que contiene las carpetas 'two_body' y 'ui')
project_root = os.path.join(current_dir, '..')

# 3. Agregar el directorio raíz a la ruta de búsqueda de módulos de Python (sys.path)
sys.path.append(project_root)

# Importaciones
try:
    from two_body.core.config import Config
    from two_body.logic.controller import ContinuousOptimizationController
    from two_body.simulation.rebound_adapter import ReboundSim
    from two_body.presentation.triDTry import Visualizer as Visualizer3D
except ImportError as e:
    # Mensaje de error si las librerías internas no se encuentran
    print(f"Error de importación de módulos internos: {e}")
    print("Asegúrate de ejecutar el script desde el directorio correcto o que 'two_body' esté configurado.")
    sys.exit(1)


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Simulación y Optimización N-Cuerpos")
        self.geometry(f"{1200}x{700}")

        # Configuración de la rejilla
        self.grid_columnconfigure(0, weight=0, minsize=180)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1, minsize=300)
        self.grid_rowconfigure(0, weight=1)

        self.body_entries = {}
        self.opt_entries = {}
        self.analysis_canvases = {}
        self.animation_ref = None # Referencia para mantener viva la animación
        self.current_fig = None # Referencia para la figura actual de la animación

        # ---BARRA LATERAL IZQUIERDA: Panel de Control ---
        self.frame_control = ctk.CTkFrame(self, corner_radius=0)
        self.frame_control.grid(row=0, column=0, sticky="nsew")
        self.frame_control.grid_columnconfigure(0, weight=1)
        self.render_control_panel()

        # --- ETIQUETA DE ESTADO (Nueva adición para mostrar errores/progreso) ---
        self.status_label = ctk.CTkLabel(self.frame_control, text="Listo para Optimizar", text_color="yellow")
        self.status_label.pack(pady=(10, 5), padx=10, fill="x")

        # ---PANEL CENTRAL: Canvas de Simulación ---
        self.frame_canvas = ctk.CTkFrame(self, corner_radius=0, fg_color="#000000")
        self.frame_canvas.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.render_canvas_panel()

        # ---BARRA LATERAL DERECHA: Panel de Análisis ---
        self.frame_analysis = ctk.CTkFrame(self, corner_radius=0)
        self.frame_analysis.grid(row=0, column=2, sticky="nsew")
        self.frame_analysis.grid_columnconfigure(0, weight=1)
        self.render_analysis_panel()

    # --- Funciones para Renderizar Contenido de los Paneles ---

    def render_control_panel(self):
        """Renderiza los controles de entrada (Masa, Posición, Velocidad) y los parámetros de optimización."""

        # --- 1. SECCIÓN SUPERIOR: CONFIGURACIÓN PLANETARIA ---

        frame_planetaria = ctk.CTkFrame(self.frame_control, fg_color="transparent")
        frame_planetaria.pack(pady=(10, 5), padx=10, fill="x")

        ctk.CTkLabel(frame_planetaria, text="CONFIGURACIÓN PLANETARIA",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Controles para Cuerpo 1
        self.render_body_controls(frame_planetaria, "Cuerpo 1",
                                  mass_val=1.25, pos_val=(-0.2916667, 0.0, 0.0), vel_val=(0.0, -4.19812977, 0.0))

        # Controles para Cuerpo 2
        self.render_body_controls(frame_planetaria, "Cuerpo 2",
                                  mass_val=0.75, pos_val=(0.4083333, 0.0, 0.0), vel_val=(0.0, 5.87738168, 0.0))

        # NUEVO: Controles para Cuerpo 3
        self.render_body_controls(frame_planetaria, "Cuerpo 3",
                                  mass_val=0.25, pos_val=(0.0, 1.8, 0.0), vel_val=(-3.2, 0.0, 0.0))

        # --- Separador ---
        ctk.CTkFrame(self.frame_control, height=2, fg_color="gray").pack(fill="x", padx=10, pady=10)

        # --- 2. SECCIÓN INFERIOR: PARÁMETROS DE LA OPTIMIZACIÓN ---

        frame_optimizacion = ctk.CTkFrame(self.frame_control, fg_color="transparent")
        frame_optimizacion.pack(pady=(5, 10), padx=10, fill="x")

        ctk.CTkLabel(frame_optimizacion, text="PARÁMETROS DE LA OPTIMIZACIÓN",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Parámetros del Algoritmo Genético
        self.render_optimization_params(frame_optimizacion)

        # Botón de Inicio
        ctk.CTkButton(self.frame_control, text="Iniciar Optimización", fg_color="green", command=self.run_optimization).pack(pady=20, padx=10, fill="x")


    def render_body_controls(self, parent_frame, title, mass_val, pos_val, vel_val):
        """
        Plantilla para los controles de un cuerpo, ahora almacenando las referencias
        de los widgets de entrada en self.body_entries.
        """

        frame_body = ctk.CTkFrame(parent_frame, fg_color="transparent", border_width=1, border_color="#333")
        frame_body.pack(pady=(5, 7), padx=0, fill="x")

        ctk.CTkLabel(frame_body, text=title, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=5, pady=(3,0))

        inner_frame = ctk.CTkFrame(frame_body, fg_color="transparent")
        inner_frame.pack(padx=5, pady=3, fill="x")

        inner_frame.columnconfigure((0, 2), weight=2)
        inner_frame.columnconfigure((1, 3), weight=1)

        prefix = title.replace(" ", "_")

        # --- Fila 0: Masa ---
        ctk.CTkLabel(inner_frame, text="Masa (M):", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w", pady=1)
        entry_mass = ctk.CTkEntry(inner_frame, placeholder_text=f"{mass_val:.2f}", height=22)
        entry_mass.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(5, 0), pady=1)
        self.body_entries[f"{prefix}_mass"] = entry_mass
        # Rellenamos con el valor de ejemplo
        entry_mass.insert(0, f"{mass_val:.2f}")

        # --- Fila 1: Posición X y Posición Y ---
        ctk.CTkLabel(inner_frame, text="Posición X:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", pady=1)
        entry_pos_x = ctk.CTkEntry(inner_frame, placeholder_text=str(pos_val[0]), height=22)
        entry_pos_x.grid(row=1, column=1, sticky="ew", padx=5, pady=1)
        self.body_entries[f"{prefix}_pos_x"] = entry_pos_x
        entry_pos_x.insert(0, str(pos_val[0]))

        ctk.CTkLabel(inner_frame, text="Posición Y:", font=ctk.CTkFont(size=11)).grid(row=1, column=2, sticky="w", pady=1)
        entry_pos_y = ctk.CTkEntry(inner_frame, placeholder_text=str(pos_val[1]), height=22)
        entry_pos_y.grid(row=1, column=3, sticky="ew", padx=5, pady=1)
        self.body_entries[f"{prefix}_pos_y"] = entry_pos_y
        entry_pos_y.insert(0, str(pos_val[1]))

        # --- Fila 2: Posición Z y Velocidad X ---
        ctk.CTkLabel(inner_frame, text="Posición Z:", font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky="w", pady=1)
        entry_pos_z = ctk.CTkEntry(inner_frame, placeholder_text=str(pos_val[2]), height=22)
        entry_pos_z.grid(row=2, column=1, sticky="ew", padx=5, pady=1)
        self.body_entries[f"{prefix}_pos_z"] = entry_pos_z
        entry_pos_z.insert(0, str(pos_val[2]))

        ctk.CTkLabel(inner_frame, text="Velocidad X:", font=ctk.CTkFont(size=11)).grid(row=2, column=2, sticky="w", pady=1)
        entry_vel_x = ctk.CTkEntry(inner_frame, placeholder_text=str(vel_val[0]), height=22)
        entry_vel_x.grid(row=2, column=3, sticky="ew", padx=5, pady=1)
        self.body_entries[f"{prefix}_vel_x"] = entry_vel_x
        entry_vel_x.insert(0, str(vel_val[0]))

        # --- Fila 3: Velocidad Y y Velocidad Z ---
        ctk.CTkLabel(inner_frame, text="Velocidad Y:", font=ctk.CTkFont(size=11)).grid(row=3, column=0, sticky="w", pady=1)
        entry_vel_y = ctk.CTkEntry(inner_frame, placeholder_text=str(vel_val[1]), height=22)
        entry_vel_y.grid(row=3, column=1, sticky="ew", padx=5, pady=1)
        self.body_entries[f"{prefix}_vel_y"] = entry_vel_y
        entry_vel_y.insert(0, str(vel_val[1]))

        ctk.CTkLabel(inner_frame, text="Velocidad Z:", font=ctk.CTkFont(size=11)).grid(row=3, column=2, sticky="w", pady=1)
        entry_vel_z = ctk.CTkEntry(inner_frame, placeholder_text=str(vel_val[2]), height=22)
        entry_vel_z.grid(row=3, column=3, sticky="ew", padx=5, pady=1)
        self.body_entries[f"{prefix}_vel_z"] = entry_vel_z
        entry_vel_z.insert(0, str(vel_val[2]))


    def render_optimization_params(self, parent_frame):
        """Renderiza los parámetros para el algoritmo genético y almacena las referencias."""

        frame_opt = ctk.CTkFrame(parent_frame, fg_color="transparent")
        frame_opt.pack(pady=5, padx=0, fill="x")

        # Configuramos la rejilla para 4 columnas: LBL | ENTRY | LBL | ENTRY
        frame_opt.columnconfigure((0, 2), weight=2) # Etiquetas (Columna 0 y 2)
        frame_opt.columnconfigure((1, 3), weight=1) # Entradas (Columna 1 y 3)

        # --- Fila 0: Nº Generaciones (n_gen_step) y Nº Individuos (pop_size) ---

        # Columna 0 y 1: Nº Generaciones (Lo mapearemos a 'n_gen_step' en la Config)
        ctk.CTkLabel(frame_opt, text="Nº Generaciones:", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w", pady=3)
        entry_gen = ctk.CTkEntry(frame_opt, placeholder_text="10", height=22) # Usamos 10 para ejecucion rapida
        entry_gen.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=3)

        # 💾 Almacenamiento
        self.opt_entries["n_gen"] = entry_gen
        entry_gen.insert(0, "10") # Valor reducido para ejecucion rapida

        # Columna 2 y 3: Nº Individuos (Tamaño de Población)
        ctk.CTkLabel(frame_opt, text="Nº Individuos:", font=ctk.CTkFont(size=11)).grid(row=0, column=2, sticky="w", pady=3)
        entry_pop = ctk.CTkEntry(frame_opt, placeholder_text="50", height=22) # Usamos 50 para ejecucion rapida
        entry_pop.grid(row=0, column=3, sticky="ew", padx=5, pady=3)

        # 💾 Almacenamiento
        self.opt_entries["pop_size"] = entry_pop
        entry_pop.insert(0, "50") # Valor reducido para ejecucion rapida

        # --- Fila 1: Tasa de Mutación y Peso de Periodicidad ---

        # Columna 0 y 1: Tasa de Mutación
        ctk.CTkLabel(frame_opt, text="Tasa Mutación:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", pady=3)
        entry_mut = ctk.CTkEntry(frame_opt, placeholder_text="0.2", height=22) # Usamos 0.2 como valor del ejemplo
        entry_mut.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=3)

        # 💾 Almacenamiento
        self.opt_entries["mutation"] = entry_mut
        entry_mut.insert(0, "0.2") # Valor de ejemplo de la configuración GA (mutation)

        # Columna 2 y 3: Peso de Periodicidad
        ctk.CTkLabel(frame_opt, text="Peso Periodicidad:", font=ctk.CTkFont(size=11)).grid(row=1, column=2, sticky="w", pady=3)
        entry_weight = ctk.CTkEntry(frame_opt, placeholder_text="0.02", height=22) # Usamos 0.02 como valor del ejemplo
        entry_weight.grid(row=1, column=3, sticky="ew", padx=5, pady=3)

        # 💾 Almacenamiento
        self.opt_entries["periodicity_weight"] = entry_weight
        entry_weight.insert(0, "0.02") # Valor de ejemplo de la configuración (periodicity_weight)


    def render_canvas_panel(self):
        """Renderiza el área para la simulación (Canvas)."""
        # Contenedor inicial para la animación/simulación
        ctk.CTkLabel(self.frame_canvas, text="Lienzo de Simulación 3D",
                     text_color="gray", font=ctk.CTkFont(size=18, slant="italic")).pack(expand=True, padx=20, pady=20)

    def render_analysis_panel(self):
        """Renderiza las gráficas de análisis (Lyapunov y Evolución del Fitness)."""
        ctk.CTkLabel(self.frame_analysis, text="Análisis de Estabilidad y Optimización",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # --- Gráfica 1: Lyapunov ---
        ctk.CTkLabel(self.frame_analysis, text="Exponente de Lyapunov").pack(pady=(5,0))
        # Se reduce el figsize para que ocupe menos espacio. Antes era (3, 2).
        self.draw_plot_placeholder(self.frame_analysis, "lyapunov", figsize=(3, 1.5))

        # --- Eliminado: Gráfica de Divergencia ---

        # --- Gráfica 2: Evolución del Fitness ---
        ctk.CTkLabel(self.frame_analysis, text="Evolución del Fitness").pack(pady=(15,0))
        # Se reduce el figsize para que ocupe menos espacio. Antes era (3, 2).
        self.draw_plot_placeholder(self.frame_analysis, "fitness", figsize=(3, 1.5))

        # --- Métrica de ejemplo ---
        self.best_fitness_label = ctk.CTkLabel(self.frame_analysis, text="Mejor Fitness: N/A", text_color="green")
        self.best_fitness_label.pack(pady=(20, 5))

    def draw_plot_placeholder(self, parent_frame, plot_name, figsize=(3, 2)):
        """Crea un placeholder de gráfica usando Matplotlib."""
        fig = Figure(figsize=figsize, dpi=100)
        plot = fig.add_subplot(111)
        plot.plot([1, 2, 3, 4, 5], [10, 8, 6, 4, 2], label=plot_name)
        plot.set_title(plot_name.capitalize())

        canvas_widget = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(pady=5, padx=10, fill="x", expand=False)
        self.analysis_canvases[plot_name] = canvas_widget
        plt.close(fig) # Cierra la figura de Matplotlib para liberar memoria


    def _slice_vectors(self, vectors, count):
                if len(vectors) < count:
                    raise ValueError("Config no tiene suficientes vectores iniciales")
                return tuple(tuple(float(coord) for coord in vectors[i]) for i in range(count))

    def run_optimization(self):
        """Genera el objeto Config a partir de la GUI y ejecuta el controlador."""

        # 1. Recolección y Validación de Datos (Muy simplificado)

        try:
            # 1.1. Parámetros Físicos (Masas, R0, V0)
            r0_list = []
            v0_list = []
            masses = []

            for i in range(1, 4): # Asumiendo Cuerpo 1, Cuerpo 2, Cuerpo 3
                prefix = f"Cuerpo_{i}"

                # Masas
                mass = float(self.body_entries[f"{prefix}_mass"].get() or "1.0")
                masses.append(mass) # Usaremos la masa central para los bounds

                # Posición (R0)
                px = float(self.body_entries[f"{prefix}_pos_x"].get() or "0.0")
                py = float(self.body_entries[f"{prefix}_pos_y"].get() or "0.0")
                pz = float(self.body_entries[f"{prefix}_pos_z"].get() or "0.0")
                r0_list.append((px, py, pz))

                # Velocidad (V0)
                vx = float(self.body_entries[f"{prefix}_vel_x"].get() or "0.0")
                vy = float(self.body_entries[f"{prefix}_vel_y"].get() or "0.0")
                vz = float(self.body_entries[f"{prefix}_vel_z"].get() or "0.0")
                v0_list.append((vx, vy, vz))

            # 1.2. Parámetros de Optimización (GA)
            pop_size = int(self.opt_entries["pop_size"].get() or "64")
            mutation = float(self.opt_entries["mutation"].get() or "0.2")
            periodicity_weight = float(self.opt_entries["periodicity_weight"].get() or "0.0")
            n_gen = int(self.opt_entries["n_gen"].get() or "5") # Usando la entrada de Nº Generaciones

            # 2. Creación del Diccionario 'case' (Similar a tu ejemplo)
            gui_case_dict = {
                # SIMULACIÓN (Usando valores por defecto o predefinidos)
                "r0": tuple(r0_list),
                "v0": tuple(v0_list),
                "t_end_short": 0.5,
                "t_end_long": 4.0,
                "dt": 0.02,
                "integrator": "ias15",
                "periodicity_weight": periodicity_weight,

                # FÍSICOS (Generamos un rango +/- 10% alrededor de la masa de la GUI)
                "mass_bounds": tuple([(m * 0.9, m * 1.1) for m in masses]),
                "G": 39.47841760435743,

                # GA
                "pop_size": pop_size,
                "mutation": mutation,
                "n_gen_step": 5,

                "max_epochs": n_gen,

                # I/O
                "artifacts_dir": "artifacts/gui_run",
                # PONEMOS HEADLESS EN FALSE PARA QUE EL VIZ_3D NO INTENTE MOSTRAR LA FIGURA INMEDIATAMENTE
                "headless": False,
            }

            # 3. Creación del Objeto Config
            cfg = Config(**gui_case_dict)

            # 4. Ejecución Cronometrada
            self.update_status("Iniciando optimización...", color="orange")

            controller = ContinuousOptimizationController(cfg)

            results = controller.run()
            metrics = controller.metrics

            best_fitness = results["best"]["fitness"]
            self.update_status(f"Optimización finalizada. Mejor Fitness: {best_fitness:.4e}", color="green")
            self.best_fitness_label.configure(text=f"Mejor Fitness: {best_fitness:.4e}")

            # Inicialización del simulador para la trayectoria final
            sim_builder = ReboundSim(G=cfg.G, integrator=cfg.integrator)
            best_masses = tuple(results["best"]["masses"])

            r0 = self._slice_vectors(cfg.r0, len(best_masses))
            v0 = self._slice_vectors(cfg.v0, len(best_masses))

            sim = sim_builder.setup_simulation(best_masses, r0, v0)
            traj = sim_builder.integrate(sim, t_end=cfg.t_end_long, dt=cfg.dt)

            xyz_tracks = [traj[:, i, :3] for i in range(traj.shape[1])]

            # Usamos un Visualizer3D para las gráficas 2D
            viz_3d_analysis = Visualizer3D(headless=True) # Headless para las gráficas 2D

            self.update_analysis_plots(metrics, viz_3d_analysis)

            # Usamos otro Visualizer3D para la animación (o reusamos con headless=False, pero mejor uno nuevo)
            viz_3d_animation = Visualizer3D(headless=False)
            self.update_canvas_3d_animation(viz_3d_animation, xyz_tracks, best_masses)

        except ValueError as e:
            self.update_status(f"Error de entrada: Verifica que todos los campos sean números. Detalle: {e}", color="red")
        except Exception as e:
            self.update_status(f"Error de ejecución: {type(e).__name__}: {e}", color="red")

    def update_status(self, message, color="white"):
        """Actualiza el estado en la GUI y en la consola."""
        print(f"ESTADO: {message}")
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=message, text_color=color)

    def update_analysis_plots(self, metrics, viz_3d):
        """
        Actualiza las gráficas de lambda y fitness en el panel de análisis,
        conservando el tamaño reducido.
        """
        # Eliminar placeholders anteriores
        for key in ['lyapunov', 'fitness']:
            if key in self.analysis_canvases:
                # Destruye el widget de tkinter que contiene la gráfica
                self.analysis_canvases[key].get_tk_widget().destroy()
                del self.analysis_canvases[key]

        # Recrear gráfica de Evolución de Lambda (Lyapunov)
        fig_lambda = viz_3d.plot_lambda_evolution(
            lambda_history=metrics.best_lambda_per_epoch,
            epoch_history=metrics.epoch_history,
            title="Exponente de Lyapunov (λ)",
            moving_average_window=5,
            figsize=(6.0, 3.0)
        )
        # El label del título de la gráfica está justo antes de donde se insertará el nuevo canvas
        self._insert_plot_after_label("Exponente de Lyapunov", fig_lambda, 'lyapunov')

        # Recrear gráfica de Evolución del Fitness
        fig_fitness = viz_3d.plot_fitness_evolution(
            fitness_history=metrics.best_fitness_per_epoch,
            epoch_history=metrics.epoch_history,
            title="Evolución del Fitness",
            moving_average_window=5,
            figsize=(6.0, 3.0)
        )
        # El label del título de la gráfica está justo antes de donde se insertará el nuevo canvas
        self._insert_plot_after_label("Evolución del Fitness", fig_fitness, 'fitness')

        plt.close(fig_lambda)
        plt.close(fig_fitness)

    def _insert_plot_after_label(self, label_text, fig, plot_key):
        """Helper para insertar un plot de Matplotlib después de una etiqueta específica."""

        # Buscamos el widget que tiene el texto 'label_text' para determinar dónde insertar
        widgets = self.frame_analysis.winfo_children()
        insert_index = -1
        for i, widget in enumerate(widgets):
            if isinstance(widget, ctk.CTkLabel) and widget.cget("text") == label_text:
                # La posición de inserción debe ser después de este label, pero antes del siguiente widget
                insert_index = i + 1
                break

        if insert_index != -1:
            canvas_widget = FigureCanvasTkAgg(fig, master=self.frame_analysis)
            canvas_widget.draw()

            # Usamos grid_slaves para obtener los widgets que usan pack y determinar la posición de inserción
            # Sin embargo, dado que estás usando PACK, debemos hacer un poco de trampa.
            # La manera más simple usando pack es simplemente volver a empaquetar, ya que el anterior se destruyó.
            canvas_widget.get_tk_widget().pack(pady=5, padx=10, fill="x", expand=False)
            self.analysis_canvases[plot_key] = canvas_widget

        else:
            # Fallback si no encuentra el label (sólo para la versión con pack)
            canvas_widget = FigureCanvasTkAgg(fig, master=self.frame_analysis)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(pady=5, padx=10, fill="x", expand=False)
            self.analysis_canvases[plot_key] = canvas_widget

    def update_canvas_3d_animation(self, viz_3d, xyz_tracks, best_masses):
        """
        Integra la animación 3D de órbitas en el panel central (self.frame_canvas).
        """
        # 1. Limpiar el panel central
        for widget in self.frame_canvas.winfo_children():
            widget.destroy()

        # Limpiar figura anterior si existe para liberar memoria
        if self.current_fig:
            plt.close(self.current_fig)
            self.current_fig = None

        # 2. Obtener la figura y la animación desde el Visualizer
        try:
            fig_3d, self.animation_ref = viz_3d.animate_3d(
                trajectories=xyz_tracks,
                title=f"Trayectorias 3D m1={best_masses[0]:.3f}, m2={best_masses[1]:.3f}, m3={best_masses[2]:.3f}",
                total_frames=len(xyz_tracks[0]),
                interval_ms=50,
                figsize=(10, 10) # Hacemos la animacion cuadrada y mas grande
            )
            # **IMPORTANTE para la animación**:
            # Es vital mantener la referencia a `self.animation_ref` (la animación de Matplotlib).
            # Si se destruye el objeto de animación, el bucle de actualización se detiene.
        except Exception as e:
            self.update_status(f"Error al crear animación 3D: {e}", color="red")
            return

        # 3. Integrar la figura en el Canvas de CustomTkinter
        canvas_3d = FigureCanvasTkAgg(fig_3d, master=self.frame_canvas)
        canvas_3d.draw()

        canvas_3d.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Guardamos la referencia a la figura actual
        self.current_fig = fig_3d
        # NO cerramos la figura aquí (plt.close(fig_3d)) porque si no, la animación se detiene.
        # La cerraremos antes de crear la siguiente.

        self.update_status("✅ Animación 3D de órbita cargada en el panel central.", color="green")

if __name__ == "__main__":
    app = App()
    app.mainloop()