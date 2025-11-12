import customtkinter as ctk
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{1200}x{700}")
        self.grid_columnconfigure(0, weight=0, minsize=180) 
        self.grid_columnconfigure(1, weight=3)             
        self.grid_columnconfigure(2, weight=1, minsize=300) 
        self.grid_rowconfigure(0, weight=1)

        # ---BARRA LATERAL IZQUIERDA: Panel de Control ---
        self.frame_control = ctk.CTkFrame(self, corner_radius=0)
        self.frame_control.grid(row=0, column=0, sticky="nsew")
        self.frame_control.grid_columnconfigure(0, weight=1)
        self.render_control_panel()

        # ---PANEL CENTRAL: Canvas de Simulación ---
        self.frame_canvas = ctk.CTkFrame(self, corner_radius=0, fg_color="#000000") # Fondo negro para simulación espacial
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
                                  mass_val=1.0, pos_val=(-50.0, 0.0, 0.0), vel_val=(0.0, 5.0, 0.0))
        
        # Controles para Cuerpo 2
        self.render_body_controls(frame_planetaria, "Cuerpo 2", 
                                  mass_val=0.01, pos_val=(100.0, 0.0, 0.0), vel_val=(0.0, -2.0, 0.0))
        
        # NUEVO: Controles para Cuerpo 3
        self.render_body_controls(frame_planetaria, "Cuerpo 3", 
                                  mass_val=0.005, pos_val=(0.0, 75.0, 0.0), vel_val=(-1.0, 0.0, 1.0))

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
        ctk.CTkButton(self.frame_control, text="Iniciar Optimización", fg_color="green").pack(pady=20, padx=10, fill="x")


    def render_body_controls(self, parent_frame, title, mass_val, pos_val, vel_val):
        """Plantilla para los controles de un cuerpo con compresión máxima."""
        
        frame_body = ctk.CTkFrame(parent_frame, fg_color="transparent", border_width=1, border_color="#333")
        frame_body.pack(pady=(5, 7), padx=0, fill="x") # Reducimos pady para ahorrar espacio vertical
        
        ctk.CTkLabel(frame_body, text=title, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=5, pady=(3,0)) 

        inner_frame = ctk.CTkFrame(frame_body, fg_color="transparent")
        inner_frame.pack(padx=5, pady=3, fill="x")
        
        # Columna 0 y 2 (Etiquetas) tienen weight=2
        inner_frame.columnconfigure((0, 2), weight=2) 
        # Columna 1 y 3 (Entradas) tienen weight=1 
        inner_frame.columnconfigure((1, 3), weight=1) 

        # --- Fila 0: Masa ---
        ctk.CTkLabel(inner_frame, text="Masa (M):", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w", pady=1) # Fuente 11
        ctk.CTkEntry(inner_frame, placeholder_text=f"{mass_val:.2f}", height=22).grid(row=0, column=1, columnspan=3, sticky="ew", padx=(5, 0), pady=1)

        # --- Fila 1: Posición X y Posición Y ---
        ctk.CTkLabel(inner_frame, text="Posición X:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", pady=1)
        ctk.CTkEntry(inner_frame, placeholder_text=str(pos_val[0]), height=22).grid(row=1, column=1, sticky="ew", padx=5, pady=1)
        ctk.CTkLabel(inner_frame, text="Posición Y:", font=ctk.CTkFont(size=11)).grid(row=1, column=2, sticky="w", pady=1)
        ctk.CTkEntry(inner_frame, placeholder_text=str(pos_val[1]), height=22).grid(row=1, column=3, sticky="ew", padx=5, pady=1)

        # --- Fila 2: Posición Z y Velocidad X ---
        ctk.CTkLabel(inner_frame, text="Posición Z:", font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky="w", pady=1)
        ctk.CTkEntry(inner_frame, placeholder_text=str(pos_val[2]), height=22).grid(row=2, column=1, sticky="ew", padx=5, pady=1)
        ctk.CTkLabel(inner_frame, text="Velocidad X:", font=ctk.CTkFont(size=11)).grid(row=2, column=2, sticky="w", pady=1)
        ctk.CTkEntry(inner_frame, placeholder_text=str(vel_val[0]), height=22).grid(row=2, column=3, sticky="ew", padx=5, pady=1)

        # --- Fila 3: Velocidad Y y Velocidad Z ---
        ctk.CTkLabel(inner_frame, text="Velocidad Y:", font=ctk.CTkFont(size=11)).grid(row=3, column=0, sticky="w", pady=1)
        ctk.CTkEntry(inner_frame, placeholder_text=str(vel_val[1]), height=22).grid(row=3, column=1, sticky="ew", padx=5, pady=1)
        ctk.CTkLabel(inner_frame, text="Velocidad Z:", font=ctk.CTkFont(size=11)).grid(row=3, column=2, sticky="w", pady=1)
        ctk.CTkEntry(inner_frame, placeholder_text=str(vel_val[2]), height=22).grid(row=3, column=3, sticky="ew", padx=5, pady=1)


    def render_optimization_params(self, parent_frame):
        """Renderiza los parámetros para el algoritmo genético con layout de dos entradas por fila."""
        frame_opt = ctk.CTkFrame(parent_frame, fg_color="transparent")
        frame_opt.pack(pady=5, padx=0, fill="x")
        
        # Configuramos la rejilla para 4 columnas: LBL | ENTRY | LBL | ENTRY
        frame_opt.columnconfigure((0, 2), weight=2) # Etiquetas (Columna 0 y 2)
        frame_opt.columnconfigure((1, 3), weight=1) # Entradas (Columna 1 y 3)

        # --- Fila 0: Nº Generaciones y Nº Individuos ---
        
        # Columna 0 y 1: Nº Generaciones
        ctk.CTkLabel(frame_opt, text="Nº Generaciones:", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w", pady=3)
        ctk.CTkEntry(frame_opt, placeholder_text="100", height=22).grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=3)
        
        # Columna 2 y 3: Nº Individuos (Tamaño de Población)
        ctk.CTkLabel(frame_opt, text="Nº Individuos:", font=ctk.CTkFont(size=11)).grid(row=0, column=2, sticky="w", pady=3)
        ctk.CTkEntry(frame_opt, placeholder_text="50", height=22).grid(row=0, column=3, sticky="ew", padx=5, pady=3)
        
        # --- Fila 1: Tasa de Mutación y Peso de Periodicidad ---
        
        # Columna 0 y 1: Tasa de Mutación
        ctk.CTkLabel(frame_opt, text="Tasa Mutación:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", pady=3)
        ctk.CTkEntry(frame_opt, placeholder_text="0.01", height=22).grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=3)
        
        # Columna 2 y 3: Peso de Periodicidad
        ctk.CTkLabel(frame_opt, text="Peso Periodicidad:", font=ctk.CTkFont(size=11)).grid(row=1, column=2, sticky="w", pady=3)
        ctk.CTkEntry(frame_opt, placeholder_text="0.01", height=22).grid(row=1, column=3, sticky="ew", padx=5, pady=3)


    def render_canvas_panel(self):
        """Renderiza el área para la simulación (Canvas)."""
        ctk.CTkLabel(self.frame_canvas, text="Lienzo de Simulación (Integración de Matplotlib/Canvas)", 
                     text_color="gray", font=ctk.CTkFont(size=18, slant="italic")).pack(expand=True, padx=20, pady=20)
        
    def render_analysis_panel(self):
        """Renderiza las gráficas de análisis (Lyapunov, Divergencia y Evolución del Fitness)."""
        ctk.CTkLabel(self.frame_analysis, text="Análisis de Estabilidad y Optimización", 
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # --- Gráfica 1: Lyapunov ---
        ctk.CTkLabel(self.frame_analysis, text="Exponente de Lyapunov").pack(pady=(5,0))
        self.draw_plot_placeholder(self.frame_analysis, "lyapunov")

        # --- Gráfica 2: Divergencia ---
        ctk.CTkLabel(self.frame_analysis, text="Divergencia de Trayectorias").pack(pady=(15,0))
        self.draw_plot_placeholder(self.frame_analysis, "divergence")
        
        # --- Gráfica 3: Evolución del Fitness ---
        ctk.CTkLabel(self.frame_analysis, text="Evolución del Fitness").pack(pady=(15,0))
        self.draw_plot_placeholder(self.frame_analysis, "fitness")
        
        # --- Métrica de ejemplo ---
        ctk.CTkLabel(self.frame_analysis, text="Mejor Fitness: N/A", text_color="green").pack(pady=(20, 5))

    def draw_plot_placeholder(self, parent_frame, plot_name):
        """Crea un placeholder de gráfica usando Matplotlib."""
        fig = Figure(figsize=(3, 2), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot([1, 2, 3, 4, 5], [10, 8, 6, 4, 2], label=plot_name)
        plot.set_title(plot_name.capitalize())
        
        canvas_widget = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(pady=5, padx=10, fill="x", expand=False)
        
        
if __name__ == "__main__":
    app = App()
    app.mainloop()