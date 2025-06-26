import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

def plot_surface_3d_with_constraints(objective_func, g_constraints, lower_bound, upper_bound, 
                                  best_solutions_list, func_key, func_name):
    """
    Grafica la superficie 3D de la función, su proyección en el plano XY y
    resalta la región que cumple con las restricciones.
    
    Args:
        objective_func: Función objetivo a graficar
        g_constraints: Lista de funciones de restricción g(x) <= 0
        lower_bound: Límites inferiores [x1_min, x2_min]
        upper_bound: Límites superiores [x1_max, x2_max]
        best_solutions_list: Lista de mejores soluciones encontradas
        func_key: Clave de la función para guardar archivos
        func_name: Nombre de la función para las etiquetas
    """
    # Crear malla para graficación
    num_points = 100
    x_vals = np.linspace(lower_bound[0], upper_bound[0], num_points)
    y_vals = np.linspace(lower_bound[1], upper_bound[1], num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    # Calcular valor de la función objetivo en cada punto
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_func([X[i, j], Y[i, j]])
    
    # Crear máscara para la región factible
    feasible_region = np.ones_like(Z, dtype=bool)
    
    # Para cada restricción, actualizar la máscara
    for g in g_constraints:
        constraint_values = np.zeros_like(Z)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                constraint_values[i, j] = g([X[i, j], Y[i, j]])
        # g(x) <= 0 para que sea factible
        feasible_region = feasible_region & (constraint_values <= 0)
    
    # Colores para las soluciones
    colors = [plt.cm.jet(i/len(best_solutions_list)) for i in range(len(best_solutions_list))]
    
    # Crear figura con dos subplots
    fig = plt.figure(figsize=(14, 7))
    
    # Subplot 1: Vista 3D
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Crear una fuente de luz para mejorar la visualización 3D
    ls = LightSource(azdeg=315, altdeg=45)
    illuminated_surface = ls.shade(Z, plt.cm.viridis)
    
    # Graficar superficie con transparencia
    surf = ax3d.plot_surface(X, Y, Z, facecolors=illuminated_surface,
                            rstride=1, cstride=1, alpha=0.7, linewidth=0)
    
    # Graficar la región factible en la superficie con un color distinto
    Z_feasible = np.where(feasible_region, Z, np.nan)
    feasible_surf = ax3d.plot_surface(X, Y, Z_feasible, color='green', alpha=0.5, linewidth=0)
    
    # Añadir barra de color
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label='Valor de la función objetivo')
    
    # Configurar etiquetas
    ax3d.set_title(f'Superficie 3D con Región Factible - {func_name}')
    ax3d.set_xlabel('x1')
    ax3d.set_ylabel('x2')
    ax3d.set_zlabel('Fitness')
    
    # Marcar los mejores puntos encontrados
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        f_val = objective_func([bx, by])
        col = colors[idx]
        # Aumentar tamaño de los puntos y añadir contorno negro para visibilidad
        ax3d.scatter(bx, by, f_val, color=col, s=100, marker='o', edgecolors='black')
    
    # Ajustar ángulo de vista
    ax3d.view_init(elev=30, azim=45)
    
    # Subplot 2: Proyección XY con restricciones
    ax_xy = fig.add_subplot(1, 2, 2)
    
    # Graficar contorno de función objetivo
    cont = ax_xy.contourf(X, Y, Z, cmap='viridis', alpha=0.7)
    cbar = fig.colorbar(cont, ax=ax_xy, label='Valor de la función objetivo')
    
    # Marcar región factible con un color semi-transparente
    feasible_mask = np.ma.masked_where(~feasible_region, np.ones_like(X))
    ax_xy.imshow(feasible_mask, extent=[lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]], 
                origin='lower', cmap=plt.cm.Greens, alpha=0.3, aspect='auto')
    
    # Dibujar líneas de restricción
    contour_levels = [0]  # g(x) = 0 marca el límite
    linestyles = ['--', '-.', ':']  # Diferentes estilos para cada restricción
    for i, g in enumerate(g_constraints):
        constraint_values = np.zeros_like(Z)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                constraint_values[j, k] = g([X[j, k], Y[j, k]])
        
        # Dibujar línea donde g(x) = 0
        cs = ax_xy.contour(X, Y, constraint_values, levels=contour_levels, 
                         colors=['red'], linestyles=[linestyles[i % len(linestyles)]], 
                         linewidths=2)
        # Etiquetar la restricción
        ax_xy.clabel(cs, inline=1, fontsize=10, fmt=f'g{i+1}(x)=0')
    
    # Marcar los mejores puntos encontrados
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        col = colors[idx]
        # Verificar si el punto está en la región factible
        is_feasible = all(g([bx, by]) <= 0 for g in g_constraints)
        edge_color = 'lime' if is_feasible else 'red'
        
        ax_xy.scatter(bx, by, color=col, s=100, marker='o', 
                    edgecolors=edge_color, linewidths=2,
                    label=f'Run {idx+1} {"(factible)" if is_feasible else "(no factible)"}')
    
    # Etiquetas y leyenda
    ax_xy.set_title('Proyección XY con Región Factible')
    ax_xy.set_xlabel('x1')
    ax_xy.set_ylabel('x2')
    ax_xy.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Añadir una leyenda para la región factible
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.3, edgecolor='none', label='Región Factible')]
    ax_xy.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f"outputs/{func_key}/surface_3d_with_constraints_{func_key}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfica adicional: solo restricciones para mejor visualización
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colorear región factible
    ax.imshow(feasible_mask, extent=[lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]], 
             origin='lower', cmap=plt.cm.Greens, alpha=0.5, aspect='auto')
    
    # Dibujar líneas de restricción con leyenda
    for i, g in enumerate(g_constraints):
        constraint_values = np.zeros_like(Z)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                constraint_values[j, k] = g([X[j, k], Y[j, k]])
        
        cs = ax.contour(X, Y, constraint_values, levels=[0], 
                      colors=['red'], linestyles=[linestyles[i % len(linestyles)]], 
                      linewidths=2, label=f'g{i+1}(x)=0')
        ax.clabel(cs, inline=1, fontsize=10, fmt=f'g{i+1}(x)=0')
    
    # Marcar los mejores puntos
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        is_feasible = all(g([bx, by]) <= 0 for g in g_constraints)
        edge_color = 'lime' if is_feasible else 'red'
        ax.scatter(bx, by, color=colors[idx], s=120, marker='o', 
                 edgecolors=edge_color, linewidths=2,
                 label=f'Solución {idx+1}')
    
    # Configuración final
    ax.set_title('Espacio de Búsqueda y Restricciones')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(lower_bound[0], upper_bound[0])
    ax.set_ylim(lower_bound[1], upper_bound[1])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir leyenda personalizada
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='Región Factible'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='g1(x)=0'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='-.', label='g2(x)=0'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markeredgecolor='lime', markersize=10, label='Solución Factible'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markeredgecolor='red', markersize=10, label='Solución No Factible')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(f"outputs/{func_key}/constraints_only_{func_key}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_evolucion_fitness_with_violations(all_runs_history, constraint_violations_history, 
                                        func_key, func_name):
    """
    Grafica la evolución del fitness y de las violaciones de restricciones.
    
    Args:
        all_runs_history: Lista con el historial de fitness para cada corrida
        constraint_violations_history: Lista con historial de violaciones para cada corrida
        func_key: Clave de la función para guardar archivo
        func_name: Nombre de la función para las etiquetas
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Fitness original
    for idx, history in enumerate(all_runs_history):
        axs[0, 0].plot(history, label=f'Ejecución {idx+1}')
    axs[0, 0].set_xlabel('Generación')
    axs[0, 0].set_ylabel('Mejor Fitness')
    axs[0, 0].set_title(f'Evolución del Fitness (Original) - {func_name}')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Subplot 2: Fitness normalizado
    for idx, history in enumerate(all_runs_history):
        h = np.array(history)
        h_min, h_max = h.min(), h.max()
        if h_max == h_min:
            norm_history = np.zeros_like(h)
        else:
            norm_history = (h - h_min) / (h_max - h_min)
        axs[0, 1].plot(norm_history, label=f'Ejecución {idx+1}')
    axs[0, 1].set_xlabel('Generación')
    axs[0, 1].set_ylabel('Fitness Normalizado')
    axs[0, 1].set_title(f'Evolución del Fitness (Normalizado) - {func_name}')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Subplot 3: Violaciones de restricciones
    if constraint_violations_history:
        for idx, history in enumerate(constraint_violations_history):
            axs[1, 0].plot(history, label=f'Ejecución {idx+1}')
        axs[1, 0].set_xlabel('Generación')
        axs[1, 0].set_ylabel('Suma de Violaciones')
        axs[1, 0].set_title('Evolución de Violaciones de Restricciones')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Subplot 4: Violaciones normalizadas
        for idx, history in enumerate(constraint_violations_history):
            h = np.array(history)
            h_max = h.max()
            if h_max == 0:
                norm_history = np.zeros_like(h)
            else:
                norm_history = h / h_max
            axs[1, 1].plot(norm_history, label=f'Ejecución {idx+1}')
        axs[1, 1].set_xlabel('Generación')
        axs[1, 1].set_ylabel('Violaciones Normalizadas')
        axs[1, 1].set_title('Evolución de Violaciones (Normalizada)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    else:
        # Si no hay datos de violaciones, eliminar los subplots inferiores
        fig.delaxes(axs[1, 0])
        fig.delaxes(axs[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"outputs/{func_key}/evolucion_fitness_with_constraints_{func_key}.png", dpi=300)
    plt.show()

def plot_surface_3d(objective_func, lower_bound, upper_bound, best_solutions_list, func_key, func_name):
    """Grafica la superficie 3D de la función y su proyección en el plano XY."""
    num_points = 100
    x_vals = np.linspace(lower_bound[0], upper_bound[0], num_points)
    y_vals = np.linspace(lower_bound[1], upper_bound[1], num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_func([X[i, j], Y[i, j]])
    
    colors = [plt.cm.jet(i/len(best_solutions_list)) for i in range(len(best_solutions_list))]
    
    fig = plt.figure(figsize=(12, 6))
    
    # Subplot 1: Vista 3D
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
    ax3d.set_title(f'Superficie 3D - {func_name}')
    ax3d.set_xlabel('x1')
    ax3d.set_ylabel('x2')
    ax3d.set_zlabel('Fitness')
    
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        f_val = objective_func([bx, by])
        col = colors[idx]
        ax3d.scatter(bx, by, f_val, color=col, s=100, marker='o')
    ax3d.view_init(elev=30, azim=30)
    
    # Subplot 2: XY
    ax_xy = fig.add_subplot(1, 2, 2)
    cont = ax_xy.contourf(X, Y, Z, cmap='viridis')
    fig.colorbar(cont, ax=ax_xy)
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        col = colors[idx]
        ax_xy.scatter(bx, by, color=col, s=100, marker='o', label=f'Run {idx+1}')
    ax_xy.set_title('Proyección XY')
    ax_xy.set_xlabel('x1')
    ax_xy.set_ylabel('x2')
    ax_xy.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/{func_key}/surface_3d_{func_key}.png")
    plt.show()