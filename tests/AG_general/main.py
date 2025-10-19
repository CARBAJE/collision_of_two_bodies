import os
import pandas as pd
from AG_confs import *

from AG import genetic_algorithm
from libs.plot import *
from libs.functions import g1_finanzas
import time

def main():
    # Crear carpetas de salida generales
    os.makedirs("outputs", exist_ok=True)

    for func_key, func_data in FUNCTIONS.items():
        execution_time_start = time.time()
        f_obj = func_data["func"]
        g = func_data["g"]
        h = func_data["h"]
        lb = func_data["lb"]
        ub = func_data["ub"]
        func_name = func_data["name"]
        num_runs = func_data["num_runs"]

        # Carpetas específicas de cada función
        func_folder = f"outputs/{func_key}"
        os.makedirs(func_folder, exist_ok=True)
        hist_folder = os.path.join(func_folder, "historiales")
        res_folder = os.path.join(func_folder, "resumenes")
        os.makedirs(hist_folder, exist_ok=True)
        os.makedirs(res_folder, exist_ok=True)

        print(f"\n==============================================")
        print(f"  FUNCIÓN: {func_name}")
        print(f"==============================================")

        all_runs_history = []
        all_violations_history = []  # Nuevo: para almacenar violaciones de todas las corridas
        best_solutions_all_runs = []  # Guardaremos los mejores individuos (x1, x2) de cada corrida
        best_values_across_runs = []  # Guardaremos el best_val (fitness) de cada corrida

        for run in range(num_runs):
            print(f"\nEjecución {run+1}/{num_runs}")

            (best_sol, best_val,
             worst_sol, worst_val,
             avg_sol,  avg_val,
             std_val,
             best_fitness_history,
             best_x_history,
             constraint_violations_history,  # Nuevo: recibir historial de violaciones
             population_final,
             fitness_final,
             best_solutions_over_time) = genetic_algorithm(
                 f_obj, g, h, lb, ub,
                 pop_size=POP_SIZE,
                 num_generations=NUM_GENERATIONS,
                 tournament_size=TOURNAMENT_SIZE,
                 crossover_prob=CROSSOVER_PROB,
                 eta_c=ETA_C,
                 mutation_prob=MUTATION_PROB,
                 eta_mut=ETA_MUT,
                 lam = LAMBDA
             )

            # 1) Guardar historial
            df_historial = pd.DataFrame({
                "Generacion": np.arange(1, NUM_GENERATIONS + 1),
                "Mejor individuo": best_x_history,
                "Mejor Fitness": best_fitness_history
            })
            # Añadir columna de violaciones si existe
            if constraint_violations_history:
                df_historial["Violaciones"] = constraint_violations_history

            historial_filename = os.path.join(hist_folder, f"historial_run_{run+1}.csv")
            df_historial.to_csv(historial_filename, index=False)

            # 2) Guardar resumen de la corrida
            data_resumen = [
                ["Mejor"] + list(best_sol) + [best_val] ,
                ["Media"] + list(avg_sol) + [avg_val] ,
                ["Peor"] + list(worst_sol) + [worst_val]
            ]

            column_names = ["Indicador"] + [f"x{i}" for i in range(len(best_sol))] + ["Fitness"]

            df_resumen = pd.DataFrame(data_resumen, columns=column_names)
            resumen_filename = os.path.join(res_folder, f"resumen_run_{run+1}.csv")
            df_resumen.to_csv(resumen_filename, index=False)

            print(df_resumen.to_string(index=False))

            all_runs_history.append(best_fitness_history)
            if constraint_violations_history:
                all_violations_history.append(constraint_violations_history)
            best_solutions_all_runs.append(best_sol)
            best_values_across_runs.append(best_val)

        # ===========================================
        #       RESUMEN GLOBAL DE LAS CORRIDAS
        # ===========================================
        best_values_arr = np.array(best_values_across_runs)
        solutions_arr = np.array(best_solutions_all_runs)  # Cada fila: [x1, x2] del mejor individuo de cada corrida

        # Para el "Mejor" y "Peor", buscamos el índice de la corrida con mínimo y máximo fitness
        min_index = np.argmin(best_values_arr)
        max_index = np.argmax(best_values_arr)
        num_vars = solutions_arr.shape[1]

        data_global = [
            ["Mejor (Fitness)"] + list(solutions_arr[min_index, :]) + [best_values_arr[min_index]],
            ["Peor (Fitness)"] + list(solutions_arr[max_index, :]) + [best_values_arr[max_index]],
            ["Media"] + [np.mean(solutions_arr[:, i]) for i in range(num_vars)] + [np.mean(best_values_arr)],
            ["Desv. Estándar"] + [np.std(solutions_arr[:, i]) for i in range(num_vars)] + [np.std(best_values_arr)]
        ]

        # Generamos nombres de columnas automáticamente: x0, x1, ..., xN
        column_names = ["Indicador"] + [f"x{i}" for i in range(num_vars)] + ["Fitness"]

        df_global = pd.DataFrame(data_global, columns=column_names)

        global_filename = os.path.join(res_folder, "resumen_global_corridas.csv")
        df_global.to_csv(global_filename, index=False)
        print("\n==============================================")
        execution_time_end = time.time()
        execution_time = execution_time_end - execution_time_start
        print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
        # Graficar evolución del fitness y violaciones
        plot_evolucion_fitness_with_violations(all_runs_history, all_violations_history, func_key, func_name)

        # Graficar superficie 3D con restricciones si la función es de 2 variables
        #if len(lb) == 2 and g:
        #    plot_surface_3d_with_constraints(f_obj, g, lb, ub, best_solutions_all_runs, func_key, func_name)
        # Si no hay restricciones, usar la función original
        #elif len(lb) == 2:
        #    plot_surface_3d(f_obj, lb, ub, best_solutions_all_runs, func_key, func_name)

if __name__ == "__main__":
    main()
