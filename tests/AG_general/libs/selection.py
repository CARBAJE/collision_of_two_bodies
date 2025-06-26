import numpy as np

def vectorized_tournament_selection(fitness, num_tournaments, tournament_size, pop_size,
                                    unique_in_column=True, unique_in_row=False):
    """
    Genera una matriz de torneos de forma vectorizada y retorna, para cada torneo,
    el índice del individuo ganador (el de menor fitness).
    
    Args:
      - fitness: array con los fitness de la población (longitud = pop_size).
      - num_tournaments: número de torneos a realizar (por ejemplo, el número total
                         de selecciones de padres requeridas en la generación).
      - tournament_size: número de individuos que participan en cada torneo.
      - pop_size: tamaño de la población.
      - unique_in_column: si True, para cada posición (columna) se eligen candidatos sin
                          repetición entre torneos.
      - unique_in_row: si True, en cada torneo (fila) los candidatos serán únicos.
                    (Por defecto se permite repetir en la fila).
    
    Returns:
      - winners: array de índices ganadores (uno por torneo).
      - tournament_matrix: la matriz de candidatos (de tamaño [num_tournaments x tournament_size]).
    """
    if unique_in_row:
        # Para cada torneo (fila), muestreamos sin reemplazo (cada fila es única)
        tournament_matrix = np.array([np.random.choice(pop_size, size=tournament_size, replace=False)
                                      for _ in range(num_tournaments)])
    else:
        # Permitir repetición en la fila, pero controlar la no repetición en cada columna
        if unique_in_column:
            # Para cada columna, se genera una permutación de los índices (o se usan números aleatorios sin repetición)
            # Siempre que num_tournaments <= pop_size.
            if num_tournaments > pop_size:
                # Si se requieren más torneos que individuos, se hace sin la restricción por columna.
                tournament_matrix = np.random.randint(0, pop_size, size=(num_tournaments, tournament_size))
            else:
                cols = []
                for j in range(tournament_size):
                    # Para la columna j, se toman num_tournaments índices sin repetición
                    perm = np.random.permutation(pop_size)
                    cols.append(perm[:num_tournaments])
                tournament_matrix = np.column_stack(cols)
        else:
            # Sin restricciones, se muestrea con reemplazo para cada candidato.
            tournament_matrix = np.random.randint(0, pop_size, size=(num_tournaments, tournament_size))
    
    # Para cada torneo (fila de la matriz), se selecciona el candidato con el menor fitness.
    winners = []
    for row in tournament_matrix:
        row_fitness = fitness[row]
        winner_index = row[np.argmin(row_fitness)]
        winners.append(winner_index)
    winners = np.array(winners)
    return winners, tournament_matrix