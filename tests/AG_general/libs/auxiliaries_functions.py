from libs.functions import *
import numpy as np
import os
# ---------------------------
# Funciones auxiliares del GA
# ---------------------------
def initialize_population(pop_size, num_variables, lower_bound, upper_bound):
    """Inicializa la población uniformemente en el espacio de búsqueda."""
    return np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, num_variables))

def evaluate_individuals_with_constraints(population, f, Q, H, lam):
    """
    Evalúa las restricciones de un individuo.
    
    :param population: Lista de individuos a evaluar.
    :param f: Función objetivo.
    :param Q: Lista de restricciones de desigualdad (q(x) ≤ 0).
    :param H: Lista de restricciones de igualdad (h(x) = 0).
    :param LAMBDA: Parámetro de penalización.
    :return: Lista de aptitudes penalizadas.
    """
    
    Fp = []
    for ind in population:
        Rd = [max(0, q(ind))**2 for q in Q] if Q else [] # Penalización de desigualdades
        
        Re = [h(ind)**2 for h in H] if H else [] # Penalización de igualdades (sin max porque son = 0)
        
        P = sum(Rd) + sum(Re)  # Penalización total
        
        Fp.append(f(ind) + lam * P)  # Función penalizada
    
    return Fp


def get_image_paths(folder):
    """
    Obtiene las rutas a todas las imágenes dentro de la carpeta indicada.
    
    :param folder: Ruta de la carpeta que contiene las imágenes.
    :return: Lista con las rutas completas de los archivos que sean imágenes.
    """
    
    # Extensiones permitidas
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    image_paths = []
    
    # Recorremos todos los archivos de la carpeta
    for file in os.listdir(folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in allowed_extensions:
            image_paths.append(os.path.join(folder, file))
    return image_paths

def create_functions_json(lb, ub, num_runs, image_paths=None):
    """
    Crea un diccionario con dos entradas para cada imagen, similar a la estructura
    presentada en el código original, pero utilizando cada imagen de la lista.
    
    :param image_paths: Lista de rutas de imágenes.
    :return: Diccionario con la configuración para cada imagen.
    """
    functions_dict = {}

    # Para cada imagen se crean dos entradas: una para Entropy y otra para Standard Deviation
    for img in image_paths:
        base_name = os.path.splitext(os.path.basename(img))[0]
        
        functions_dict[f"{base_name}_Entropy"] = {
            "func": Entropy,  # Si necesitas almacenar la función, considera guardar su nombre o path
            "lb": lb,
            "ub": ub,
            "name": "Entropy",
            "num_runs": num_runs,
            "img_path": img
        }
        functions_dict[f"{base_name}_Std_Deviation"] = {
            "func": std_img,
            "lb": lb,
            "ub": ub,
            "name": "Standard_Deviation",
            "num_runs": num_runs,
            "img_path": img
        }
    return functions_dict