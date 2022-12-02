import numpy as np
import pandas as pd 
import json
import scipy as sp
import time
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

class Individuo:
    def __init__(self, genotipo, fitness):
        self.genotipo = genotipo
        self.fitness = fitness
    # Ordenamos de menor a mayor los Individuos por su fitness
    def __lt__(self, other):
        return self.fitness < other.fitness

def init_ciudades(nciudades, radio):
    # Generamos nciudades en coordenadas cartesianas desde -radio a +radio para x e y
    ciudades = (np.random.random([nciudades,2]) * radio*2) - radio
    np.savetxt("ciudades.csv", ciudades, delimiter=",")

def cargar_ciudades(path):
    ciudades = np.loadtxt(path, delimiter=",")
    # p es la p-norma de Minkowski. p=1 distancia rectilínea, p=2 distancia euclidiana
    return (ciudades, sp.spatial.distance_matrix(ciudades, ciudades, p=2))
    

def init_poblacion(tam_poblacion, matriz_distancias, nciudades):
    # Generamos los genotipos de la poblacion por pormutaciones aleatorias
    genotipos_poblacion = [np.random.permutation(nciudades) for i in range(tam_poblacion)]
    # Generamos la poblacion a partir de los genotipos y sus fitness
    poblacion = np.array([Individuo(genotipo, fitness(genotipo, matriz_distancias, nciudades)) for genotipo in genotipos_poblacion])
    return np.sort(poblacion)

def generacion(params, poblacion, matriz_distancias): # STEP
    padres = seleccion_padres(params["npadres"], params["tam_poblacion"], poblacion.copy())
    cruzados = cruce(params["pcruce"], params["tam_poblacion"], params["nciudades"], padres.copy(), matriz_distancias)
    mutados = mutacion(params["pmutacion"], params["tam_poblacion"], cruzados.copy(), matriz_distancias, params["nciudades"])
    hijos = seleccion_hijos(poblacion[0], mutados.copy(), params["eliminacionelitismo"], params["tam_poblacion"])
    return hijos

def seleccion_padres(npadres, tam_poblacion, poblacion):
    padres = []
    for i_torneo in range(tam_poblacion):
        padres_elegidos = poblacion[np.random.choice(tam_poblacion, int(npadres))]
        ind_padre = np.argmin(padres_elegidos)
        padres.append(padres_elegidos[ind_padre])
    return np.array(padres)

def cruce(pcruce, tam_poblacion, nciudades, padres, matriz_distancias):
    cruzados = []
    for i_cruce in range(tam_poblacion//2):
        indices = np.random.choice(np.arange(len(padres)), 2, replace=False)
        padre_1 = padres[indices[0]]
        padre_2 = padres[indices[1]]
        padres = np.delete(padres, indices, 0)
        if np.random.random() <= pcruce:
            # Elegimos las posiciones del segmento mapeado
            posiciones = np.sort(np.random.choice(nciudades, 2, replace=False))
            hijo_1 = cruce_parcialmente_mapeado(padre_1.genotipo, padre_2.genotipo, posiciones)
            hijo_2 = cruce_parcialmente_mapeado(padre_2.genotipo, padre_1.genotipo, posiciones)
            cruzados.append(Individuo(hijo_1, fitness(hijo_1, matriz_distancias, nciudades)))
            cruzados.append(Individuo(hijo_2, fitness(hijo_2, matriz_distancias, nciudades)))
        else:
            cruzados.append(padre_1)
            cruzados.append(padre_2)
    if tam_poblacion%2 == 1:
        cruzados.append(padres[0])
    return np.array(cruzados)

def cruce_parcialmente_mapeado(genotipo_1, genotipo_2, posiciones):
    # Segmentamos
    hijo = np.zeros(len(genotipo_1)) - 1
    mascara_segmento = (np.arange(len(genotipo_1)) >= posiciones[0]) & (np.arange(len(genotipo_1)) <= posiciones[1])
    hijo[mascara_segmento] = genotipo_1[mascara_segmento]

    # Lo que no forma parte del segmento
    sobrante = genotipo_1[mascara_segmento==False]
    segmento_2 = genotipo_2[mascara_segmento]

    # Separamos en conjunto dentro y fuera del segmento del genotipo 2
    pertenencia = np.isin(sobrante, segmento_2)
    con_conflicto = sobrante[pertenencia]
    sin_conflicto = np.sort(sobrante[pertenencia==False])
    
    # Añadimos los elementos sin conflicto (que no están dentro del segmento del genotipo 2)
    indices_sin_conflicto = np.where(np.isin(genotipo_2,sin_conflicto))
    hijo[indices_sin_conflicto] = sin_conflicto

    # Tratamos conflicto
    for elem in con_conflicto:
        posicion = elem.copy()
        while(posicion != -1):
            genotipo_en_posicion = posicion
            posicion = hijo[np.where(genotipo_2 == genotipo_en_posicion)][0]
        hijo[int(np.where(genotipo_2 == genotipo_en_posicion)[0])] = elem
    return hijo

def mutacion(pmutacion, tam_poblacion, poblacion, matriz_distancias, nciudades):
    mascara_mutacion = np.random.random(tam_poblacion) <= pmutacion
    poblacion[mascara_mutacion] = mutacion_por_intercambio(poblacion[mascara_mutacion], matriz_distancias, nciudades)
    return np.sort(poblacion)
    

def mutacion_por_intercambio(individuos, matriz_distancias, nciudades):
    mutados = []
    for indiv in individuos:
        indices = np.random.choice(indiv.genotipo, 2, replace=False)
        genotipo = indiv.genotipo.copy()
        genotipo[int(indices[0])], genotipo[int(indices[1])] = genotipo[int(indices[1])], genotipo[int(indices[0])]
        mutados.append(Individuo(genotipo, fitness(genotipo, matriz_distancias, nciudades)))
    return np.array(mutados)


def seleccion_hijos(mejor_padre, mutados, eliminacionelitismo, tam_poblacion):
    # Minimizamos, por lo que el padre es mejor si es <= al menor hijo
    if mejor_padre.fitness <= mutados[0].fitness:
        if eliminacionelitismo == "peor":
            return np.insert(mutados[1:], 0, mejor_padre, axis=0)
        elif eliminacionelitismo == "aleatorio":
            posicion = np.random.randint(tam_poblacion)
            # Sustituye el mejor padre por el hijo que ocupa una posicion aleatoria y ordena por mejor fitness
            return np.sort(np.insert(np.delete(mutados, posicion, 0), posicion, mejor_padre, axis=0))
    else:
        return mutados
    

def fitness(indiv, matriz_distancias, nciudades):
    distancia = 0
    for i in range(nciudades):
        if i==(nciudades-1):
            distancia += matriz_distancias[int(indiv[i])][int(indiv[0])]
        else:
            distancia += matriz_distancias[int(indiv[i])][int(indiv[i+1])]
    return distancia

def generar_grafica(evolucion_fitness, array_bars, pasos_intervalos, ngen, file_name, robustez_parametro=False, x_vector=None):
    fig = plt.figure()
    #x = np.arange(1, (ngen/pasos_intervalos))
    if not robustez_parametro:
        x = np.arange(ngen)
    else:
        x = x_vector
    #plt.plot(evolucion_fitness)
    plt.errorbar(x, evolucion_fitness, yerr=array_bars, errorevery=pasos_intervalos, capsize=3.0, ecolor='black')
    plt.ylabel("Fitness")
    if not robustez_parametro:
        plt.xlabel("Generaciones")
        plt.title("Curva de progreso")
    else:
        plt.xlabel(robustez_parametro)
        plt.title(f'Robustez frente a cambios de valor en {robustez_parametro}')
    plt.ticklabel_format(axis='y', style="sci", scilimits=None)
    plt.savefig(file_name)
    plt.show()


def algoritmo_genetico(params):
    ciudades, matriz_distancias = cargar_ciudades("./ciudades.csv")

    # Generación de población inicial
    poblacion = init_poblacion(params["tam_poblacion"], matriz_distancias, params["nciudades"])

    if params["verbose"]:
        print("Estado Inicial: \n")
        print(f'Fitness mejor individuo: {poblacion[0].fitness} \n\n')

    t_init = time.time()
    display_timer = time.time()

    evolucion_fitness = []

    # Iteraciones en el orden de ngen
    for i in range(int(params["ngen"])):
        i_poblacion = generacion(params, poblacion.copy(), matriz_distancias)
        evolucion_fitness.append(i_poblacion[0].fitness)
        poblacion = i_poblacion.copy()
        if params["verbose"] and time.time() - display_timer >= params["display_time"]:
            print(f'Generacion: {i}\n Tiempo: {(time.time()-t_init):0.3} \n Fitness mejor individuo: {poblacion[0].fitness}\n\n')
            display_timer = time.time()

    return evolucion_fitness, poblacion

def algoritmo_genetico_paralelo(params):
    # Solo como wraper porque la poblacion de Individuos no se puede juntar en el pool de resultados
    evolucion_fitness, _ = algoritmo_genetico(params)
    return evolucion_fitness

def algoritmo_genetico_comparacion_parametros(params, value, comp_param_name):
    params[comp_param_name] = value
    evolucion_fitness, _ = algoritmo_genetico(params)
    return evolucion_fitness[-1]

def main_progreso():
    # Lectura de parámetros
    f_params = open("params.json")
    params = json.load(f_params)
    f_params.close()

    # Inicialización de ciudades en función de parámetros
    init_ciudades(params["nciudades"], params["radio"])

    delayed_funcs = [delayed(algoritmo_genetico_paralelo)(params.copy()) for i in range(params["niter"])]
    parallel_pool = Parallel(n_jobs=params["niter"]) 
    resultado = parallel_pool(delayed_funcs)

    evolucion_fitness_iter = np.reshape(resultado, (params["niter"],params["ngen"]))

    print(evolucion_fitness_iter[0][:100])
    mean_evolucion_fitness = evolucion_fitness_iter.mean(axis=0)
    std_evolucion_fitness = evolucion_fitness_iter.std(axis=0)*(sp.stats.norm.isf((1-params["conf"])/2)/np.sqrt(params["niter"]))

    generar_grafica(mean_evolucion_fitness, std_evolucion_fitness, params["pasos_intervalos"], int(params["ngen"]), params["file_name"])


def main_parametro():
    # Lectura de parámetros
    f_params = open("params.json")
    params = json.load(f_params)
    f_params.close()

    # Inicialización de ciudades en función de parámetros
    init_ciudades(params["nciudades"], params["radio"])

    comp_param_vector = np.linspace(params["comp_param_inicio"], params["comp_param_final"], params["comp_param_pasos"])
    delayed_funcs = [delayed(algoritmo_genetico_comparacion_parametros)(params.copy(), i, params["comp_param_name"]) for i in np.sort(np.repeat(comp_param_vector,params["niter"]))]
    parallel_pool = Parallel(n_jobs=params["niter"]) 
    resultado = parallel_pool(delayed_funcs)

    evolucion_fitness_iter = np.reshape(resultado, (len(comp_param_vector),params["niter"])).T

    mean_evolucion_fitness = evolucion_fitness_iter.mean(axis=0)
    std_evolucion_fitness = evolucion_fitness_iter.std(axis=0)*(sp.stats.norm.isf((1-params["conf"])/2)/np.sqrt(params["niter"]))

    generar_grafica(mean_evolucion_fitness, std_evolucion_fitness, params["pasos_intervalos"], int(params["ngen"]), params["file_name"], robustez_parametro=params["comp_param_name"], x_vector=comp_param_vector)


if __name__ == "__main__":
    #main_progreso()
    main_parametro()
