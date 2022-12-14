import numpy as np
import pandas as pd 
import json
import scipy as sp
import time
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

"""
  Clase que repesenta la estructura de los individuos
  que formarán parte de la población. Estos individuos 
  tendrán su genotipo (solución) asociaciado y el valor
  de su fitness. Esto se hace con el objetivo de no volver
  a calcular el fitness nuevamente y evitar evaluaciones
  innecesarias. El ndarray de la población estará compuesto
  por objetos de esta clase.
  Por otro lado, la clase también nos sirve para poder fijar
  un criterio de comparación de soluciones (__lt__).
"""
class Individuo:
    def __init__(self, genotipo, fitness):
        self.genotipo = genotipo
        self.fitness = fitness
    # Ordenamos de menor a mayor los Individuos por su fitness
    def __lt__(self, other):
        return self.fitness < other.fitness

"""
 |---------------------------------------------------|
 | init_ciudades                                     |
 |---------------------------------------------------|
 | Función que inicializa la disposición de las      |
 | ciudades dada la cantidad de ciudades y el radio  |
 | de la circunferencia inscrita dentro del cuadrado |
 | que delimita el area en el que están las ciudades.|
 |___________________________________________________|
 | int, int ->                                       |
 |___________________________________________________|
 | Entrada:                                          |
 | nciudades: numero de ciudades a inicializar.      |
 | radio: radio que define el area en el que se      |
 |        encuentran las ciudades.                   |
 |___________________________________________________|
 | Salida:                                           |
 | no hay salida, solo se crea un csv con la         |
 | disposición de las ciudades.                      |
 |---------------------------------------------------|
"""
def init_ciudades(nciudades, radio):
    # Generamos nciudades en coordenadas cartesianas desde -radio a +radio para x e y
    ciudades = (np.random.random([nciudades,2]) * radio*2) - radio
    np.savetxt("ciudades.csv", ciudades, delimiter=",")

"""
 |---------------------------------------------------|
 | cargas_ciudades                                   |
 |---------------------------------------------------|
 | Función que lee las ciudades de un .csv, las carga|
 | en una matriz y calcula la matriz de distancias.  |
 |___________________________________________________|
 | string -> ndarray, ndarray                        |
 |___________________________________________________|
 | Entrada:                                          |
 | path: string que indica el archivo a leer.        |
 |___________________________________________________|
 | Salida:                                           |
 | ciudades: ndarray que contiene las ciudades. Esto |
 |           significa que la ciudad que ocupa la    |
 |           posición 0 en este ndarray será la que  |
 |           esté representada como 0 en los array de|
 |           permutaciones, y así para cada ciudad.  |
 | distance_matrix: matriz de distancias entre las   |
 |                  ciudades. Se usará para calcular |
 |                  la función de fitness.           |
 |---------------------------------------------------|
"""
def cargar_ciudades(path):
    ciudades = np.loadtxt(path, delimiter=",")
    # p es la p-norma de Minkowski. p=1 distancia rectilínea, p=2 distancia euclidiana
    return (ciudades, sp.spatial.distance_matrix(ciudades, ciudades, p=2))

"""
 |---------------------------------------------------|
 | init_poblacion                                    |
 |---------------------------------------------------|
 | Función que inicializa la población del algoritmo |
 | genético, dado un tamaño y número de ciudades.    |
 |___________________________________________________|
 | int, ndarray, int -> ndarray                      |
 |___________________________________________________|
 | Entrada:                                          |
 | tam_poblacion: tamaño de la población de genotipos|
 |                sobre la que se operará.           |
 | matriz_distancias: matriz de distancias entre las |
 |                    ciudades.                      |
 | nciudades: número de ciudades. Esto define también|
 |            el tamaño que tendrá el genotipo.      |
 |___________________________________________________|
 | Salida:                                           |
 | población: ndarray que contiene la población de   |
 |            genotipos sobre la que se trabajará.   |
 |---------------------------------------------------|
"""
def init_poblacion(tam_poblacion, matriz_distancias, nciudades):
    # Generamos los genotipos de la poblacion por pormutaciones aleatorias
    genotipos_poblacion = [np.random.permutation(nciudades) for i in range(tam_poblacion)]
    # Generamos la poblacion a partir de los genotipos y sus fitness
    poblacion = np.array([Individuo(genotipo, fitness(genotipo, matriz_distancias, nciudades)) for genotipo in genotipos_poblacion])
    return np.sort(poblacion)

"""
 |---------------------------------------------------|
 | generacion                                        |
 |---------------------------------------------------|
 | Función que sintetiza el flujo del algoritmo      |
 | genético. El algoritmo está compuesto por varias  |
 | generaciones o pasos. Cada generación está, a su  |
 | vez, compuesta por una selección de padres, un    |
 | cruce, una mutación y una selección de hijos.     |
 |___________________________________________________|
 | dict, ndarray, ndarray -> ndarray                 |
 |___________________________________________________|
 | Entrada:                                          |
 | params: un diccionario cargado de params.json que |
 |         contiene todos los parámetros necesarios. |
 | población: ndarray que contiene la población de   |
 |            genotipos al inicio de la generación.  |
 | matriz_distancias: matriz de distancias entre las |
 |                    ciudades.                      |
 |___________________________________________________|
 | Salida:                                           |
 | hijos: ndarray que contiene la población de       |
 |        genotipos tras realizar la generación.     |
 |---------------------------------------------------|
"""
def generacion(params, poblacion, matriz_distancias): # STEP
    padres = seleccion_padres(params["npadres"], params["tam_poblacion"], poblacion.copy())
    cruzados = cruce(params["pcruce"], params["tam_poblacion"], params["nciudades"], padres.copy(), matriz_distancias)
    mutados = mutacion(params["pmutacion"], params["tam_poblacion"], cruzados.copy(), matriz_distancias, params["nciudades"])
    hijos = seleccion_hijos(poblacion[:params["nelitismo"]], mutados.copy(), params["eliminacionelitismo"], params["tam_poblacion"], params["nelitismo"])
    return hijos

"""
 |---------------------------------------------------|
 | seleccion_padres                                  |
 |---------------------------------------------------|
 | Función que lleva a cabo la selección de padres   |
 | por torneo.                                       |
 |___________________________________________________|
 | int, int, ndarray -> ndarray                      |
 |___________________________________________________|
 | Entrada:                                          |
 | npadres: numero de padres que se seleccionan de la|
 |          población por cada torneo.               |
 | tam_poblacion: tamaño de la población de genotipos|
 |                sobre la que se operará.           |
 | población: ndarray que contiene la población de   |
 |            genotipos al inicio de la generación.  |
 |___________________________________________________|
 | Salida:                                           |
 | padres: ndarray que contiene la población de      |
 |         genotipos tras la seleción de padres.     |
 |---------------------------------------------------|
"""
def seleccion_padres(npadres, tam_poblacion, poblacion):
    padres = []
    for i_torneo in range(tam_poblacion):
        padres_elegidos = poblacion[np.random.choice(tam_poblacion, int(npadres))]
        ind_padre = np.argmin(padres_elegidos)
        padres.append(padres_elegidos[ind_padre])
    return np.array(padres)

"""
 |---------------------------------------------------|
 | cruce                                             |
 |---------------------------------------------------|
 | Función que elige los 2 padres a cruzar, hace las |
 | preparaciones pertinentes, llama al cruce que     |
 | corresponda si se da la probabilidad de curce y   |
 | gestiona la creación de la población cruzada.     |
 |___________________________________________________|
 | int, int, int, ndarray, ndarray -> ndarray        |
 |___________________________________________________|
 | Entrada:                                          |
 | pcruce: probabilidad de cruce.                    |
 | tam_poblacion: tamaño de la población de genotipos|
 |                sobre la que se operará.           |
 | nciudades: número de ciudades o, lo que es lo     |
 |            mismo, el tamaño del genotipo.         |
 | padres: ndarray que contiene la población de      |
 |            genotipos que se cruzará.              |
 | matriz_distancias: matriz de distancias entre las |
 |                    ciudades.                      |
 |___________________________________________________|
 | Salida:                                           |
 | cruzados: ndarray que contiene la población de    |
 |           genotipos tras el cruce.                |
 |---------------------------------------------------|
"""
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

"""
 |---------------------------------------------------|
 | cruce_parcialmente_mapeado                        |
 |---------------------------------------------------|
 | Función que lleva a cabo el cruce parcialmente    |
 | mapeado sobre 2 genotipos y unas posiciones dadas.|
 |___________________________________________________|
 | ndarray, ndarray, ndarray -> ndarray              |
 |___________________________________________________|
 | Entrada:                                          |
 | genotipo_1: genotipo que hará de padre 1.         |
 | genotipo_2: genotipo que hará de padre 2.         |
 | posiciones: posiciones que definen el segmento.   |
 |___________________________________________________|
 | Salida:                                           |
 | hijo: el hijo 1 producido por el curce.           |
 |---------------------------------------------------|
"""
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

"""
 |---------------------------------------------------|
 | mutacion                                          |
 |---------------------------------------------------|
 | Función que prepara la mutación sobre un conjunto |
 | de genotipos seleccionados de la población.       |
 |___________________________________________________|
 | int, int, ndarray, ndarray, int -> ndarray        |
 |___________________________________________________|
 | Entrada:                                          |
 | pmutacion: probabilidad de que un genotipo de la  |
 |            población sea mutado.                  |
 | tam_poblacion: tamaño de la población de genotipos|
 |                sobre la que se operará.           |
 | población: ndarray que contiene la población de   |
 |            genotipos al inicio de la generación.  |
 | matriz_distancias: matriz de distancias entre las |
 |                    ciudades.                      |
 | nciudades: número de ciudades o, lo que es lo     |
 |            mismo, el tamaño del genotipo.         |
 |___________________________________________________|
 | Salida:                                           |
 | poblacion: la población tras haber realizado las  |
 |            mutaciones correspondientes.           |
 |---------------------------------------------------|
"""
def mutacion(pmutacion, tam_poblacion, poblacion, matriz_distancias, nciudades):
    mascara_mutacion = np.random.random(tam_poblacion) <= pmutacion
    poblacion[mascara_mutacion] = mutacion_por_intercambio(poblacion[mascara_mutacion], matriz_distancias, nciudades)
    return np.sort(poblacion)

"""
 |---------------------------------------------------|
 | mutacion_por_intercambio                          |
 |---------------------------------------------------|
 | Función que lleva a cabo la mutación en concreto, |
 | en este caso, por intercambio.                    |
 |___________________________________________________|
 | ndarray, ndarray, int -> ndarray                  |
 |___________________________________________________|
 | Entrada:                                          |
 | individuos: ndarray de los genotipos a los que se |
 |             aplicará la mutación.                 |
 | matriz_distancias: matriz de distancias entre las |
 |                    ciudades.                      |
 | nciudades: número de ciudades o, lo que es lo     |
 |            mismo, el tamaño del genotipo.         |
 |___________________________________________________|
 | Salida:                                           |
 | mutados: la población tras haber realizado las    |
 |            mutaciones por intercambio.            |
 |---------------------------------------------------|
"""
def mutacion_por_intercambio(individuos, matriz_distancias, nciudades):
    mutados = []
    for indiv in individuos:
        indices = np.random.choice(indiv.genotipo, 2, replace=False)
        genotipo = indiv.genotipo.copy()
        genotipo[int(indices[0])], genotipo[int(indices[1])] = genotipo[int(indices[1])], genotipo[int(indices[0])]
        mutados.append(Individuo(genotipo, fitness(genotipo, matriz_distancias, nciudades)))
    return np.array(mutados)

"""
 |---------------------------------------------------|
 | seleccion_hijos                                   |
 |---------------------------------------------------|
 | Función que selecciona los hijos mediante un      |
 | modelo generacional con eletismo para el mejor de |
 | los padres según una estrategia definida.         |
 |___________________________________________________|
 | ndarray, ndarray, string, int -> ndarray          |
 |___________________________________________________|
 | Entrada:                                          |
 | mejor_padre: genotipo del mejor padre.            |
 | mutados: ndarray de los genotipos a los que se    |
 |          aplicará la selección.                   |
 | eliminacionelismo: tipo de eliminacion que se     |
 |                    llevará a cabo en los hijos.   |
 | tam_poblacion: tamaño de la población de genotipos|
 |                sobre la que se operará.           |
 |___________________________________________________|
 | Salida:                                           |
 | mutados: la población tras haber realizado las    |
 |            mutaciones por intercambio.            |
 |---------------------------------------------------|
"""
def seleccion_hijos(mejores_padres, mutados, eliminacionelitismo, tam_poblacion, nelitismo=1):
    # Minimizamos, por lo que el padre es mejor si es <= al menor hijo
    if nelitismo==1:
        if mejores_padres.fitness <= mutados[0].fitness:
            if eliminacionelitismo == "peor":
                return np.insert(mutados[1:], 0, mejores_padres, axis=0)
            elif eliminacionelitismo == "aleatorio":
                posicion = np.random.randint(tam_poblacion)
                # Sustituye el mejor padre por el hijo que ocupa una posicion aleatoria y ordena por mejor fitness
                return np.sort(np.insert(np.delete(mutados, posicion, 0), posicion, mejores_padres, axis=0))
        else:
            return mutados
    else:
        mask_padres = mejores_padres < mutados[0]
        if len(mask_padres[mask_padres==1]) > 0:
            elite_padres = mejores_padres[mask_padres]
            if eliminacionelitismo == "peor":
                seleccionados = np.concatenate((elite_padres, mutados[:-len(elite_padres)]),axis=0)
                return seleccionados
            elif eliminacionelitismo == "aleatorio":
                posicion = np.random.choice(np.arange(tam_poblacion),nelitismo,replace=False)
                mutados[posicion] = elite_padres
                return np.sort(mutados)
        else:
            return mutados



"""
 |---------------------------------------------------|
 | fitness                                           |
 |---------------------------------------------------|
 | Función a optimizar por parte del algoritmo. En   |
 | este caso la optimización trata de minimizar. Es  |
 | habitual hacer uso de un factor -1 para convertir |
 | la minimización en maximización o viceversa.      |
 | Necesario para la creación de un Individuo.       |
 |___________________________________________________|
 | ndarray, ndarray, int -> float                    |
 |___________________________________________________|
 | Entrada:                                          |
 | indiv: genotipo cuyo fitness se desea evaluar.    |
 | matriz_distancias: matriz de distancias entre las |
 |                    ciudades.                      |
 | nciudades: número de ciudades o, lo que es lo     |
 |            mismo, el tamaño del genotipo.         |
 |___________________________________________________|
 | Salida:                                           |
 | distancia: distancia entre las ciudades recorridas|
 |            según la dirección del genotipo.       |
 |---------------------------------------------------|
"""
def fitness(indiv, matriz_distancias, nciudades):
    distancia = 0
    for i in range(nciudades):
        if i==(nciudades-1):
            distancia += matriz_distancias[int(indiv[i])][int(indiv[0])]
        else:
            distancia += matriz_distancias[int(indiv[i])][int(indiv[i+1])]
    return distancia

"""
 |---------------------------------------------------|
 | generar_grafica                                   |
 |---------------------------------------------------|
 | Función que genera las gráficas necesarias para el|
 | issue #7, esto es, para la generación de los      |
 | resultados dados unos experimentos. Originalmente |
 | se diseñó para la curva de progreso, por lo que   |
 | son este tipo de gráficas las que genera por      |
 | defecto. Con unos flags se puede cambiar el tipo  |
 | de gráfica.                                       |
 |___________________________________________________|
 | ndarray, ndarray, int, int, string,               |
 |    bool/string, ndarray ->                        |
 |___________________________________________________|
 | Entrada:                                          |
 | evolucion_fitness: array que contiene los valores |
 |                    que ha tomado el fitness en    |
 |                    cada generación.               |
 | array_bars: array que contiene el valor de los    |
 |             intervalos de confianza.              |
 | pasos_intervalos: cada cuantas generaciones se    |
 |                   muestran los intervalos.        |
 | ngen: número de generaciones.                     |
 | file_name: nombre del archivo donde se guardará la|
 |            gráfica generada.                      |
 | robustez_parametro: flag por defecto False. Este  |
 |             implica que por defecto se generará la|
 |             curva de progreso. En caso de no ser  |
 |             False, contendrá el nombre del        |
 |             parámetro respecto al cual queremos   |
 |             analizar la robustez del algoritmo. En|
 |             ese caso se generará una gráfica de   |
 |             robustez frente a los cambios de un   |
 |             parámetro.                            |
 | x_vector: solo en caso de gráfica de robustez.    |
 |           Contiene los valores que toma el        |
 |           parámetro en este análisis.             |
 |___________________________________________________|
 | Salida:                                           |
 | no produce salida, guarda en file_name la gráfica |
 | que se ha generado.                               |
 |---------------------------------------------------|
"""
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

"""
 |---------------------------------------------------|
 | algoritmo_genetico                                |
 |---------------------------------------------------|
 | Función que hace de wrapper y gestora del proceso |
 | que lleva a cabo el algoritmo genético. Se encarga|
 | de llamar a las funciones necesarias definidas    |
 | anteriormente.                                    |
 | Como algoritmo genético contiene una selección de |
 | padres, cruce, mutación y selección de hijos, pero|
 | podría no ser el caso. Por ello, es en generación |
 | donde se especifica el orden de y qué funciones   |
 | se utilizan.                                      |
 |___________________________________________________|
 | dict -> ndarray, ndarray                          |
 |___________________________________________________|
 | Entrada:                                          |
 | params: diccionario que contiene todos los        |
 |         parámetros nnecesarios para el algoritmo. |
 |         Estos se cargan previamente de un ".json".|
 |___________________________________________________|
 | Salida:                                           |
 | evolucion_fitness: array con los valores que toma |
 |                    el fitness en cada generación. |
 | población: la población final resultante.         |
 |---------------------------------------------------|
"""
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

"""
 |---------------------------------------------------|
 | algoritmo_genetico_paralelo                       |
 |---------------------------------------------------|
 | Wrapper al algorimo_genetico que permite la       |
 | utilización de Parallel de joblib. Esto reduce el |
 | tiempo de cómputo del algoritmo.                  |
 |___________________________________________________|
 | dict -> ndarray                                   |
 |___________________________________________________|
 | Entrada:                                          |
 | params: diccionario que contiene todos los        |
 |         parámetros nnecesarios para el algoritmo. |
 |         Estos se cargan previamente de un ".json".|
 |___________________________________________________|
 | Salida:                                           |
 | evolucion_fitness: array con los valores que toma |
 |                    el fitness en cada generación. |
 |---------------------------------------------------|
"""
def algoritmo_genetico_paralelo(params):
    # Solo como wraper porque la poblacion de Individuos no se puede juntar en el pool de resultados
    evolucion_fitness, _ = algoritmo_genetico(params)
    return evolucion_fitness

"""
 |---------------------------------------------------|
 | algoritmo_genetico_comparacion_parametros         |
 |---------------------------------------------------|
 | Wrapper al algorimo_genetico que permite la       |
 | utilización de Parallel de joblib para el caso    |
 | concreto de análisis de la robustez.              |
 |___________________________________________________|
 | dict -> float                                     |
 |___________________________________________________|
 | Entrada:                                          |
 | params: diccionario que contiene todos los        |
 |         parámetros nnecesarios para el algoritmo. |
 |         Estos se cargan previamente de un ".json".|
 |___________________________________________________|
 | Salida:                                           |
 | evolucion_fitness[-1]: valor del fitness en la    |
 |                        última generación.         |
 |---------------------------------------------------|
"""
def algoritmo_genetico_comparacion_parametros(params, value, comp_param_name):
    params[comp_param_name] = value
    evolucion_fitness, _ = algoritmo_genetico(params)
    return evolucion_fitness[-1]

"""
 |---------------------------------------------------|
 | main_progreso                                     |
 |---------------------------------------------------|
 | Función main para generar la curva de progreso    |
 | tras un número de ejecuciones, todo ello, al igual|
 | que lo demás, especificado por unos parámetros.   |
 | Hace las llamadas necesarias a algoritmo_genetico |
 | y trata de la forma indicada sus salidas.         |
 |___________________________________________________|
 |  ->                                               |
 |___________________________________________________|
 | Entrada:                                          |
 |___________________________________________________|
 | Salida:                                           |
 | no hay, se genera gráfica de progreso y se guarda.|
 |---------------------------------------------------|
"""
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

"""
 |---------------------------------------------------|
 | main_robustez                                     |
 |---------------------------------------------------|
 | Función main para generar la robustez frente a    |
 | cambios en un parámetro. Todo está especificado en|
 | un ".json".
 | Hace las llamadas necesarias a algoritmo_genetico |
 | y trata de la forma indicada sus salidas.         |
 |___________________________________________________|
 |  ->                                               |
 |___________________________________________________|
 | Entrada:                                          |
 |___________________________________________________|
 | Salida:                                           |
 | no hay, se genera gráfica de robustez y se guarda.|
 |---------------------------------------------------|
"""
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
    main_progreso()
    #main_parametro()
