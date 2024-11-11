import pandas as pd
import numpy as np
from config import *
from constants import *
from basic_funcs import *

from clasificacion.red_neuronal import *
from clasificacion.bayes import * 

def k_fold(k_partitions, clasificador):
    data = load_data(file_path)
    X, y = seperate_target(data, target_column) 

    #---SEPARACION---
    X_particiones = np.array_split(X, k_partitions)
    y_particiones = np.array_split(y, k_partitions)


    #----CLASIFICADORES-----
    metrics_list = []
    counter = 0
    
    if clasificador == NEURAL_NETWORK:
        model_list =[]
        for _ in range(k_partitions):
            model_list.append(build_model(X_particiones[0].shape[1]))

        for X_test, y_test in zip(X_particiones[1:-1], y_particiones[1:-1]): #ciclar las particiones de entranamientos
            counter += 1
            print(f"Iteracion: {counter}")
            metrics_list.append(train_and_evaluate(model_list[counter], X_particiones[0], X_test, y_particiones[0], y_test, NO, YES, NO)) 
        metrics_list.append(train_and_evaluate(model_list[-1], X_particiones[0], X_particiones[-1], y_particiones[0], y_particiones[-1], YES, YES, YES))
        metrics_print(*metrics_list[-1]) #last iteration metrics
    
    elif clasificador == BAYES: 
        for X_test, y_test in zip(X_particiones[1:], y_particiones[1:]): #ciclar las particiones de entranamientos
            counter += 1
            print(f"Iteracion: {counter}")
            metrics_list.append(bayes(X_particiones[0], X_test, y_particiones[0], y_test, NO))


    #----METRICS----
    results_list = []
    for metrics in metrics_list:
        results_list.append(metrics_no_print(*metrics))
    results_suma_list = [0, 0, 0, 0, 0]
    for results in results_list:
        results_suma_list[0] += results[0]
        results_suma_list[1] += results[1]
        results_suma_list[2] += results[2]
        results_suma_list[3] += results[3]
        results_suma_list[4] += results[4]
    if not feature_selection_f:
        print("-----Normal dataset:----")
    else:
        print("-----Feature selection-----")
    print(f"\nDespues de {k_partitions} iteraciones:")
    print(f"Exactitud promedio: {results_suma_list[0] / len(results_list):.2f}%")
    print(f"Tasa de error promedio: {results_suma_list[1] / len(results_list):.3f}")
    print(f"Precision promedio: {results_suma_list[2] / len(results_list):.2f}%")
    print(f"Recuerdo promedio: {results_suma_list[3] / len(results_list):.2f}%")
    print(f"Puntaje F1 promedio: {results_suma_list[4] / len(results_list):.2f}%")