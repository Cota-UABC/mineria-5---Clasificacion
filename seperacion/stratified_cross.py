import pandas as pd
import numpy as np
from config import *
from constants import *
from basic_funcs import *
from sklearn.model_selection import StratifiedKFold

from clasificacion.red_neuronal import *

def startfield(k_partitions_strat, clasificador):
    data = load_data(file_path)
    X, y = seperate_target(data, target_column)
    
    #---SEPARACION---
    skf = StratifiedKFold(n_splits=k_partitions_strat, shuffle=True)

    metrics_list = []
    counter = 0

    for train_index, test_index in skf.split(X, y):
        counter += 1
        print(f"Iteracion: {counter}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #----CLASIFICADORES-----
        if clasificador == NEURAL_NETWORK:
            if counter != k_partitions_strat:
                model = build_model(X_train.shape[1]) 
                metrics_list.append(train_and_evaluate(model, X_train, X_test, y_train, y_test, NO, YES, NO))
            else:
                model = build_model(X_train.shape[1]) #last iteration
                metrics_list.append(train_and_evaluate(model, X_train, X_test, y_train, y_test, YES, YES, YES))

    #----METRICS----
    metrics_print(*metrics_list[-1]) #last iteration metrics

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
    print(f"\nDespues de {k_partitions_strat} iteraciones:")
    print(f"Exactitud promedio: {results_suma_list[0] / len(results_list):.2f}%")
    print(f"Tasa de error promedio: {results_suma_list[1] / len(results_list):.3f}")
    print(f"Precision promedio: {results_suma_list[2] / len(results_list):.2f}%")
    print(f"Recuerdo promedio: {results_suma_list[3] / len(results_list):.2f}%")
    print(f"Puntaje F1 promedio: {results_suma_list[4] / len(results_list):.2f}%")