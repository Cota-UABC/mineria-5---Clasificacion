import pandas as pd
import numpy as np
from config import *
from constants import *
from basic_funcs import *

from clasificacion.red_neuronal import *
from clasificacion.bayes import * 

def random_sampling(clasificador):
	data = load_data(file_path)
	X, y = seperate_target(data, target_column) 

    #---SEPARACION---
	split_data_list = []
	for _ in range(k_iterations):
		split_data_list.append(split_data(X, y, test_size))


	#----CLASIFICADORES-----
	metrics_list = []
	counter = 0
	if clasificador == NEURAL_NETWORK:
		for split_d in split_data_list[:-1]:
			counter += 1
			print(f"Iteracion: {counter}")
			model = build_model(split_d[0].shape[1]) #X_train shape
			metrics_list.append(train_and_evaluate(model, *split_d, NO, YES, NO))
		model = build_model(split_d[0].shape[1]) 
		metrics_list.append(train_and_evaluate(model, *split_data_list[-1], YES, YES, YES))
		
		metrics_print(*metrics_list[-1]) #last iteration metrics
	
	elif clasificador == BAYES:
		for split_d in split_data_list:
			counter += 1
			print(f"Iteracion: {counter}")
			metrics_list.append(bayes(*split_d, NO))


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
	print(f"\nDespues de {k_iterations} iteraciones:")
	print(f"Exactitud promedio: {results_suma_list[0] / len(results_list):.2f}%")
	print(f"Tasa de error promedio: {results_suma_list[1] / len(results_list):.3f}")
	print(f"Precision promedio: {results_suma_list[2] / len(results_list):.2f}%")
	print(f"Recuerdo promedio: {results_suma_list[3] / len(results_list):.2f}%")
	print(f"Puntaje F1 promedio: {results_suma_list[4] / len(results_list):.2f}%")