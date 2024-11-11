import pandas as pd
import numpy as np
from config import *
from constants import *
from basic_funcs import *

from clasificacion.red_neuronal import *
from clasificacion.bayes import * 

def leave_one(clasificador):
	data = load_data(file_path)
	X, y = seperate_target(data, target_column) 

	num_filas = data.shape[0]

	#---SEPARACION---
	X_dataframes = []
	y_dataframes = []
	for i in range(len(X)):
		loo_df = X.drop(index=i).reset_index(drop=True)
		X_dataframes.append(loo_df)
		loo_df = y.drop(index=i).reset_index(drop=True)
		y_dataframes.append(loo_df)
	

	#----CLASIFICADORES-----
	metrics_list = []
	counter = 0

	if clasificador == NEURAL_NETWORK:
		for X_test, y_test in zip(X_dataframes[1:-1], y_dataframes[1:-1]): #ciclar las particiones de entranamientos
			counter += 1
			print(f"Iteracion: {counter}")
			model = build_model(X_dataframes[0].shape[1])
			metrics_list.append(train_and_evaluate(model, X_dataframes[0], X_test, y_dataframes[0], y_test, NO, YES, NO)) 
		
		model = build_model(X_dataframes[0].shape[1]) #last iteration
		metrics_list.append(train_and_evaluate(model, X_dataframes[0], X_dataframes[-1], y_dataframes[0], y_dataframes[-1], YES, YES, YES)) 
		metrics_print(*metrics_list[-1]) #last iteration metrics
	
	if clasificador == BAYES:
		for X_test, y_test in zip(X_dataframes[1:], y_dataframes[1:]): #ciclar las particiones de entranamientos
			counter += 1
			print(f"Iteracion: {counter}")
			metrics_list.append(bayes(X_dataframes[0], X_test, y_dataframes[0], y_test, NO))

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
	print(f"\nDespues de {num_filas} iteraciones:")
	print(f"Exactitud promedio: {results_suma_list[0] / len(results_list):.2f}%")
	print(f"Tasa de error promedio: {results_suma_list[1] / len(results_list):.3f}")
	print(f"Precision promedio: {results_suma_list[2] / len(results_list):.2f}%")
	print(f"Recuerdo promedio: {results_suma_list[3] / len(results_list):.2f}%")
	print(f"Puntaje F1 promedio: {results_suma_list[4] / len(results_list):.2f}%")