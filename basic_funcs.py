import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def seperate_target(data, target_column):
    X = data.drop(target_column, axis=1)  # Características
    y = data[target_column]  # Etiquetas

    # Normalizar características
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    # Si la variable objetivo es categórica, convertir a formato one-hot
    #if data[target_column].dtype == 'object':
    #    y = pd.get_dummies(y)
    #else:
    #    y = to_categorical(y)

    return X, y

def split_data(X, y, test_sz):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz)
    return X_train, X_test, y_train, y_test

def metrics_print(accuracy, tp, fp, tn, fn, positives, negatives):

    matriz = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=['Clase Positiva Real', 'Clase Negativa Real'],
        columns=['Predicción Positiva', 'Predicción Negativa']
    )

    # Mostrar la matriz y los totales de positivos y negativos
    print("Matriz de Confusión:")
    print(matriz)
    print("\nTotales:")
    print(f"Positivos : {positives}")
    print(f"Negativos : {negatives}\n")

    if not accuracy:
        accuracy = (tp + tn) / (positives + negatives)
    accuracy = accuracy*100
    print(f"Exactitud: {accuracy:.2f}%")

    error_rate = (fp + fn) / (positives + negatives)
    print(f"Tasa de Error: {error_rate:.3f}") 

    precision = (tp / (tp + fp)) * 100
    print(f"Precision: {precision:.2f}%") 

    recall = (tp / positives) * 100
    print(f"Recuerdo: {recall:.2f}%") 

    f1_score = ((precision * recall) / (precision + recall)) * 2
    print(f"Puntaje F1: {f1_score:.2f}%") 

def metrics_no_print(accuracy, tp, fp, tn, fn, positives, negatives):
    #print(f"\nac:{accuracy} tp:{tp} fp:{fp} tn:{tn} fn{fn} pos:{positives} neg:{negatives}")

    if not accuracy:
        try:
            accuracy = (tp + tn) / (positives + negatives)
        except ZeroDivisionError:
            accuracy = 0
    accuracy = accuracy*100

    #print(accuracy)

    try:
        error_rate = (fp + fn) / (positives + negatives)
    except ZeroDivisionError:
        error_rate = 0

    try:
        precision = (tp / (tp + fp)) * 100
    except ZeroDivisionError:
        precision = 0

    try:
        recall = (tp / positives) * 100
    except ZeroDivisionError:
        recall = 0
        
    try:
        f1_score = ((precision * recall) / (precision + recall)) * 2
    except ZeroDivisionError:
        f1_score = 0

    return accuracy, error_rate, precision, recall, f1_score

def save_predictions(X_test, y_test, predictions):
    results = X_test.copy()
    results['Actual'] = y_test
    results['Predicted'] = predictions

    # Guardar el DataFrame en un archivo CSV
    results.to_csv(out_file_out, index=False)
    print(f"Archivo {out_file_out} guardado con éxito.")