import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nombres_columnas import obtener_columnas_csv
from config import *

def normalizar_columnas_csv(file_path, columnas_a_normalizar, ruta_salida_csv_normalizado):
    df = pd.read_csv(file_path)
    
    #print("Datos originales:")
    print(df.head())
    
    scaler = MinMaxScaler()

    df[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])
    
    print("\nDatos normalizados:")
    print(df.head())

    df.to_csv(ruta_salida_csv_normalizado, index=False)
  

lista_columnas = obtener_columnas_csv(file_path)
lista_columnas.pop() # drop last column

normalizar_columnas_csv(file_path, lista_columnas, ruta_salida_csv_normalizado)
