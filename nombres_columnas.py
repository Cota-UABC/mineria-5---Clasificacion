import pandas as pd

def obtener_columnas_csv(ruta_csv):
    df = pd.read_csv(ruta_csv)

    columnas = df.columns.tolist()
    
    return columnas

#ruta_csv = 'datasets/wine.csv'  # Cambia esto por la ruta de tu archivo CSV
#columnas = obtener_columnas_csv(ruta_csv)
#print(columnas)
