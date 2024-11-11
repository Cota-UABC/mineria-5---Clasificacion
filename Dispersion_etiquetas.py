import pandas as pd
import matplotlib.pyplot as plt
from nombres_columnas import obtener_columnas_csv

from config import *

predict_f = 1

if predict_f:
    TARGET_CLASS = 'Predicted'
    color_map = 'viridis'
else: 
    TARGET_CLASS = 'Actual'
    color_map = 'winter'

#indice
indice_1 = 0
#indice_2 = 2 #patch

df = pd.read_csv(out_file_out)

lista_columnas = obtener_columnas_csv(out_file_out)

x = df[lista_columnas[indice_1]]
y = df['Sweetness'] #patch
clase = df[TARGET_CLASS]


plt.scatter(x, y, c=clase, cmap=color_map) 
plt.xlabel(lista_columnas[indice_1])
plt.ylabel('Sweetness') #patch
plt.title(f'Gráfico de Dispersión según etiqueta {TARGET_CLASS}')
plt.show()
