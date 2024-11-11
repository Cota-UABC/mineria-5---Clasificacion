import pandas as pd
from sklearn.feature_selection import f_classif
from config import *
from basic_funcs import *

p_threshold = 0.05

data = pd.read_csv(file_path)

X, y = seperate_data(data, target_column)

f_values, p_values = f_classif(X, y)

anova_results = pd.DataFrame({'Feature': X.columns, 'F_value': f_values, 'p_value': p_values})

significant_features = anova_results[anova_results['p_value'] < p_threshold]['Feature']

print("ANOVA:")
print(anova_results)

print("Features selecionados:")
print(significant_features)

data_significant = data[significant_features.tolist() + [target_column]]
data_significant.to_csv(file_path_FS, index=False)
print(f"Archivo con features selecionados guardado: {file_path_FS}")