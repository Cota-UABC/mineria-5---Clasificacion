import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from config import *
from basic_funcs import save_predictions

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Para clasificación multiclase

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test, plot_f, matrix, file_f):
    history = model.fit(X_train, y_train, epochs=epocas, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    #print(f"Loss: {loss}, Accuracy: {accuracy}")

    if plot_f:
        plt.plot(history.history['accuracy'], label='Accuracy de Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Accuracy de Validación')
        plt.title('Accuracy por época')
        plt.xlabel('Épocas')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)  # Convertir las predicciones a 0 o 1 (clasificación binaria)

    if matrix:
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        positives = tp + fn  
        negatives = tn + fp
    else: 
        tn, fp, fn, tp, positives, negatives = 0,0,0,0,0,0 

    # Guardar las predicciones en un archivo CSV
    if file_f:
        save_predictions(X_test, y_test, predictions)

    return accuracy, tp, fp, tn, fn, positives, negatives


