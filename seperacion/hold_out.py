import pandas as pd
import numpy as np
from config import *
from constants import *
from basic_funcs import *

from clasificacion.red_neuronal import * 
from clasificacion.bayes import * 



def hold_out(test_size, clasificador):
    data = load_data(file_path)
    X, y = seperate_target(data, target_column) 

    #---SEPARACION---
    if test_size == 1:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = split_data(X, y, test_size)


    #----CLASIFICADORES-----
    if clasificador == NEURAL_NETWORK:
        model = build_model(X_train.shape[1]) 
        accuracy, tp, fp, tn, fn, positives, negatives = train_and_evaluate(model, X_train, X_test, y_train, y_test, YES, YES, YES)
        
    elif clasificador == BAYES: 
        accuracy, tp, fp, tn, fn, positives, negatives = bayes(X_train, X_test, y_train, y_test, YES)  

    #----METRICS----
    metrics_print(accuracy, tp, fp, tn, fn, positives, negatives)