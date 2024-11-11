from constants import *

feature_selection_f = 1
if feature_selection_f:
    file_path = 'datasets/apple_FS.csv'
else:
    file_path = "datasets/apple.csv"

out_file_out = "datasets/apple_predc.csv"

target_column = "Quality" 


#------SEPARADORES--------
#Hold out
test_size = 0.4

#Random subsampling
k_iterations = 30

#K-fold
k_partitions = 10

#startfield
k_partitions_strat = 10


#------CLASIFICADORES-------
#Red neuronal
epocas = 200


#----seleccion de separador-----
hold_out_f = 0
random_s = 0
k_fold_f = 0
leave_one_f = 1
strat_f = 0

#----selecion de clasificador-----
clasificador = BAYES