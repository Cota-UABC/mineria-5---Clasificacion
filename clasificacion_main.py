import pandas as pd
import numpy as np
from config import *
from constants import *
from basic_funcs import *

from seperacion.hold_out import *
from seperacion.random_sampling import *
from seperacion.k_fold import *
from seperacion.leave_one_out import *
from seperacion.stratified_cross import *

from clasificacion.red_neuronal import *


if __name__ == "__main__":

    if hold_out_f:
        hold_out(test_size, clasificador)
    elif random_s:
        random_sampling(clasificador)
    elif k_fold_f:
        k_fold(k_partitions, clasificador)
    elif leave_one_f:
        leave_one(clasificador)
    elif strat_f:
        startfield(k_partitions_strat, clasificador)
    else:
        print("No se elegio un metodo")
