import os

import andi
import numpy as np

AD = andi.andi_datasets()


X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N = 60000, tasks = 2, dimensions = 3)

project_directory = os.getcwd()
path_to_save = os.path.join(project_directory, "Data","datasets", "dataset_Andi_3")
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


X = X2[2]
Y = Y2[2]


np.save(os.path.join(path_to_save, "X.npy"), X)
np.save(os.path.join(path_to_save, "Y.npy"), Y)