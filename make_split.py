import json
import os

import numpy as np
from matplotlib import pyplot as plt

data_train = os.listdir("/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices_reshaped/train")
data_test = os.listdir("/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices_reshaped/test")

train_dict = {"train": [d.split(".")[0] for d in data_train]}
test_dict = {"test": [d.split(".")[0] for d in data_test]}

with open('/home/laura/exclude_backup/pycharm_projects_deployment/Deep-Flow-Prediction/splits/gyroids_slice_train.json', 'w') as outfile:
    json.dump(train_dict, outfile)

with open('/home/laura/exclude_backup/pycharm_projects_deployment/Deep-Flow-Prediction/splits/gyroids_slice_test.json', "w") as outfile:
    json.dump(test_dict, outfile)


train_pairs = []
for filename in train_dict.get("train"):
    th = int(filename.split("_")[1])
    dp = int(filename.split("_")[3])
    train_pairs.append([th, dp])

test_pairs = []
for filename in test_dict.get("test"):
    th = int(filename.split("_")[1])
    dp = int(filename.split("_")[3])
    test_pairs.append([th, dp])

train_pairs = np.unique(np.array(train_pairs), axis=0)
test_pairs = np.unique(np.array(test_pairs), axis=0)

plt.scatter(train_pairs[:, 0], train_pairs[:, 1])
plt.scatter(test_pairs[:, 0], test_pairs[:, 1], s=30, c="r")
plt.xlabel("thickness")
plt.ylabel("dP")
plt.savefig("gyroid_slice_train_test.png")

