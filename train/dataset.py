################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Dataset handling
#
################

from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random


# compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


def LoaderNormalizer(data, isTest=False, shuffle=0):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    """

    # load single directory
    files = listdir(data.dataDir)
    files.sort()
    for i in range(shuffle):
        random.shuffle(files)

    data.totalLength = len(files)
    data.inputs = np.empty((len(files), 2, 128, 128))
    data.targets = np.empty((len(files), 3, 128, 128))
    data.thicknesses = np.empty(len(files))
    data.slice_indexes = np.empty(len(files))

    print("Loading {:d} training files from {:s} ...".format(data.totalLength, data.dataDir))

    for i, file in enumerate(files):
        npfile = np.load(data.dataDir + file)
        d = npfile['a']
        data.inputs[i] = d[0:2]
        data.targets[i] = d[2:5]
        data.thicknesses[i] = int(file.split("_")[1])
        data.slice_indexes[i] = int(file.split("_")[-1].split(".")[0])
    print("Number of training data:", len(data.inputs))

    data.max_inputs_0 = find_absmax(data, 0, 0)
    data.max_inputs_1 = find_absmax(data, 0, 1)
    print("Max training sdf = {:.5f}; Max training dP =  {:.1f}".format(data.max_inputs_0, data.max_inputs_1))

    data.max_targets_0 = find_absmax(data, 1, 0)
    data.max_targets_1 = find_absmax(data, 1, 1)
    data.max_targets_2 = find_absmax(data, 1, 2)
    print("Maxima training targets " + format([data.max_targets_0, data.max_targets_1, data.max_targets_2]))

    if not isTest:
        print("Building Training Dataset ..")

        data.inputs[:, 0, :, :] *= (1.0 / data.max_inputs_0)
        data.inputs[:, 1, :, :] *= (1.0 / data.max_inputs_1)

        data.targets[:, 0, :, :] *= (1.0 / data.max_targets_0)
        data.targets[:, 1, :, :] *= (1.0 / data.max_targets_1)
        data.targets[:, 2, :, :] *= (1.0 / data.max_targets_2)

    else:
        files = listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        print("Loading {:d} test files from {:s} ...".format(data.totalLength, data.dataDirTest))
        data.inputs = np.empty((len(files), 2, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))
        data.thicknesses = np.empty(len(files))
        data.slice_indexes = np.empty(len(files))
        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            d = npfile['a']
            data.inputs[i] = d[0:2]
            data.targets[i] = d[2:5]
            data.thicknesses[i] = int(file.split("_")[1])
            data.slice_indexes[i] = int(file.split("_")[-1].split(".")[0])

        print("Building Test Dataset ..")
        data.inputs[:, 0, :, :] *= (1.0 / data.max_inputs_0)
        data.inputs[:, 1, :, :] *= (1.0 / data.max_inputs_1)

        data.targets[:, 0, :, :] *= (1.0 / data.max_targets_0)
        data.targets[:, 1, :, :] *= (1.0 / data.max_targets_1)
        data.targets[:, 2, :, :] *= (1.0 / data.max_targets_2)

    print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % (
        np.mean(np.abs(data.inputs), keepdims=False), np.max(np.abs(data.inputs), keepdims=False),
        np.mean(np.abs(data.targets), keepdims=False), np.max(np.abs(data.targets), keepdims=False)))

    return data


class SlicesDataset(Dataset):
    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST = 2

    def __init__(self, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0):
        """
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        """
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - TurbDataset invalid mode " + format(mode))
            exit(1)

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest  # only for mode==self.TEST

        # load & normalize data
        self = LoaderNormalizer(self, isTest=(mode == self.TEST), shuffle=shuffle)

        if not self.mode == self.TEST:
            print("Splitting 80/20")
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - int(self.totalLength * 0.2)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    #  reverts normalization 
    def denormalize(self, data):
        a = data.copy()
        a[0, :, :] /= (1.0 / self.max_targets_0)
        a[1, :, :] /= (1.0 / self.max_targets_1)
        a[2, :, :] /= (1.0 / self.max_targets_2)

        return a


class ValiDataset(SlicesDataset):
    def __init__(self, dataset):
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
