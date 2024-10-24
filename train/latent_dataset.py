import json
import os
import torch

from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random

from utils import set_device


def load_normalization_parameters(experiment_directory):
    with open(os.path.join(experiment_directory, "normalization_parameters.json"), "r") as f:
        normalization_parameters = json.load(f)
    return normalization_parameters


class SlicesDataset(Dataset):
    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST = 2

    def __init__(self, dataDir, latent_codes_dir, device, split=None,  mode=TRAIN,  shuffle=0, normalization_parameters=None):
        """
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing  data
        """
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - TurbDataset invalid mode " + format(mode))
            exit(1)

        self.mode = mode
        self.shuffle = shuffle
        self.dataDir = dataDir
        self.files = listdir(self.dataDir) if split is None else self.get_files_list(split)
        self.totalLength = len(self.files)
        self.latent_codes, self.latent_codes_thickness = self.load_train_validation_encoding(latent_codes_dir)

        self.inputs = np.empty((self.totalLength, 2, 128, 128))
        self.targets = np.empty((self.totalLength, 4, 128, 128))
        self.thicknesses = np.empty(self.totalLength)
        self.slice_indexes = np.empty(self.totalLength)
        self.device = device
        self.load_data()

        self.normalization_parameters = normalization_parameters if normalization_parameters is not None \
            else self.get_normalization_parameters()

        self.normalize()

        if not self.mode == self.TEST:
            print("Splitting 80/20")
            # split for train/validation sets (80/20)
            targetLength = self.totalLength - int(self.totalLength * 0.2)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength
            self.valiLatent_codes = self.latent_codes[targetLength:]

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]
            self.latent_codes = self.latent_codes[:targetLength]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.latent_codes[idx]

    def load_latent_codes(self, latent_codes_dir):
        thickness_idx = json.load(open(os.path.join(latent_codes_dir, "rows_thickness.json")))
        latent_vectors_path = os.path.join(latent_codes_dir, "latest.pth")
        data = torch.load(latent_vectors_path,  map_location=self.device if self.device=="cpu" else None)
        if isinstance(data["latent_codes"], torch.Tensor):
            num_vecs = data["latent_codes"].size()[0]
            lat_vecs = []
            for i in range(num_vecs):
                lat_vecs.append((set_device(data["latent_codes"][i], self.device)))
            return lat_vecs, thickness_idx

        else:
            num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
            lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
            lat_vecs.load_state_dict(data["latent_codes"])
            return lat_vecs.weight.data.detach(), thickness_idx

    def load_train_validation_encoding(self, encoding_dir):
        latent_codes_train, latent_codes_thickness_train = self.load_latent_codes(os.path.join(encoding_dir, "LatentCodes"))
        latent_codes_validation, latent_codes_thickness_validation = self.load_latent_codes(os.path.join(encoding_dir, "LatentCodes_validation"))

        return np.row_stack(( latent_codes_train.detach().cpu(),  torch.cat(latent_codes_validation).detach().cpu())), latent_codes_thickness_train + latent_codes_thickness_validation

    def load_data(self):
        # load single directory
        files = self.files
        for i in range(self.shuffle):
            random.shuffle(files)

        print("Loading {:d} training files from {:s} ...".format(self.totalLength, self.dataDir))
        latent_codes = np.empty_like(self.latent_codes)
        for i, file in enumerate(files):
            np_file = np.load(os.path.join(self.dataDir, file))
            d = np_file['a']
            self.inputs[i] = d[0:2]
            self.targets[i] = d[2:]
            self.targets[i, -1, :, :] /= np.unique(self.inputs[i, 1, :, :])[0]
            sample_name = file.split("/")[-1].split(".")[0]
            self.thicknesses[i] = int(sample_name.split("_")[1])
            self.slice_indexes[i] = int(sample_name.split("_")[-1])
            latent_codes[i] = self.latent_codes[self.latent_codes_thickness.index(str(int(self.thicknesses[i])).zfill(3))]

        self.latent_codes = latent_codes
        print("Number of training data:", len(self.inputs))

    def get_normalization_parameters(self):
        print("Getting normalization parameters from loaded data")
        max_inputs_0 = np.max(np.abs(self.inputs[:, 0, :, :]))
        max_inputs_1 = np.max(np.abs(self.inputs[:, 1, :, :]))
        print("Max training sdf = {:.5f}; Max training dP =  {:.1f}".format(max_inputs_0, max_inputs_1))

        max_targets_0 = np.max(np.abs(self.targets[:, 0, :, :]))
        max_targets_1 = np.max(np.abs(self.targets[:, 1, :, :]))
        max_targets_2 = np.max(np.abs(self.targets[:, 2, :, :]))
        print("Maxima training targets " + format([max_targets_0, max_targets_1, max_targets_2]))

        return {"max_input_0": max_inputs_0, "max_input_1": max_inputs_1,
                "max_target_0": max_targets_0, "max_target_1": max_targets_1, "max_target_2": max_targets_2}

    def normalize(self):
        self.inputs[:, 0, :, :] *= (1.0 / self.normalization_parameters.get("max_input_0"))
        self.inputs[:, 1, :, :] *= (1.0 / self.normalization_parameters.get("max_input_1"))

        self.targets[:, 0, :, :] *= (1.0 / self.normalization_parameters.get("max_target_0"))
        self.targets[:, 1, :, :] *= (1.0 / self.normalization_parameters.get("max_target_1"))
        self.targets[:, 2, :, :] *= (1.0 / self.normalization_parameters.get("max_target_2"))

        print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % (
            np.mean(np.abs(self.inputs), keepdims=False), np.max(np.abs(self.inputs), keepdims=False),
            np.mean(np.abs(self.targets), keepdims=False), np.max(np.abs(self.targets), keepdims=False)))

    def denormalize(self, data, deltaP):
        a = data.copy()
        a[0, :, :] *= self.normalization_parameters.get("max_target_0")
        a[1, :, :] *= self.normalization_parameters.get("max_target_1")
        a[2, :, :] *= self.normalization_parameters.get("max_target_2")
        a[3, :, :] *= deltaP

        return a

    def save_normalization_parameters(self, experiment_directory):
        with open(os.path.join(experiment_directory, "normalization_parameters.json"), "w") as f:
            json.dump(self.normalization_parameters, f)

    def get_files_list(self, split):
        npz_files = []
        for dataset in split:
            for instance_name in sorted(split[dataset]):
                instance_filename = os.path.join(dataset, instance_name + ".npz")
                if not os.path.isfile(os.path.join(self.dataDir, instance_filename)):
                    print("Requested non-existent file '{}'".format(instance_filename))
                npz_files.append(instance_filename)
        return npz_files


class ValiDataset(Dataset):
    def __init__(self, dataset):
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength
        self.latent_codes = dataset.valiLatent_codes

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.latent_codes[idx]
