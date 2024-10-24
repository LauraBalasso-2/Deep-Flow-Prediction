import json
import os.path
import argparse

from torch import nn

import encoding_slices_dataset
from torch.utils.data import DataLoader
import torch
from decoder_architecture import Decoder
from torch.autograd import Variable
import numpy as np
import utils

import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description="U-Net validation")
arg_parser.add_argument(
    "--device",
    dest="device",
    default="cpu",
    help="Device on which the training is performed. Must be cpu or gpu."
)
arg_parser.add_argument(
    "--experiment",
    "-e",
    dest="experiment_directory",
    required=True,
    help="The experiment directory. This directory should include "
         + "experiment specifications in 'specs.json', and logging will be "
         + "done in this directory as well.",
)

args = arg_parser.parse_args()

device = args.device
experiment_directory = args.experiment_directory

specs = utils.load_experiment_specifications(experiment_directory)

batch_size = 1

norm_params = encoding_slices_dataset.load_normalization_parameters(experiment_directory=experiment_directory)
with open(specs["test_split"], "r") as f:
    test_split = json.load(f)
dataValidation = encoding_slices_dataset.LatentSlicesDataset(dataDir=specs["data_source"],
                                                             shuffle=0,
                                                             latent_codes_dir=specs["latent_code_dir"],
                                                             mode=2,
                                                             split=test_split,
                                                             device=device,
                                                             normalization_parameters=norm_params)

valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

## Load model
dropout = specs['dropout']
netG = Decoder(latent_size=dataValidation.latent_size)
netG.load_state_dict(torch.load(os.path.join(experiment_directory, "model_D")))
netG.eval()

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, dataValidation.latent_size + 2, 1, 1))

criterionL1 = utils.CustomWeightedL1Loss(0.0, sdf_threshold=0.0)
loss_vector = []
loss_x = []
loss_y = []
loss_z = []

for i, validata in enumerate(valiLoader, 0):
    inputs_cpu, targets_cpu, sdf = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs, sdf.float())
    outputs_cpu = outputs.data.cpu().numpy()

    lossL1_x = criterionL1(outputs[:, 0:1, :, :], targets[:, 0:1, :, :], sdf).item()
    lossL1_y = criterionL1(outputs[:, 1:2, :, :], targets[:, 1:2, :, :], sdf).item()
    lossL1_z = criterionL1(outputs[:, 2:3, :, :], targets[:, 2:3, :, :], sdf).item()

    loss_vector.append(np.mean([lossL1_x, lossL1_y, lossL1_z]))
    loss_x.append(lossL1_x)
    loss_y.append(lossL1_y)
    loss_z.append(lossL1_z)
    original_dp = inputs[:, 1, :, :] * norm_params.get("max_input_dp")

argmin_loss = np.argmin(loss_vector)
argmax_loss = np.argmax(loss_vector)
arg_med_loss = np.argsort(loss_vector)[len(loss_vector) // 2]

stats_idx = {argmin_loss: "min",
             argmax_loss: "max",
             arg_med_loss: "median"}

validation_dir = os.path.join(experiment_directory, "validation")
utils.makeDirs([validation_dir])

for i, validata in enumerate(valiLoader, 0):
    if i not in stats_idx.keys():
        continue
    print(stats_idx.get(i), "error on index {}".format(i), "with value ", loss_vector[i])
    inputs_cpu, targets_cpu, sdf = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs, sdf.float())
    outputs_cpu = outputs.data.cpu().numpy()
    dp = dataValidation.inputs[i, -2, 0, 0]

    outputs_denormalized = dataValidation.denormalize(outputs_cpu[0], dp)
    targets_denormalized = dataValidation.denormalize(targets_cpu.cpu().numpy()[0], dp)

    utils.save_true_pred_img(os.path.join(validation_dir, stats_idx.get(i) + "_err_pred"),
                             outputs_denormalized,
                             targets_denormalized,
                             sdf.reshape(128, 128),
                             smoothing=False)


def plot_error_hist(loss_list, median_idx, save_path):
    plt.hist(loss_list)
    plt.axvline(loss_list[median_idx], color="r")
    plt.legend(["median L1 error: {median:.5f}".format(median=loss_list[median_idx])])
    plt.savefig(save_path)
    plt.close()


plot_error_hist(loss_vector, arg_med_loss, os.path.join(validation_dir, "loss_hist.png"))
plot_error_hist(loss_x, median_idx=np.argsort(loss_x)[len(loss_x) // 2],
                save_path=os.path.join(validation_dir, "loss_hist_Ux.png"))
plot_error_hist(loss_y, median_idx=np.argsort(loss_y)[len(loss_y) // 2],
                save_path=os.path.join(validation_dir, "loss_hist_Uy.png"))
plot_error_hist(loss_z, median_idx=np.argsort(loss_z)[len(loss_z) // 2],
                save_path=os.path.join(validation_dir, "loss_hist_Uz.png"))
