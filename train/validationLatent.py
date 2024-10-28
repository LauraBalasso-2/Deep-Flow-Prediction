import json
import os.path
import argparse

import latent_dataset as dataset
from torch.utils.data import DataLoader
import torch
from uNet_latent_architecture import UNet
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

## Load model
dropout = specs['dropout']
expo = specs["unet_channel_exponent"]


batch_size = 1
with open(specs["test_split"], "r") as f:
    test_split = json.load(f)
norm_params = dataset.load_normalization_parameters(experiment_directory=experiment_directory)
dataValidation = dataset.SlicesDataset(dataDir=specs["data_source"],
                                       split=test_split,
                                       shuffle=0,
                                       mode=2,
                                       normalization_parameters=norm_params)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

netG = UNet(channelExponent=expo, dropout=dropout, latent_size=dataValidation.latent_size)
netG.load_state_dict(torch.load(os.path.join(experiment_directory, "model_U")))
netG.eval()

targets = Variable(torch.FloatTensor(batch_size, 4, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 2, 128, 128))

criterionL1 = utils.CustomWeightedL1Loss(0.0, sdf_threshold=0.0)
loss_vector = []
loss_x = []
loss_y = []
loss_z = []
loss_p = []

for i, validata in enumerate(valiLoader, 0):
    inputs_cpu, targets_cpu = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()

    lossL1_x = criterionL1(outputs[:, 0:1, :, :], targets[:, 0:1, :, :], inputs[:, :1, :, :]).item()
    lossL1_y = criterionL1(outputs[:, 1:2, :, :], targets[:, 1:2, :, :], inputs[:, :1, :, :]).item()
    lossL1_z = criterionL1(outputs[:, 2:3, :, :], targets[:, 2:3, :, :], inputs[:, :1, :, :]).item()
    lossL1_p = criterionL1(outputs[:, 3:4, :, :], targets[:, 3:4, :, :], inputs[:, :1, :, :]).item()

    loss_vector.append(np.mean([lossL1_x, lossL1_y, lossL1_z, lossL1_p]))
    loss_x.append(lossL1_x)
    loss_y.append(lossL1_y)
    loss_z.append(lossL1_z)
    loss_p.append(lossL1_p)

argmin_loss_ux = np.argmin(loss_x)
argmax_loss_ux = np.argmax(loss_x)
arg_med_loss_ux = np.argsort(loss_x)[len(loss_x) // 2]

argmin_loss_uy = np.argmin(loss_y)
argmax_loss_uy = np.argmax(loss_y)
arg_med_loss_uy = np.argsort(loss_y)[len(loss_y) // 2]

argmin_loss_uz = np.argmin(loss_z)
argmax_loss_uz = np.argmax(loss_z)
arg_med_loss_uz = np.argsort(loss_z)[len(loss_z) // 2]

argmin_loss_p = np.argmin(loss_p)
argmax_loss_p = np.argmax(loss_p)
arg_med_loss_p = np.argsort(loss_p)[len(loss_p) // 2]

arg_med_loss = np.argsort(loss_vector)[len(loss_vector) // 2]

stats_idx = {"ux": {"min": (argmin_loss_ux, loss_x[argmin_loss_ux]),
                    "max": (argmin_loss_ux, loss_x[argmin_loss_ux]),
                    "median": (arg_med_loss_ux, loss_x[arg_med_loss_ux])},
             "uy": {"min": (argmin_loss_uy, loss_y[argmin_loss_uy]),
                    "max": (argmax_loss_uy, loss_y[argmax_loss_uy]),
                    "median": (arg_med_loss_uy, loss_y[arg_med_loss_uy])},
             "uz": {"min": (argmin_loss_uz, loss_z[argmin_loss_uz]),
                    "max": (argmax_loss_uz, loss_z[argmax_loss_uz]),
                    "median": (arg_med_loss_uz, loss_z[arg_med_loss_uz])},
             "p": {"min": (argmin_loss_p, loss_p[argmin_loss_p]),
                   "max": (argmax_loss_p, loss_p[argmax_loss_p]),
                   "median": (arg_med_loss_p, loss_p[arg_med_loss_p])}}

validation_dir = os.path.join(experiment_directory, "validation")
utils.makeDirs([validation_dir])

for field, stats in stats_idx.items():
    for stat, (i, val) in stats.items():
        inputs_cpu, targets_cpu = dataValidation[i]
        print(stat, "error on index {}, with val {:.4f}".format(i, val))
        outputs = netG(torch.from_numpy(inputs_cpu.reshape(batch_size, -1, 128, 128)).float())
        outputs_cpu = outputs.data.cpu().numpy()

        dp = dataValidation.inputs[i, 1, 0, 0]

        outputs_denormalized = dataValidation.denormalize(outputs_cpu[0], deltaP=dp)
        targets_denormalized = dataValidation.denormalize(targets_cpu, deltaP=dp)

        utils.save_single_field_true_pred_img(os.path.join(validation_dir, stat + "_err_pred_"),
                                              outputs_denormalized,
                                              targets_denormalized,
                                              field_name=field,
                                              _input=inputs_cpu[0].reshape(128, 128))


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
plot_error_hist(loss_p, median_idx=np.argsort(loss_p)[len(loss_p) // 2],
                save_path=os.path.join(validation_dir, "loss_hist_p.png"))
