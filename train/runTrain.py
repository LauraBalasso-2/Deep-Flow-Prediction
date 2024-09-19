################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script
#
################

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset
import utils
from uNet_architecture import UNet, weights_init

arg_parser = argparse.ArgumentParser(description="Train U-Net")
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

batch_size = specs['batch_size']
lrG = specs['learning_rate']
decayLr = specs['lr_decay']
expo = specs["unet_channel_exponent"]
saveL1 = False

dropout = specs["dropout"]
doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Dropout: {}".format(dropout))

##########################

seed = specs["random_seed"]
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic=True # warning, slower

# create pytorch data object with dfp dataset
data = dataset.SlicesDataset(dataDir="/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices/train/",
                             dataDirTest="/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices/test/",
                             shuffle=0)

trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = specs["epochs"]
netG = utils.set_device(UNet(channelExponent=expo, dropout=dropout), device=device)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized UNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = utils.set_device(utils.CustomWeightedL1Loss(0.0, sdf_threshold=0.0001), device=device)

optimizer = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = utils.set_device(Variable(torch.FloatTensor(batch_size, 3, 128, 128)), device=device)
inputs = utils.set_device(Variable(torch.FloatTensor(batch_size, 2, 128, 128)), device=device)

##########################

train_loss_list = []
val_loss_list = []
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs_cpu = utils.set_device(inputs_cpu.float(), device)
        targets_cpu = utils.set_device(targets_cpu.float(), device)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizer.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)

        lossL1 = criterionL1(gen_out, targets, inputs[:, :1, :, :])
        lossL1.backward()

        optimizer.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        inputs_cpu = utils.set_device(inputs_cpu.float(), device)
        targets_cpu = utils.set_device(targets_cpu.float(), device)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets, inputs[:, :1, :, :])
        L1val_accum += lossL1.item()

        if epoch % 50 == 0 and i == 0:
            input_ndarray = inputs_cpu.cpu().numpy()[0]
            outputs_denormalized = data.denormalize(outputs_cpu[0])
            targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0])

            utils.makeDirs(["results_train"])
            # utils.imageOut("results_train/epoch{}_{}".format(epoch, i), outputs_cpu[0], targets_cpu.cpu().numpy()[0],
            #                saveTargets=True)
            utils.save_true_pred_img("results_train/epoch{}_{}".format(epoch, i),
                                     outputs_denormalized,
                                     targets_denormalized,
                                     input_ndarray[0].reshape(128, 128))

    # data for graph plotting
    L1_accum /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    train_loss_list.append(L1_accum)
    val_loss_list.append(L1val_accum)
    if saveL1:
        if epoch == 0:
            utils.resetLog(os.path.join(experiment_directory, "L1.txt"))
            utils.resetLog(os.path.join(experiment_directory, "L1val.txt"))
        utils.log(os.path.join(experiment_directory, "L1.txt", "{} ".format(L1_accum)), False)
        utils.log(os.path.join(experiment_directory, "L1val.txt", "{} ".format(L1val_accum)), False)

plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.xlabel("Epoch")
plt.legend(["Train", "Val"])
plt.savefig(os.path.join(experiment_directory, "losses.png"), dpi=120)
torch.save(netG.state_dict(), os.path.join(experiment_directory, "model_U"))
