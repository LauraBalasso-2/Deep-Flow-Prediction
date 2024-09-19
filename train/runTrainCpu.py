################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script
#
################

import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from uNet_architecture import UNet, weights_init
import dataset
import utils

######## Settings ########

# number of training iterations
iterations = 10000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
prop = None  # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = False

##########################

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout = 0.  # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = 0 # random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# create pytorch data object with dfp dataset
data = dataset.SlicesDataset(prop,
                             dataDir="/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices/train/",
                             dataDirTest= "/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices/test/",
                             shuffle=0)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=True, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = 1000  # int(iterations / len(trainLoader) + 0.5)
netG = UNet(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = utils.CustomWeightedL1Loss(0.0, sdf_threshold=0.0001)  # nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 2, 128, 128))

##########################

train_loss_list = []
val_loss_list = []
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        # compute LR decay

        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)
        print(f"Generator output: {gen_out.shape}")
        print(f"Target output: {targets_cpu.shape}")

        lossL1 = criterionL1(gen_out, targets, inputs[:, :1, :, :])
        print("Loss L1: {}".format(lossL1))

        loss_comparison_x = criterionL1(gen_out[:, 0:1, :, :], targets[:, 0:1, :, :], inputs[:, :1, :, :]).detach().numpy()
        loss_comparison_y = criterionL1(gen_out[:, 1:2, :, :], targets[:, 1:2, :, :], inputs[:, :1, :, :]).detach().numpy()
        loss_comparison_z = criterionL1(gen_out[:, 2:3, :, :], targets[:, 2:3, :, :], inputs[:, :1, :, :]).detach().numpy()
        print(f"Loss comparison x: {loss_comparison_x}")
        print(f"Loss comparison y: {loss_comparison_y}")
        print(f"Loss comparison z: {loss_comparison_z}")

        print("Loss Comparison: {}".format(np.mean([loss_comparison_x, loss_comparison_y, loss_comparison_z])))

        exit()
        lossL1.backward()

        optimizerG.step()

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
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets, inputs[:, :1, :, :])
        L1val_accum += lossL1.item()

        if epoch % 50 == 0 and i == 0:
            input_ndarray = inputs_cpu.cpu().numpy()[0]
            v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :]))**2) ** 0.5
            outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
            targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)

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
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)


plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.xlabel("Epoch")
plt.legend(["Train", "Val"])
plt.savefig(prefix + "losses.png", dpi=120)
torch.save(netG.state_dict(), prefix + "modelG")
