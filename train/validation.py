import os.path

import dataset
from torch.utils.data import DataLoader
import torch
from DfpNet import TurbNetG, weights_init
from torch.autograd import Variable
import numpy as np
import utils
import matplotlib.pyplot as plt

## Load model
dropout = 0.
expo = 3
netG = TurbNetG(channelExponent=expo, dropout=dropout)
netG.load_state_dict(torch.load("modelG"))
netG.eval()

prop = None
batch_size = 1

data = dataset.TurbDataset(prop, shuffle=1)
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))


targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 1, 128, 128))

criterionL1 = utils.CustomWeightedL1Loss()
loss_vector = []

for i, validata in enumerate(valiLoader, 0):
    inputs_cpu, targets_cpu = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()

    lossL1 = criterionL1(outputs, targets, inputs)
    loss_vector.append(lossL1.item())


argmin_loss = np.argmin(loss_vector)
argmax_loss = np.argmax(loss_vector)
arg_med_loss = np.argsort(loss_vector)[len(loss_vector)//2]

stats_idx = {argmin_loss: "min",
             argmax_loss: "max",
             arg_med_loss: "median"}

validation_dir = "validation"
utils.makeDirs([validation_dir])

for i, validata in enumerate(valiLoader, 0):
    if i not in stats_idx.keys():
        continue

    inputs_cpu, targets_cpu = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()
    input_ndarray = inputs_cpu.cpu().numpy()[0]
    v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2) ** 0.5

    outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
    targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)

    utils.save_true_pred_img(os.path.join(validation_dir, stats_idx.get(i) + "_err_pred"), outputs_denormalized, targets_denormalized)


plt.hist(loss_vector)
plt.axvline(loss_vector[arg_med_loss], color="r")
plt.legend(["median L1 error: {median:.5f}".format(median=loss_vector[arg_med_loss])])
plt.savefig(os.path.join(validation_dir, "loss_hist.png"))
