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

dataValidation = dataset.TurbDataset(prop,
                           dataDir="/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices/train/",
                           dataDirTest="/home/laura/exclude_backup/gyroids/sdf_velocity_dP_slices/test/",
                           shuffle=1,
                           mode=2)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))


targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 2, 128, 128))

criterionL1 = utils.CustomWeightedL1Loss(0.0, sdf_threshold=0.0)
loss_vector = []
loss_dict = {
    10: [],
    25: [],
    50: [],
    100: [],
    200: [],
    210: []
}

dp = []

for i, validata in enumerate(valiLoader, 0):
    inputs_cpu, targets_cpu = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()

    lossL1 = criterionL1(outputs, targets, inputs[:, :1, :, :])
    loss_vector.append(lossL1.item())
    original_dp = inputs[:, 1, :, :] * dataValidation.max_inputs_1
    dp.append(np.unique(original_dp)[0])
    loss_dict.get(int(np.unique(original_dp)[0])).append(lossL1.item())


argmin_loss = np.argmin(loss_vector)
argmax_loss = np.argmax(loss_vector)
arg_med_loss = np.argsort(loss_vector)[len(loss_vector)//2]

stats_idx = {argmin_loss: "min",
             argmax_loss: "max",
             arg_med_loss: "median"}

validation_dir = "validation_masked_unmasked"
utils.makeDirs([validation_dir])

for i, validata in enumerate(valiLoader, 0):
    if i not in stats_idx.keys():
        continue
    print(stats_idx.get(i), "error on index {}".format(i), "with value ", loss_vector[i])
    inputs_cpu, targets_cpu = validata
    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()
    input_ndarray = inputs_cpu.cpu().numpy()[0]
    v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :]))**2) ** 0.5

    outputs_denormalized = dataValidation.denormalize(outputs_cpu[0], v_norm)
    targets_denormalized = dataValidation.denormalize(targets_cpu.cpu().numpy()[0], v_norm)

    utils.save_true_pred_img(os.path.join(validation_dir, stats_idx.get(i) + "_err_pred"),
                             outputs_denormalized,
                             targets_denormalized,
                             input_ndarray[0].reshape(128, 128),
                             smoothing=False)


plt.hist(loss_vector)
plt.axvline(loss_vector[arg_med_loss], color="r")
plt.legend(["median L1 error: {median:.5f}".format(median=loss_vector[arg_med_loss])])
plt.savefig(os.path.join(validation_dir, "loss_hist.png"))
plt.close()


for k in loss_dict.keys():
    loss = loss_dict.get(k)
    plt.hist(loss)
    plt.legend(["median L1 error: {median:.5f}".format(median=np.median(loss))])
    plt.savefig(os.path.join(validation_dir, "loss_hist_{}.png".format(str(k).zfill(3))))
    plt.close()
