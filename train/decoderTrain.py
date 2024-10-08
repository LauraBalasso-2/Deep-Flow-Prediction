import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import encoding_slices_dataset
import utils
from decoder_architecture import Decoder
from uNet_architecture import weights_init

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

doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
seed = specs["random_seed"]
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic=True # warning, slower

sys.stdout.flush()

with open(specs["train_split"], "r") as f:
    train_split = json.load(f)
data = encoding_slices_dataset.LatentSlicesDataset(dataDir=specs["data_source"],
                                                   latent_codes_dir=specs["latent_code_dir"],
                                                   split=train_split,
                                                   mode=0,
                                                   shuffle=0,
                                                   device=device)

data.save_normalization_parameters(experiment_directory)

trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = encoding_slices_dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

sys.stdout.flush()

# setup training
epochs = specs["epochs"]
netD = utils.set_device(Decoder(latent_size=data.latent_size), device=device)

print(netD)
model_parameters = filter(lambda p: p.requires_grad, netD.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized UNet with {} trainable params ".format(params))

sys.stdout.flush()

netD.apply(weights_init)
if len(doLoad) > 0:
    netD.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = utils.set_device(utils.CustomWeightedL1Loss(0.0, sdf_threshold=0.0001), device=device)
criterion = nn.MSELoss()

optimizer = optim.Adam(netD.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = utils.set_device(Variable(torch.FloatTensor(batch_size, 3, 128, 128)), device=device)
inputs = utils.set_device(Variable(torch.FloatTensor(batch_size, data.latent_size + 2, 1, 1)), device=device)

##########################

train_loss_list = []
val_loss_list = []
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs), flush=True)

    netD.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu, sdf_cpu = traindata
        inputs_cpu = utils.set_device(inputs_cpu.float(), device)
        targets_cpu = utils.set_device(targets_cpu.float(), device)
        sdf_cpu = utils.set_device(sdf_cpu.float(), device)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizer.param_groups:
                    g['lr'] = currLr

        netD.zero_grad()
        gen_out = netD(inputs)

        lossL1 = criterionL1(gen_out, targets, sdf_cpu)
        lossL1.backward()

        optimizer.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)
        if epoch % 20 == 0:
            sys.stdout.flush()
        if epoch % 100 == 0:
            torch.save(netD.state_dict(), os.path.join(experiment_directory, "model_D"))
            np.savetxt(os.path.join(experiment_directory, "loss_log.txt"), np.asarray([train_loss_list, val_loss_list]))

    # validation
    netD.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu, sdf_cpu = validata
        inputs_cpu = utils.set_device(inputs_cpu.float(), device)
        targets_cpu = utils.set_device(targets_cpu.float(), device)
        sdf_cpu = utils.set_device(sdf_cpu.float(), device)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netD(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets, sdf_cpu)
        L1val_accum += lossL1.item()

    L1_accum /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    train_loss_list.append(L1_accum)
    val_loss_list.append(L1val_accum)

plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.xlabel("Epoch")
plt.legend(["Train", "Val"])
plt.savefig(os.path.join(experiment_directory, "losses.png"), dpi=120)
torch.save(netD.state_dict(), os.path.join(experiment_directory, "model_D"))
