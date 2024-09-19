################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Helper functions for image output
#
################

import math, re, os
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from scipy.ndimage import gaussian_filter


# add line to logfiles
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint: print(line)


# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()


# compute learning rate with decay in second half
def computeLR(i, epochs, minLR, maxLR):
    if i < epochs * 0.5:
        return maxLR
    e = (i / float(epochs) - 0.5) * 2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e * (fmax - fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR - minLR) * f


def save_true_pred_img(filename, _outputs, _targets, _input, smoothing=True):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    mask = np.copy(_input)
    components_dict = {0: "_x", 1: "_y", 2: "_z"}
    for i in range(outputs.shape[0]):

        if smoothing:
            target_i = gaussian_filter(targets[i], sigma=5)
        else:
            target_i = targets[i]

        fig, axs = plt.subplots(2, 1)
        masked_target = np.ma.masked_where(mask > 0, target_i)
        aa = axs[0].matshow(masked_target)
        axs[0].set_title("true")
        fig.colorbar(aa, ax=axs[0])

        masked_output = np.ma.masked_where(mask > 0, outputs[i])
        ab = axs[1].matshow(masked_output)
        axs[1].set_title("pred")
        fig.colorbar(ab, ax=axs[1])
        plt.savefig(filename + components_dict.get(i) + ".png", dpi=200)
        plt.close()

        err = np.abs(target_i - outputs[i])
        masked_err = np.ma.masked_where(mask > 0, err)
        plt.matshow(masked_err)
        plt.title("masked error")
        plt.colorbar()
        plt.savefig(filename + components_dict.get(i) + "_masked_error.png", dpi=150)
        plt.close()


def makeDirs(directoryList):
    for directory in directoryList:
        os.makedirs(directory, exist_ok=True)


class CustomWeightedL1Loss(nn.Module):
    def __init__(self, lambda_weight=0.1, sdf_threshold=0.0001):
        super(CustomWeightedL1Loss, self).__init__()
        self.lambda_weight = lambda_weight
        self.sdf_threshold = sdf_threshold

    def forward(self, predictions, targets, additional_param):
        l1_loss = torch.abs(predictions - targets)

        # Create the weights: lambda_weight where additional_param is positive, and 1 where it is negative
        weights = torch.where(additional_param > self.sdf_threshold,
                              torch.full_like(additional_param, self.lambda_weight),
                              torch.ones_like(additional_param))

        weighted_l1_loss = l1_loss * weights

        non_zero_mask = weighted_l1_loss != 0

        # Compute the mean over the nonzero elements
        if torch.any(non_zero_mask):
            loss = weighted_l1_loss[non_zero_mask].mean()
        else:
            loss = torch.tensor(0.0, dtype=weighted_l1_loss.dtype, device=weighted_l1_loss.device)

        return loss
