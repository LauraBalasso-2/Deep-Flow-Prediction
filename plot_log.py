import argparse
import os
import numpy as np

import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description="Train U-Net")
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
experiment_directory = args.experiment_directory


loss_log = os.path.join(experiment_directory, "loss_D.txt")

loss = np.loadtxt(loss_log)

plt.plot(loss[0], label="train loss")
plt.plot(loss[1], label="val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(experiment_directory, "losses.png"))