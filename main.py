# Description: Main file to run the project

# # NOT USED HERE, ADD IN SCRIPTS?
# import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import torch.nn.functional as F  # needed in CNN training loop
# import os  # needed to save models

import datetime
import os

from torch.utils.tensorboard import SummaryWriter

from src.utils import load_config
from src.utils import load_model
from src.utils import train_model


# loading experiment configuration file:
config = load_config("docs\MLP_experiment_configuration.yml")

# creating Tensorboard writer and specifying logging folder:
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name_with_time = f"{config['exp_name']}_{current_time}"

# creating Tensorboard directory for the experiment:
current_dir = os.path.dirname(os.path.realpath(__file__))
runs_dir = os.path.join(current_dir, "runs")
experiment_dir = os.path.join(runs_dir, exp_name_with_time)
print(experiment_dir)

writer = SummaryWriter(log_dir=experiment_dir)

# opening Tensorboard in the terminal:
# tensorboard --logdir=runs

# loading, training, and testing all models:
for model_specs in config["models"]:

    # printing model name
    print(model_specs["model_name"])

    # loading model
    model = load_model(model_specs["model_name"], config["graph_size"])

    # training model and visualizing it on Tensorboard
    train_model(
        model,
        model_specs["hyperparameters"],
        config["graph_size"],
        config["p_correction_type"],
        writer,
    )

    # testing model and saving results
