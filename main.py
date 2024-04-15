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
from src.utils import test_model


# loading experiment configuration file:
config = load_config("docs\MLP_experiment_configuration.yml")

# Tensorboard:
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name_with_time = f"{config['exp_name']}_N{config['graph_size']}_{current_time}"
current_dir = os.path.dirname(os.path.realpath(__file__))
runs_dir = os.path.join(current_dir, "runs")
# create a new directory for each experiment
experiment_dir = os.path.join(runs_dir, exp_name_with_time)
# create writer and point to log directory
writer = SummaryWriter(log_dir=experiment_dir)

# creating folder in "results" folder to save the results of the experiment
results_dir = os.path.join(current_dir, "results", exp_name_with_time)
os.makedirs(results_dir)

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
    test_model(
        model,
        model_specs["hyperparameters"],
        config["graph_size"],
        config["p_correction_type"],
        results_dir,
    )
