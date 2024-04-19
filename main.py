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

# custom imports
from src.utils import load_config
from src.utils import load_model
from src.utils import save_test_results
from src.utils import save_trained_model
from src.train_test import train_model
from src.train_test import test_model


# loading experiment configuration file:
config = load_config("docs\VGG_experiment_configuration.yml")

# Tensorboard:
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name_with_time = f"{config['exp_name']}_N{config['graph_size']}_{current_time}"
current_dir = os.path.dirname(os.path.realpath(__file__))
runs_dir = os.path.join(current_dir, "runs")
# create a new directory for each experiment
experiment_dir = os.path.join(runs_dir, exp_name_with_time)
# create writer and point to log directory
writer = SummaryWriter(log_dir=experiment_dir)

# creating folder in "results" folder to save the results of the whole experiment
results_dir = os.path.join(current_dir, "results", exp_name_with_time)

# loading, training, and testing models:
for model_specs in config["models"]:

    # printing model name
    print(model_specs["model_name"])

    # loading model
    model = load_model(
        model_specs["model_name"], config["graph_size"], model_specs["hyperparameters"]
    )

    # training model and visualizing it on Tensorboard
    trained_model = train_model(
        model,
        model_specs["hyperparameters"],
        config["graph_size"],
        config["p_correction_type"],
        writer,
        model_specs["model_name"],
    )

    # testing model
    test_results = test_model(
        model,
        model_specs["hyperparameters"],
        config["graph_size"],
        config["p_correction_type"],
        model_specs["model_name"],
    )

    # creating model subfolder in current experiment folder:
    model_results_dir = os.path.join(results_dir, model_specs["model_name"])
    os.makedirs(model_results_dir)

    # - saving test results as csv file
    save_test_results(
        test_results, model_specs["model_name"], config["graph_size"], model_results_dir
    )

    # - saving the trained model (will not be synched with git due to size)
    save_trained_model(
        trained_model,
        model_specs["model_name"],
        config["graph_size"],
        model_results_dir,
    )
