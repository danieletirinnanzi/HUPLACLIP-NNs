# Description: Main file to run the project

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F  # needed in CNN training loop
import os  # needed to save models


from src.utils import load_config
from src.utils import load_model

# from src.utils import train_model

# loading experiment configuration file:
config = load_config("docs\MLP_experiment_configuration.yml")
graph_size = config["graph_size"]
# config = load_config("docs\MLP_CNN_experiment_configuration.yml")

# loading, training, and testing all models:
for model_specs in config["models"]:

    # printing model name
    print(model_specs["model_name"])

    # loading model
    model = load_model(model_specs["model_name"], graph_size)

    print(model)

    # training model
    # train_model(model, model_specs["hyperparameters"])

    # testing model and saving results


# once training is done, fitting data and comparing models:


# # for tensorboard visualization
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


# # train_model(model=model)
