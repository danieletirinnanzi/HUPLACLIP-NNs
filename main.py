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


from src.models import MLP
from src.utils import train_model
from src.utils import load_config

# config = load_config("config.yaml")

# model = get_model(config['model_name'])

# for tensorboard visualization
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# train_model(model=model)
