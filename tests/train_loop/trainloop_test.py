import torch
import os
from torch.utils.tensorboard import SummaryWriter

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

from src.models import MLP
from src.utils import load_config
from src.train_test import train_model

# TO RUN as module: FROM HOME DIRECTORY -> python -m tests.train_loop.trainloop_test

# Setting device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading MLP configuration:
config = load_config(os.path.join("docs", "MLP_exp_config.yml"))

# Create mock_writer to visualize training:
log_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mock_runs",
    config["models"][0]["model_name"],
)
mock_writer = SummaryWriter(log_dir=log_dir)

# Create an instance of the MLP model
model = MLP(config["graph_size"], config["models"][0]["architecture"])

# Sending model to device:
model.to(device)

# Train the model
model.train()
trained_model = train_model(
    model,
    config["training_parameters"],
    config["graph_size"],
    config["p_correction_type"],
    mock_writer,
    config["models"][0]["model_name"],
)
