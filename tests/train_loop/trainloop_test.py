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
print(device)

config = load_config(os.path.join("docs", "grid_exp_config.yml"))  # CHANGE THIS

model_name = "MLP"  # CHANGE THIS

# Create mock_writer to visualize training:
log_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mock_runs",
    config["models"][0]["model_name"],
)
mock_writer = SummaryWriter(log_dir=log_dir)

# Create mock results folder:
results_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mock_results",
    config["models"][0]["model_name"],
)
os.makedirs(results_dir)

# model config:
model_config = [
    model for model in config["models"] if model["model_name"] == model_name
][0]

# Initialize the model (choosing lower graph size for testing)
model = MLP(config["graph_size"][0], model_config["architecture"])

print(model)

# Sending model to device:
model.to(device)

# Train the model
model.train()
train_model(
    model,
    config["training_parameters"],
    config["graph_size"],
    config["p_correction_type"],
    mock_writer,
    config["models"][0]["model_name"],
    results_dir,
)
