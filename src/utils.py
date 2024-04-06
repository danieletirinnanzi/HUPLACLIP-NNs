import numpy as np
import yaml

# for tensorboard visualization during training
from torch.utils.tensorboard import SummaryWriter

# importing Models class from models.py
from .models import Models

# importing graphs_generation.py
# import src.graphs_generation as gen_graphs


# -----------------------------------------
# Loading experiment configuration file:
def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


# Loading model based on model name:
def load_model(model_name, graph_size):
    match model_name:
        case "MLP":
            return Models.mlp(graph_size)
        case "CNN":
            return Models.cnn(graph_size)
        case _:
            raise ValueError("Model not found")


# -----------------------------------------
# TRAINING FUNCTIONS:


# Early stopping function:
def early_stopper(
    validation_loss,
    training_loss,
    memory=3,
    validation_exit_error=0.2,
    training_exit_error=0.2,
):
    """
    Implements early stopping based on validation and training losses. If both are below the "exit_error" values defined above, then training is interrupted.

    Args:
        validation_loss (numpy.array): Array containing validation loss values.
        training_loss (numpy.array): Array containing training loss values.
        counter (int): Current count of epochs where validation loss increases.
        min_validation_loss (float): Minimum validation loss encountered so far.
        memory (int): Number of previous epochs to consider for calculating average validation and training losses.
        validation_exit_error (float): Validation error below which we can stop early
        training_exit_error (float): Training error below which we can stop early

    Returns:
        tuple: A boolean indicating whether to stop early.
    """

    # Computing length of validation and training losses
    val_length = validation_loss.shape[0]
    training_length = training_loss.shape[0]

    # Checking if average value of validation and training losses in the last "memory" epochs allows early stopping:
    if (
        np.mean(validation_loss[val_length - memory :]) < validation_exit_error
        and np.mean(training_loss[training_length - memory :]) < training_exit_error
    ):
        return True
    else:
        return False


# Training function:
def train_model(model, config_file):

    # "model is the loaded model"
    # configuration file contains all hyperparameters for training

    # dynamically clique sizes of training:
    start_clique_size = round(
        config_file["graph_size"] * 0.60
    )  # 60% of graph size rounded to closest integer
    min_clique_size = round(
        config_file["graph_size"] * 0.40
    )  # 40% of graph size rounded to closest integer
    jump = 5
    save_step = 5

    model.train()

    return model


# Testing function:
def test_model(model):

    # SHOULD RETURN THE RESULTS OF THE TESTING IN A FORMAT THAT CAN BE REPORTED IN PLOTS

    pass


# -----------------------------------------
