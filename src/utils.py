import numpy as np


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


# -----------------------------------------


def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def get_model(model_name):
    match model_name:
        case "MLP":
            model = MLP()
        case _:
            model = None
    return model


def train_model(model_name):
    model = get_model(model_name)
    model.train()
    return model
