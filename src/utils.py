import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import torch
import torch.optim as optim
import torch.nn as nn

# for tensorboard visualization during training
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()  # TODO: customize name of folder that is created

# importing Models class from models.py
from .models import Models

# importing graphs_generation.py
import src.graphs_generation as gen_graphs

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------
# Loading experiment configuration file:
def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    print("Configuration file loaded successfully.")
    return config


# Loading model based on model name:
def load_model(model_name, graph_size):

    # - building requested model
    match model_name:
        case "MLP":
            model = Models.mlp(graph_size)
        case "CNN":
            model = Models.cnn(graph_size)
        case _:
            raise ValueError("Model not found")

    # - sending model to device:
    model.to(device)
    print("Model loaded successfully.")
    return model


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
def train_model(model, training_hyperparameters, graph_size, p_correction_type):

    # - "model" is the loaded model
    # - configuration file contains all hyperparameters for training

    # TO BE SPECIFIED IN CONFIGURATION FILES AND SIMPLY READ HERE?
    optim = torch.optim.Adam(model.parameters())  # optimization with Adam
    criterion = nn.CrossEntropyLoss()  # criterion = Cross Entropy

    # Initializations
    train_error = []
    val_error = []
    generalization = []
    k_over_sqrt_n = []
    clique_sizes_array = []
    saved_steps = 0  # will increase every time we save a step, and will be on the x axis of tensorboard plots (global graphs with training and validation losses over all training)

    # calculating min clique size and max clique size (proportion of graph size):
    max_clique_size = int(
        training_hyperparameters["max_clique_size_proportion"] * graph_size
    )
    min_clique_size = int(
        training_hyperparameters["min_clique_size_proportion"] * graph_size
    )
    # calculating array of clique sizes for all training curriculum:
    clique_sizes = np.arange(
        max_clique_size, min_clique_size - 1, -training_hyperparameters["jump"]
    ).astype(int)

    # training loop:
    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # printing value of clique size when it changes:
        print("Clique size is now: ", current_clique_size)
        # printing data type of current_clique_size:
        print(
            "Data type of current_clique_size: ", type(current_clique_size.astype(int))
        )
        # data type of graph_size:
        print("Data type of graph_size: ", type(graph_size))

        # storing values needed for final plot:
        k_over_sqrt_n.append(1.0 * current_clique_size / math.sqrt(graph_size))
        clique_sizes_array.append(current_clique_size)

        # Training epochs loop:
        for epoch in range(training_hyperparameters["num_epochs"]):

            # Training steps loop:
            for training_step in range(
                training_hyperparameters["num_training_steps"] + 1
            ):

                # Generating training data
                train = gen_graphs.generate_graphs(
                    training_hyperparameters["num_train"],
                    graph_size,
                    current_clique_size,
                    p_correction_type,
                )
                # Forward pass on training data
                train_pred = model(train[0].to(device))
                # Compute loss on training data
                train_loss = criterion(
                    train_pred.type(torch.float).to(device),
                    torch.Tensor(train[1]).type(torch.long).to(device),
                )

                # Saving errors (both training and validation) and plotting at regular intervals (indicated by "save_step"):
                if training_step % training_hyperparameters["save_step"] == 0:

                    # increasing saved_steps: this will be the x axis of the tensorboard plots
                    saved_steps += 1

                    # Storing training error (refers to current task version)
                    train_error.append(train_loss.item())
                    # Tensorboard: plotting training loss for current task version
                    writer.add_scalar(
                        f"Loss/train_task_{current_clique_size}",
                        train_loss.item(),
                        saved_steps,
                    )

                    # At each save_step, generate validation set for all the task versions and compute validation error:
                    for current_clique_size_val in clique_sizes:

                        # Generating validation graphs:
                        val = gen_graphs.generate_graphs(
                            training_hyperparameters["num_val"],
                            graph_size,
                            current_clique_size_val,
                            p_correction_type,
                        )
                        # Compute loss on validation set:
                        val_pred = model(val[0].to(device))
                        val_loss = criterion(
                            val_pred.to(device),
                            torch.Tensor(val[1]).type(torch.long).to(device),
                        )
                        # Storing validation error (only when validating the current task version, needed for early stopping)
                        if current_clique_size_val == current_clique_size:
                            val_error.append(val_loss.item())

                        # Tensorboard: plotting validation loss for other task versions in the same plot as training loss for current task version
                        writer.add_scalar(
                            f"Loss/val_task_{current_clique_size_val}",
                            val_loss.item(),
                            saved_steps,
                        )

                    # Flush the writer to make sure all data is written to disk
                    writer.flush()

                # Backward pass
                train_loss.backward()
                # Update weights
                optim.step()
                # Clear gradients
                optim.zero_grad(set_to_none=True)

            # At the end of the epoch:

            # 1. checking if early stopping condition is met (based on the validation and training error in the last 3 training steps)
            early_exit = early_stopper(np.array(val_error), np.array(train_error))
            if early_exit:

                # printing training and validation errors to check that Early exit is working properly
                # NOTE: training and validation errors are saved every "save_step" steps. Validation set is generated and tested every 5 steps, while training error is always calculated, but saved every 5 steps as well
                print(
                    "Exiting early, validation error array: ",
                    np.array(val_error[-5:]),
                    ". Training error array: ",
                    np.array(train_error[-5:]),
                )  # printing out last 5 elements, to check that early stopping criterior is really met

                # clearing lists storing training and validation errors before breaking out:
                train_error = []
                val_error = []

                break

            # 2. clearing lists storing training and validation errors before starting new epoch:
            train_error = []
            val_error = []

        # After clique size has finished training (here we are inside the clique size decreasing loop):

        # 1. Testing the network with test data
        # CAN BE REMOVED? TESTING IS DONE AFTER TRAINING IS COMPLETED, WITH A SEPARATE FUNCTION
        test = gen_graphs.generate_graphs(
            training_hyperparameters["num_test"],
            graph_size,
            current_clique_size,
            p_correction_type,
        )  # generating test data
        hard_output = torch.zeros(
            [training_hyperparameters["num_test"], 2]
        )  # initializing tensor to store hard predictions
        soft_output = model(test[0].to(device))  # performing forward pass on test data
        # Converting soft predictions to hard predictions:
        for index in range(training_hyperparameters["num_test"]):
            if soft_output[index][0] > soft_output[index][1]:
                hard_output[index][0] = 1.0
            else:
                hard_output[index][1] = 1.0
        predicted_output = hard_output

        # 2. Computing and storing the generalization error for the current clique size:
        generalization.append(
            100
            * (
                1
                - torch.sum(
                    torch.square(
                        torch.Tensor(test[1])
                        - torch.transpose(predicted_output, 1, 0)[1]
                    )
                ).item()
                / (1.0 * training_hyperparameters["num_test"])
            )
        )

        # 3. Printing the current k/sqrt(n) and the corresponding test error:
        print(
            "Completed training for clique = ",
            current_clique_size,
            ". % correct on test set =",
            generalization,
        )
        print("==========================================")

    # # After training is completed, plotting generalization errors for all trained clique sizes:
    # # - Creating subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # # - Plotting on the first subplot
    # ax1.plot(clique_sizes_array, generalization)
    # ax1.set_title("Test Set Performance vs. Clique Size", fontsize=16)
    # ax1.set_xlabel("Clique Size", fontsize=14)
    # ax1.set_ylabel("Test Set Performance (%)", fontsize=14)

    # # - Plotting on the second subplot
    # ax2.plot(k_over_sqrt_n, generalization)
    # ax2.set_title("Test Set Performance vs. k/sqrt{n}", fontsize=16)
    # ax2.set_xlabel("k/sqrt{n}", fontsize=14)
    # ax2.set_ylabel("Test Set Performance (%)", fontsize=14)

    # # - Adjusting layout
    # plt.tight_layout()

    # # - Displaying the plot
    # plt.show()

    # Closing the writer:
    writer.close()

    return model


# Testing function:
def test_model(model):

    # SHOULD RETURN THE RESULTS OF THE TESTING IN A FORMAT THAT CAN BE REPORTED IN PLOTS

    pass


# -----------------------------------------
