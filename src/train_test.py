# imports
import numpy as np
import torch
import torch.optim
import torch.nn as nn

# custom import
import src.graphs_generation as gen_graphs

# TEST THAT GRAPHS ARE GENERATED CORRECTLY

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAINING FUNCTIONS:


# Early stopping function:
def early_stopper(
    validation_loss,
    memory=5,
    validation_exit_error=0.2,
):
    """
    Implements early stopping based on stored validation loss. Validation loss is stored every "saved_steps". If the AVERAGE value of the last "memory" losses is below the "exit_error" values defined above, then training is interrupted and the next task version is trained.

    Args:
        validation_loss (numpy.array): Array containing validation loss values (stored every 10 training steps).
        memory (int): Number of previous cycles to consider for calculating average validation and training losses.
        validation_exit_error (float): Validation error below which we can stop early

    Returns:
        tuple: A boolean indicating whether to stop early.
    """

    if (
        np.mean(validation_loss[validation_loss.shape[0] - memory :])
        < validation_exit_error
    ):
        return True
    else:
        return False


# Training function:
def train_model(
    model, training_hyperparameters, graph_size, p_correction_type, writer, model_name
):

    #  NOTES FOR WRITING DOCUMENTATION:
    # - "model" is the loaded model
    # - configuration file contains all hyperparameters for training
    # - writer is the Tensorboard writer
    # - model_name is the name of the model, and is needed in the case of VGG -> 3D graphs are generated

    ## START OF TESTS
    # TODO: MOVE TO A "tests.py" FILE?

    # Check if model is provided
    if model is None:
        raise ValueError("Model is not provided.")

    # Check if training_hyperparameters is a dictionary
    if not isinstance(training_hyperparameters, dict):
        raise ValueError("training_hyperparameters should be a dictionary.")

    # Check if graph_size is a positive integer
    if not isinstance(graph_size, int) or graph_size <= 0:
        raise ValueError("graph_size should be a positive integer.")

    # Check if p_correction_type is a string
    if not isinstance(p_correction_type, str):
        raise ValueError("p_correction_type should be a string.")

    # Check if writer is provided
    if writer is None:
        raise ValueError("Writer is not provided.")

    ## END OF TESTS

    # if model is VGG, graphs will be 3D:
    if model_name == "VGG16":
        vgg_input = True
    else:
        vgg_input = False

    # Notify start of training:
    print("||| Started training...")

    # Defining optimizer:
    if training_hyperparameters["optimizer"] == "Adam":
        optim = torch.optim.Adam(model.parameters())
    elif training_hyperparameters["optimizer"] == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=training_hyperparameters["learning_rate"],
            momentum=0.9,
        )  # DEFINE DYNAMICALLY?
    # ADD MORE OPTIMIZERS?
    else:
        raise ValueError("Optimizer not found")

    # Defining loss function:
    if training_hyperparameters["loss_function"] == "BCELoss":
        criterion = nn.BCELoss()
    elif training_hyperparameters["loss_function"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif training_hyperparameters["loss_function"] == "MSELoss":
        criterion = nn.MSELoss()
    # ADD MORE LOSS FUNCTIONS?
    else:
        raise ValueError("Loss function not found")

    # Initializations
    train_error = []
    val_error = []
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
        print("||| Clique size is now: ", current_clique_size)

        # Training cycles loop:
        for cycle in range(training_hyperparameters["num_cycles"]):

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
                    vgg_input,
                )
                # Forward pass on training data
                train_pred = model(train[0].to(device))
                train_pred = train_pred.squeeze()  # remove extra dimension

                # Compute loss on training data
                train_loss = criterion(
                    train_pred.type(torch.float).to(device),
                    torch.Tensor(train[1])
                    .type(torch.float)
                    .to(device),  # labels should be float for BCELoss
                )

                # At regular intervals (every "save_step"), saving errors (both training and validation) and printing to Tensorboard:
                if training_step % training_hyperparameters["save_step"] == 0:

                    # At each training step that has to be saved:
                    # - increasing saved_steps: this will be the x axis of the tensorboard plots
                    saved_steps += 1

                    # - storing training error (refers to current task version)
                    train_error.append(train_loss.item())

                    # - creating dictionary to store validation losses for all task versions (will be logged to Tensorboard):
                    val_dict = {"train_error": train_loss.item()}

                    # At each save_step, generate validation set and compute validation error for all the task versions:
                    for current_clique_size_val in clique_sizes:

                        # Generating validation graphs:
                        val = gen_graphs.generate_graphs(
                            training_hyperparameters["num_val"],
                            graph_size,
                            current_clique_size_val,
                            p_correction_type,
                            vgg_input,
                        )
                        # Compute loss on validation set:
                        val_pred = model(val[0].to(device))
                        val_pred = val_pred.squeeze()  # remove extra dimension
                        val_loss = criterion(
                            val_pred.to(device),
                            torch.Tensor(val[1])
                            .type(torch.float)
                            .to(device),  # labels should be float for BCELoss
                        )

                        # Storing validation error (only when validating the current task version, needed for early stopping)
                        if current_clique_size_val == current_clique_size:
                            val_error.append(val_loss.item())

                        # updating dictionary with validation losses for all task versions:
                        val_dict[f"val_error_{current_clique_size_val}"] = (
                            val_loss.item()
                        )

                    # Tensorboard: plotting validation loss for other task versions in the same plot as training loss for current task version
                    writer.add_scalars(
                        f"Log_{model_name}",
                        val_dict,
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

            # At the end of the cycle:

            # 1. checking if early stopping condition is met (based on the validation error in the last 3 "save_steps")
            early_exit = early_stopper(np.array(val_error))
            if early_exit:

                # if early stopping, clearing lists storing training and validation errors before breaking out of the cycle loop:
                train_error = []
                val_error = []

                break

            # 2. If no early stop, clearing lists storing training and validation errors before starting new cycle:
            train_error = []
            val_error = []

        # After clique size has finished training (here we are inside the clique size decreasing loop):

        # 1. Tensorboard: printing a vertical bar of 4 points in the plot, to separate the different task versions
        # - spacing values for the vertical lines:
        spacing_values = np.arange(0, 1.1, 0.10)
        # - dictionary with scalar values for the vertical lines:
        scalar_values = {
            f"vert_line_{round(value,2)}_{current_clique_size}": value
            for value in spacing_values
        }
        # - add the scalars to the writer
        writer.add_scalars(f"Log_{model_name}", scalar_values, saved_steps)

        # 2. Printing a message to indicate the end of training for the current task version:
        print("||| Completed training for clique = ", current_clique_size)
        print("||| ==========================================")

    # After all task versions have been trained:
    # - notify completion of training:
    print("||| Finished training.")
    # - close the writer:
    writer.close()
    # - notify completion of training function execution:
    print("- Model trained successfully.")

    return model


# TESTING FUNCTION:
def test_model(
    model, training_hyperparameters, graph_size, p_correction_type, model_name
):

    # if model is VGG, graphs will be 3D:
    if model_name == "VGG16":
        vgg_input = True
    else:
        vgg_input = False

    # Notify start of testing:
    print("||| Started testing...")

    # returns the results of testing for N=300 as a dictionary:
    # { K: fraction correct, ...}

    # creating empty dictionary:
    results = {}

    # calculating max clique size (proportion of graph size):
    max_clique_size = int(
        training_hyperparameters["max_clique_size_proportion_test"] * graph_size
    )
    # calculating array of clique sizes for all test curriculum:
    clique_sizes = np.arange(
        max_clique_size, 0, -training_hyperparameters["jump_test"]
    ).astype(int)

    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # Loop for testing iterations:
        for test_iter in range(training_hyperparameters["test_iterations"]):

            # Testing the network with test data
            test = gen_graphs.generate_graphs(
                training_hyperparameters["num_test"],
                graph_size,
                current_clique_size,
                p_correction_type,
                vgg_input,
            )  # generating test data
            hard_output = torch.zeros(
                [training_hyperparameters["num_test"]]
            )  # initializing tensor to store hard predictions
            soft_output = model(
                test[0].to(device)
            )  # performing forward pass on test data
            soft_output = soft_output.squeeze()  # remove extra dimension
            # Converting soft predictions to hard predictions:
            for index in range(training_hyperparameters["num_test"]):
                if soft_output[index] > 0.5:
                    hard_output[index] = 1.0
                else:
                    hard_output[index] = 0.0
            predicted_output = hard_output

        # Calculating fraction of correct predictions on test set:
        accuracy = (predicted_output == torch.Tensor(test[1])).sum().item() / (
            1.0 * training_hyperparameters["num_test"]
        )
        results[current_clique_size] = accuracy

        # Printing the size of the clique just tested and the corresponding test accuracy:
        print(
            "|||Completed testing for clique = ",
            current_clique_size,
            ". Accuracy on test set =",
            accuracy,
        )
        print("|||==========================================")

    # After all task versions have been tested:
    # - notify completion of testing:
    print("||| Finished testing.")
    # - notify completion of testing function execution:
    print("- Model tested successfully.")

    return results
