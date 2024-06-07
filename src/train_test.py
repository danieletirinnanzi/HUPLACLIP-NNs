# imports
import numpy as np
import torch
import torch.optim
import torch.nn as nn

# custom import
import src.graphs_generation as gen_graphs

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAINING FUNCTIONS:


# # OLD EARLY STOPPER
# def early_stopper(
#     validation_loss,
#     memory=5,
#     validation_exit_error=0.1,  # CAN BE CHANGED
# ):
#     """
#     Implements early stopping based on stored validation loss. Validation loss is stored every "saved_steps". If the AVERAGE value of the last "memory" losses is below the "exit_error" values defined above, then training is interrupted and the next task version is trained.

#     Args:
#         validation_loss (numpy.array): Array containing validation loss values (stored every 10 training steps).
#         memory (int): Number of previous cycles to consider for calculating average validation and training losses.
#         validation_exit_error (float): Validation error below which we can stop early

#     Returns:
#         tuple: A boolean indicating whether to stop early.
#     """

#     if (
#         np.mean(validation_loss[validation_loss.shape[0] - memory :])
#         < validation_exit_error
#     ):
#         return True
#     else:
#         return False


# NEW EARLY STOPPER ( adapted from: https://stackoverflow.com/a/73704579 )
class EarlyStopper:
    """
    EarlyStopper is a simple class to stop the training when the validation loss
    is not improving for a certain number of epochs.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the loss to qualify as an improvement.
        val_increase_counter (int): Counter that increments when the validation loss increases.
        min_val_loss (float): The minimum validation loss seen so far.
        val_exit_counter (int): Counter that increments when the validation loss is below the exit value.
        val_exit_loss (float): The validation loss under which the training should stop (exit early).
    """

    def __init__(self, patience=3, min_delta=0.01, val_exit_loss=0.1):
        self.patience = patience
        self.min_delta = min_delta
        # increase in validation loss:
        self.val_increase_counter = 0
        self.min_val_loss = float("inf")
        # validation loss under exit value:
        self.val_exit_counter = 0
        self.val_exit_loss = val_exit_loss

    def should_stop(self, val_loss):
        """
        Determines whether the training should stop based on the validation loss.

        Args:
            val_loss (float): The current validation loss.

        Returns:
            bool: True if the training should stop, False otherwise.
        """
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.val_increase_counter = 0
        elif val_loss > self.min_val_loss + self.min_delta:
            self.val_increase_counter += 1
            if self.val_increase_counter >= self.patience:
                return True
        else:
            self.val_increase_counter = 0

        if val_loss < self.val_exit_loss:
            self.val_exit_counter += 1
            if self.val_exit_counter >= self.patience:
                return True
        else:
            self.val_exit_counter = 0

        return False


# Training function:
def train_model(
    model, training_parameters, graph_size, p_correction_type, writer, model_name
):
    """
    Trains a model using the specified hyperparameters.

    Args:
        model (torch.nn.Module): The loaded model.
        training_parameters (dict): A dictionary containing all hyperparameters for training (read from configuration file).
        graph_size (int): The size of the graph.
        p_correction_type (str): The type of p-correction.
        writer: The Tensorboard writer.
        model_name (str): The name of the model.

    Raises:
        ValueError: If the model is not provided, training_parameters is not a dictionary,
            graph_size is not a positive integer, p_correction_type is not a string, or writer is not provided.

    Returns:
        None
    """

    # START OF TESTS

    # Check if model is provided
    if model is None:
        raise ValueError("Model is not provided.")

    # Check if training_parameters is a dictionary
    if not isinstance(training_parameters, dict):
        raise ValueError("training_parameters should be a dictionary.")

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

    # Setting input magnification flag to True if needed:
    if model_name == "MLP":
        input_magnification = False
    else:
        input_magnification = True

    # Notify start of training:
    print("||| Started training...")

    # Defining optimizer with learning rate:
    if training_parameters["optimizer"] == "Adam":
        optim = torch.optim.Adam(
            model.parameters(), lr=training_parameters["learning_rate"]
        )
    elif training_parameters["optimizer"] == "AdamW":
        optim = torch.optim.AdamW(
            model.parameters(), lr=training_parameters["learning_rate"]
        )
    elif training_parameters["optimizer"] == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=training_parameters["learning_rate"],
            momentum=0.9,
        )  # DEFINE DYNAMICALLY?
    # ADD MORE OPTIMIZERS?
    else:
        raise ValueError("Optimizer not found")

    # Defining loss function:
    if training_parameters["loss_function"] == "BCELoss":
        criterion = nn.BCELoss()
    elif training_parameters["loss_function"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif training_parameters["loss_function"] == "MSELoss":
        criterion = nn.MSELoss()
    # ADD MORE LOSS FUNCTIONS?
    else:
        raise ValueError("Loss function not found")

    # Initializations
    saved_steps = 0  # will increase every time we save a step, and will be on the x axis of tensorboard plots (global graphs with training and validation losses over all training)

    # calculating min clique size and max clique size (proportion of graph size):
    max_clique_size = int(
        training_parameters["max_clique_size_proportion"] * graph_size
    )
    min_clique_size = int(
        training_parameters["min_clique_size_proportion"] * graph_size
    )
    # calculating array of clique sizes for all training curriculum:
    clique_sizes = np.linspace(
        max_clique_size,
        min_clique_size,
        num=training_parameters["clique_training_levels"],
    ).astype(int)

    # training loop:
    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # printing value of clique size when it changes:
        print("||| Clique size is now: ", current_clique_size)

        # Initialize early stopper at the beginning of each task version training:
        early_stopper = EarlyStopper()

        # Training steps loop:
        for training_step in range(training_parameters["num_training_steps"] + 1):

            # Generating training data
            train = gen_graphs.generate_graphs(
                training_parameters["num_train"],
                graph_size,
                current_clique_size,
                p_correction_type,
                input_magnification,
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
            # Backward pass
            train_loss.backward()
            # Update weights
            optim.step()
            # Clear gradients
            optim.zero_grad(set_to_none=True)

            # At regular intervals (every "save_step"), saving errors (both training and validation) and printing to Tensorboard:
            if training_step % training_parameters["save_step"] == 0:

                # Put model in evaluation mode and disable gradient computation
                model.eval()
                with torch.no_grad():

                    # At each training step that has to be saved:
                    # - increasing saved_steps: this will be the x axis of the tensorboard plots
                    saved_steps += 1

                    # - creating dictionary (that includes training loss) to store validation losses for all task versions (will be logged to Tensorboard):
                    val_dict = {"train_loss": train_loss.item()}

                    # At each save_step, generate validation set and compute validation error for all the task versions:
                    for current_clique_size_val in clique_sizes:

                        # Generating validation graphs:
                        val = gen_graphs.generate_graphs(
                            training_parameters["num_val"],
                            graph_size,
                            current_clique_size_val,
                            p_correction_type,
                            input_magnification,
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

                        # Checking early stopping condition for the current task version:
                        if current_clique_size_val == current_clique_size:
                            early_exit = early_stopper.should_stop(val_loss.item())

                            print("Exiting early? ", early_exit)
                            print("Validation loss: ", val_loss.item())
                            print("Min validation loss: ", early_stopper.min_val_loss)
                            print(
                                "Validation increase counter: ",
                                early_stopper.val_increase_counter,
                            )
                            print(
                                "Validation exit counter: ",
                                early_stopper.val_exit_counter,
                            )
                            print("====================================")

                        # updating dictionary with validation losses for all task versions:
                        val_dict[f"val_loss_{current_clique_size_val}"] = (
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

                # Put model back in training mode after validation is done
                model.train()

                # Checking if early stopping condition was met:
                if early_exit:

                    print("||| Early stopping triggered.")

                    break

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
def test_model(model, testing_parameters, graph_size, p_correction_type, model_name):
    """
    Test the given, trained model.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        testing_parameters (dict): A dictionary containing hyperparameters for training.
        graph_size (int): The size of the graph.
        p_correction_type (str): The type of p-correction.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the results of testing for different clique sizes.
              The keys are the clique sizes and the values are the corresponding accuracies.
    """

    # Setting input magnification flag to True if needed:
    if model_name == "MLP":
        input_magnification = False
    else:
        input_magnification = True

    # Notify start of testing:
    print("||| Started testing...")

    # returns the results of testing for N=300 as a dictionary:
    # { K: fraction correct, ...}

    # creating empty dictionary:
    results = {}

    # calculating max clique size (proportion of graph size):
    max_clique_size = int(
        testing_parameters["max_clique_size_proportion_test"] * graph_size
    )
    # calculating array of clique sizes for all test curriculum:
    clique_sizes = np.linspace(
        max_clique_size,
        1,
        num=testing_parameters["clique_testing_levels"],
    ).astype(int)

    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # Loop for testing iterations:
        for test_iter in range(testing_parameters["test_iterations"]):

            # Testing the network with test data
            test = gen_graphs.generate_graphs(
                testing_parameters["num_test"],
                graph_size,
                current_clique_size,
                p_correction_type,
                input_magnification,
            )  # generating test data
            hard_output = torch.zeros(
                [testing_parameters["num_test"]]
            )  # initializing tensor to store hard predictions
            soft_output = model(
                test[0].to(device)
            )  # performing forward pass on test data
            soft_output = soft_output.squeeze()  # remove extra dimension
            # Converting soft predictions to hard predictions
            # NOTE: can be done stochastically?
            for index in range(testing_parameters["num_test"]):
                if soft_output[index] > 0.5:
                    hard_output[index] = 1.0
                else:
                    hard_output[index] = 0.0
            predicted_output = hard_output

        # Calculating fraction of correct predictions on test set:
        accuracy = (predicted_output == torch.Tensor(test[1])).sum().item() / (
            1.0 * testing_parameters["num_test"]
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
