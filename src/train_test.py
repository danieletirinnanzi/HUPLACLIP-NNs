# imports
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# defining random generator (used to define the clique size value of each graph in the batch during training)
random_generator = np.random.default_rng()

# custom import
import src.graphs_generation as gen_graphs
from src.utils import save_model

# TRAINING FUNCTIONS:

# CHECKPOINTING
class Checkpointer:
    """Checkpointer is a simple class to track the minimum validation loss and determine if the model should be saved or not.

    Attributes:
        min_avg_val_loss (float): The minimum validation loss (averaged over all task versions) seen so far.
    """

    def __init__(self):
        self.min_avg_val_loss = float("inf")

    def should_save(self, mean_val_loss):
        """Determines whether the current mean validation loss is lower than the minimum validation loss seen so far.

        Args:
            mean_val_loss (float): The current mean validation loss.

        Returns:
            bool: True if the current mean validation loss is lower than the minimum mean validation loss seen so far, False otherwise.
        """
        if mean_val_loss < self.min_avg_val_loss:
            self.min_avg_val_loss = mean_val_loss
            return True
        return False


# EARLY STOPPER ( adapted from: https://stackoverflow.com/a/73704579 )
class EarlyStopper:
    """
    EarlyStopper is a simple class to stop the training when the monitored validation loss
    is not improving or is below a certain value for a certain number of training steps.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the loss to qualify as an improvement.
        val_increase_counter (int): Counter that increments when the validation loss increases.
        min_val_loss (float): The minimum validation loss seen so far.
        val_exit_counter (int): Counter that increments when the validation loss is below the exit value.
        val_exit_loss (float): The validation loss under which the training should stop (exit early).
        stop_reason (str): The reason for stopping the training.
    """

    def __init__(self, patience=4, min_delta=0.01, val_exit_loss=0.08):
        self.patience = patience
        self.min_delta = min_delta
        # increase in validation loss:
        self.val_increase_counter = 0
        self.min_val_loss = float("inf")
        # validation loss under exit value:
        self.val_exit_counter = 0
        self.val_exit_loss = val_exit_loss
        # stop reason:
        self.stop_reason = None

    def should_stop(self, val_loss):
        """
        Determines whether the training should stop based on the monitored validation loss.

        Args:
            val_loss (float): The current mean validation loss (over all task versions).

        Returns:
            bool: True if the training should stop, False otherwise.
        """
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.val_increase_counter = 0
        elif val_loss > self.min_val_loss + self.min_delta:
            self.val_increase_counter += 1
            if self.val_increase_counter >= self.patience:
                self.stop_reason = "no_improvement"
                return True
        else:
            self.val_increase_counter = 0

        if val_loss < self.val_exit_loss:
            self.val_exit_counter += 1
            if self.val_exit_counter >= self.patience:
                self.stop_reason = "min_loss"
                return True
        else:
            self.val_exit_counter = 0

        return False


# Training function:
def train_model(
    model,
    training_parameters,
    graph_size,
    p_correction_type,
    writer,
    model_name,
    results_dir,
    world_size,
    rank
):
    """
    Trains a model using the specified hyperparameters, saving it as training progresses.
    Training is structured as a curriculum learning task, where the model is trained on graphs with decreasing clique sizes.
    Sketch of training structure:
    FOR (decreasing clique size):
        FOR (training steps):
            - initialize early stopper
            - generate training data
            - forward pass on training data
            - compute loss on training data
            - backward pass and update weights
            - at regular intervals (save_step):
                - generate validation data
                - save errors and print to Tensorboard
            - check if checkpointing condition is met -> if yes, save model
            - check if early stopping condition is met -> if yes, stop training current clique size
        END FOR
    END FOR

    Args:
        model (torch.nn.Module): The loaded model.
        training_parameters (dict): A dictionary containing all hyperparameters for training (they are read from configuration file).
        graph_size (int): The size of the graph.
        p_correction_type (str): The type of p-correction.
        writer: The Tensorboard writer.
        model_name (str): The name of the model.
        results_dir (str): The directory where the best model will be saved.
        world_size (int): Integer indicating the number of GPUs used for training.
        rank: (int): The rank of the GPU that is currently being used
        
    Raises:
        ValueError: If the model is not provided, training_parameters is not a dictionary,
            graph_size is not a positive integer, p_correction_type is not a string, or writer is not provided.
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

    # START OF TRAINING CONFIGURATION:

    # - INPUT TRANSFORMATION FLAGS:
    input_magnification = True if "CNN" in model_name else False

    # - NUMBER OF TRAINING STEPS, OPTIMIZER and LEARNING RATE:
    # reading number of training steps
    num_training_steps = int(training_parameters["num_training_steps"])
    # reading optimizer and learning rate
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
            momentum=0.9,  # default value is zero
        )
    else:
        raise ValueError("Optimizer not found")

    # - LOSS FUNCTION:
    if training_parameters["loss_function"] == "BCELoss":
        criterion = nn.BCELoss()
    elif training_parameters["loss_function"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif training_parameters["loss_function"] == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Loss function not found")

    # INITIALIZATIONS:
    # X axis for Tensorboard plots (will increase every time a step is saved):
    saved_steps = 0

    # Calculating min clique size and max clique size:
    # - max clique size is a proportion of the graph size
    max_clique_size = int(
        training_parameters["max_clique_size_proportion"] * graph_size
    )
    # - min clique size is the statistical limit associated with the graph size
    min_clique_size = int(2 * np.log2(graph_size))

    # Calculating array of clique sizes for all training curriculum:
    clique_sizes = np.linspace(
        max_clique_size,
        min_clique_size,
        num=training_parameters["clique_training_levels"],
    ).astype(int)

    # initializing checkpointer (triggers model saving when mean validation loss is lower than the minimum seen so far):
    checkpointer = Checkpointer()

    # Notify start of training (only rank 0 logs)
    if rank == 0:
        print(f"| Started training {model_name}...")

    # Loop for decreasing clique sizes
    for i, current_clique_size in enumerate(clique_sizes):

        # Defining clique list for current clique size value:
        clique_size_list = clique_sizes[: i + 1]
        if rank == 0:
            print("||| Minimum clique size is now: ", current_clique_size)
            print("||| List of available clique sizes is now: ", clique_size_list)

        # initializing early stopper (triggers passage to following training instance)
        early_stopper = EarlyStopper(
            patience=training_parameters["patience"],
            min_delta=training_parameters["min_delta"],
            val_exit_loss=training_parameters["val_exit_loss"],
        )

        # Training steps loop:
        for training_step in range(training_parameters["num_training_steps"] + 1):

            # Generate clique size value of each graph in the current batch
            clique_size_array_train = gen_graphs.generate_batch_clique_sizes(
                clique_size_list, training_parameters["num_train"]
            )

            # Generating training data (full batch)
            full_train_data = gen_graphs.generate_batch(
                training_parameters["num_train"],
                graph_size,
                clique_size_array_train,
                p_correction_type,
                input_magnification,
            )
            
            # Split training data across GPUs, checking divisibility of batch size by world size
            if training_parameters["num_train"] % world_size != 0:
                raise ValueError(
                    f"Batch size of {training_parameters['num_train']} is not evenly divisible by world_size={world_size}. "
                    f"Each rank requires an equal share of the data for DDP. Please adjust 'num_train' to be divisible by {world_size}."
                )
            # If no errors, proceed with splitting            
            local_batch_size_train = training_parameters["num_train"] // world_size
            start_idx_train = rank * local_batch_size_train
            end_idx_train = (rank + 1) * local_batch_size_train            

            # Partition data for the current rank
            train_data = (
                torch.Tensor(full_train_data[0][start_idx_train:end_idx_train]).to(rank),
                torch.Tensor(full_train_data[1][start_idx_train:end_idx_train]).to(rank),
            )

            # Forward pass on training data
            train_pred = model(train_data[0]).squeeze()
            train_loss = criterion(train_pred.type(torch.float), torch.Tensor(train_data[1]).type(torch.float))            

            # Backward pass
            train_loss.backward()   # DDP GRADIENT SYNCHRONIZATION HAPPENS HERE
            optim.step()
            optim.zero_grad(set_to_none=True)        

            # Free up memory for training data
            # del train_data
            torch.cuda.empty_cache()
            
            # At regular intervals (every "save_step"), saving errors (both training and validation) and printing to Tensorboard:            
            if training_step % training_parameters["save_step"] == 0:
                
                # Waiting for all processes to finish previous tasks (double check, synchronization happens automatically at each forward and backward passes, and at each optimizer step)
                torch.distributed.barrier()   
                # Put model in evaluation mode and disable gradient computation
                model.eval()
                with torch.no_grad():

                    # Increasing saved_steps counter: this will be the x axis of the tensorboard plots
                    saved_steps += 1
                            
                    # Aggregating training loss across GPUs:
                    train_loss_tensor = torch.tensor(train_loss.item(), device=rank)
                    torch.distributed.all_reduce(train_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    if rank == 0:
                        global_train_loss = train_loss_tensor.item() / world_size                 
                        
                        # CREATING TENSORBOARD DICTIONARIES:
                        # - creating dictionary to store training, standard validation and mean validation losses
                        train_val_dict = {
                            f"train-loss-{current_clique_size}": global_train_loss
                        }
                        # - creating dictionary to store validation losses for all task versions (will be logged to Tensorboard):
                        complete_val_dict = {}

                    # STANDARD VALIDATION LOSS (mirrors training loss and is used for early stopping):
                    # - Generate clique size value of each graph in the current batch
                    clique_size_array_stdval = gen_graphs.generate_batch_clique_sizes(
                        clique_size_list, training_parameters["num_val"]
                    )

                    # Generating standard validation data
                    full_stdval_data = gen_graphs.generate_batch(
                        training_parameters["num_val"],
                        graph_size,
                        clique_size_array_stdval,
                        p_correction_type,
                        input_magnification,
                    )
                    
                    # Split validation data across GPUs:
                    local_batch_size_val = training_parameters["num_val"] // world_size
                    start_idx_val = rank * local_batch_size_val
                    end_idx_val = (rank + 1) * local_batch_size_val                     
         
                    # Partition data for the current rank
                    stdval_data = (
                        torch.Tensor(full_stdval_data[0][start_idx_val:end_idx_val]).to(rank),
                        torch.Tensor(full_stdval_data[1][start_idx_val:end_idx_val]).to(rank),
                    )        

                    # Compute loss on standard validation set:
                    stdval_pred = model(stdval_data[0]).squeeze()
                    stdval_loss = criterion(stdval_pred.type(torch.float),torch.Tensor(stdval_data[1]).type(torch.float))

                    # Aggregate validation loss across GPUs:
                    stdval_loss_tensor = torch.tensor(stdval_loss.item(), device=rank)
                    torch.distributed.all_reduce(stdval_loss_tensor, op=torch.distributed.ReduceOp.SUM) 
                    global_stdval_loss = stdval_loss_tensor.item() / training_parameters["num_val"]
                    # Check early stopping condition:
                    early_stop = early_stopper.should_stop(global_stdval_loss)
                    
                    if rank == 0:                        
                        # storing standard validation loss in the training and validation losses dictionary:
                        train_val_dict[f"stdval-loss-{current_clique_size}"] = global_stdval_loss

                    # Free up memory for validation data
                    del stdval_pred, stdval_data
                    torch.cuda.empty_cache()

                    # VALIDATING MODEL ON ALL TASK VERSIONS:
                    for current_clique_size_val in clique_sizes:

                        # Generate clique size value of each graph in the current batch (in this case, we only need one value -> all graphs have the same clique size)
                        clique_size_array_val = gen_graphs.generate_batch_clique_sizes(
                            np.array([current_clique_size_val]),
                            training_parameters["num_val"],
                        )

                        # Generating validation graphs:
                        full_val_data = gen_graphs.generate_batch(
                            training_parameters["num_val"],
                            graph_size,
                            clique_size_array_val,
                            p_correction_type,
                            input_magnification,
                        )
                        # Partition data for the current rank                        
                        val_data = (
                            torch.Tensor(full_val_data[0][start_idx_val:end_idx_val]).to(rank),
                            torch.Tensor(full_val_data[1][start_idx_val:end_idx_val]).to(rank),
                        )                        
                        # Compute loss on validation set:
                        val_pred = model(val_data[0]).squeeze()
                        val_loss = criterion(
                            val_pred.type(torch.float),
                            torch.Tensor(val_data[1])
                            .type(torch.float)
                        )
                        
                        # Aggregate the validation loss across GPUs 
                        val_loss_tensor = torch.tensor(val_loss.item(), device=rank)
                        torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)                        

                        # updating dictionary with validation losses for all task versions:
                        if rank == 0:
                            global_val_loss = val_loss_tensor.item() / training_parameters["num_val"]
                            complete_val_dict[f"val-loss-{current_clique_size_val}"] = (
                                global_val_loss
                            )

                        # Free up memory for validation data
                        del val_pred, val_data
                        torch.cuda.empty_cache()

                    # End of validation on all task versions:
                    if rank == 0:
                        # Calculate mean validation loss:
                        mean_validation_loss = np.mean(list(complete_val_dict.values()))

                        # Check if checkpointing condition is met:
                        if checkpointer.should_save(mean_validation_loss):
                            save_model(model, model_name, graph_size, results_dir)

                        # Storing mean validation loss in the training and validation losses dictionary:
                        train_val_dict["mean-val-loss"] = mean_validation_loss

                        # Tensorboard logging:
                        # - plotting training, standard validation and mean validation losses:
                        writer.add_scalars(
                            f"{model_name}_train-stdval-meanval-losses",
                            train_val_dict,
                            saved_steps,
                        )
                        # - plotting validation losses for all task versions:
                        writer.add_scalars(
                            f"{model_name}_validation-losses",
                            complete_val_dict,
                            saved_steps,
                        )

                        # Flush the writer to make sure all data is written to disk
                        writer.flush()

                # Put model back in training mode after validation is completed
                model.train()

            # Check if early stopping condition is met (early stopper is defined only in rank 0)
            if early_stop:
                if rank==0:
                    # Print the reason for early stopping
                    if early_stopper.stop_reason == "min_loss":
                        print(
                            f"||||| Early stopping triggered: standard validation loss was below the exit value for {early_stopper.patience} consecutive validation steps."
                        )
                    elif early_stopper.stop_reason == "no_improvement":
                        print(
                            f"||||| Early stopping triggered: standard validation loss did not improve for {early_stopper.patience} consecutive validation steps."
                        )
                    else:
                        print(
                            "||||| Early stopping triggered for unknown reason. Check for mistakes in the code."
                        )
                    # Print the training step at which early stopping was triggered
                    print(
                        f"||||| Breaking out at training step number {int(training_step)} out of {num_training_steps}."
                    )
                # breaking out of clique size loop
                break

        # After clique size has finished training (here we are inside the clique size decreasing loop):

        # 1. Tensorboard: printing a vertical bar of 4 points in the plot, to separate the different task versions
        if rank == 0:        
            # - spacing values for the vertical lines:
            spacing_values = np.arange(0, 1.1, 0.10)
            # - dictionary with scalar values for the vertical lines:
            scalar_values = {
                f"vert-line-{round(value,2)}_{current_clique_size}": value
                for value in spacing_values
            }
        # - add the scalars to both writers (only from rank 0 process):
            writer.add_scalars(
                f"{model_name}_train-stdval-meanval-losses", scalar_values, saved_steps
            )
            writer.add_scalars(
                f"{model_name}_validation-losses", scalar_values, saved_steps
            )

        # 2. Printing a message to indicate the end of training for the current task version:
            print("||| Completed training for clique = ", current_clique_size)
            print(
                "||| ================================================================================="
            )

    # After all task versions have been trained:
    # - notify completion of training:
    if rank == 0:
        print(f"| Finished training {model_name}.")
    # - close the writer (opened on all GPUs):
    writer.close()


# TESTING FUNCTION:
def test_model(model, testing_parameters, graph_size, p_correction_type, model_name, world_size, rank):
    """
    Test the given model.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        testing_parameters (dict): A dictionary containing parameters for testing.
        graph_size (int): The size of the graph.
        p_correction_type (str): The type of p-correction.
        model_name (str): The name of the model.
        world_size (int): Integer indicating the number of GPUs used for training.
        rank: (int): The rank of the GPU that is currently being used        

    Returns:
        tuple: A tuple containing two dictionaries:
            - fraction_correct_results: A dictionary containing the results of testing for different clique sizes
              (The keys are the clique sizes and the values are the corresponding accuracies).
            - metrics_results: A dictionary containing various metrics calculated during testing.
    """

    # - INPUT TRANSFORMATION FLAGS:
    input_magnification = True if "CNN" in model_name else False

    # Notify start of testing:
    if rank == 0:
        print(f"| Started testing {model_name}...")

    # Create empty dictionaries for storing testing results:
    fraction_correct_results = {}  # Fraction correct for each clique size
    metrics_results = {}  # Metrics dictionary

    # Calculate max clique size (proportion of graph size):
    max_clique_size = int(
        testing_parameters["max_clique_size_proportion_test"] * graph_size
    )

    # Calculate array of clique sizes for all test curriculum:
    if max_clique_size < testing_parameters["clique_testing_levels"]:
        # If max clique size is less than the the number of test levels, use max clique size as the number of test levels
        clique_sizes = (
            np.linspace(max_clique_size, 1, num=min(max_clique_size, testing_parameters["clique_testing_levels"]))
            .astype(int)
        )

    # Metrics initialization (local to each GPU)
    TP, FP, TN, FN = 0, 0, 0, 0
    y_scores = []
    y_true = []

    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # Initialize fraction correct list, updated at each test iteration
        fraction_correct_list = []

        # Loop for testing iterations:
        for test_iter in range(testing_parameters["test_iterations"]):

            # Generate clique size value of each graph in the current batch
            clique_size_array_test = gen_graphs.generate_batch_clique_sizes(
                np.array([current_clique_size]),
                testing_parameters["num_test"],
            )

            # Generate validation graphs
            full_test_data = gen_graphs.generate_batch(
                testing_parameters["num_test"],
                graph_size,
                clique_size_array_test,
                p_correction_type,
                input_magnification,
            )

            # Split test data across GPUs
            local_batch_size_test = testing_parameters["num_test"] // world_size
            start_idx_test = rank * local_batch_size_test
            end_idx_test = (rank + 1) * local_batch_size_test            

            # Partition data for the current rank
            test_data = (
                torch.Tensor(full_test_data[0][start_idx_test:end_idx_test]).to(rank),
                torch.Tensor(full_test_data[1][start_idx_test:end_idx_test]).to(rank),
            )
            
            # Perform prediction on test data
            soft_output = model(test_data[0]).squeeze()

            # Update global metrics for AUC-ROC
            y_scores.extend(soft_output.cpu().tolist())
            y_true.extend(test_data[1].cpu().tolist())

            # Convert soft predictions to hard predictions
            hard_output = (soft_output > 0.5).float()

            # Compute metrics
            TP += ((hard_output == 1) & (test_data[1] == 1)).sum().item()
            FP += ((hard_output == 1) & (test_data[1] == 0)).sum().item()
            TN += ((hard_output == 0) & (test_data[1] == 0)).sum().item()
            FN += ((hard_output == 0) & (test_data[1] == 1)).sum().item()

            # Calculate fraction correct for current test iteration
            fraction_correct = (hard_output == test_data[1]).float().mean().item()
            fraction_correct_list.append(fraction_correct)

            # Free memory after this iteration
            del soft_output, hard_output, test_data
            torch.cuda.empty_cache()

        # After all test iterations for the current clique size:
        # - calculate total fraction correct for each rank
        total_fraction_correct = torch.tensor(
            sum(fraction_correct_list) / len(fraction_correct_list),
            device=rank,
        )     
        # - aggregate fraction correct across GPUs
        torch.distributed.all_reduce(total_fraction_correct, op=torch.distributed.ReduceOp.SUM)
        total_fraction_correct /= world_size
        
        # Store aggregated results (only rank 0 updates dictionary)
        if rank == 0:        
            fraction_correct_results[current_clique_size] = round(total_fraction_correct.item(), 2)
            print(
                f"||| Completed testing for clique = {current_clique_size}. "
                f"Average fraction correct = {fraction_correct_results[current_clique_size]}"
            )
            print("|||===========================================================")

    # - retrieve TP, FP, TN and FN from each GPU
    TP = torch.tensor(TP, device=rank)
    FP = torch.tensor(FP, device=rank)
    TN = torch.tensor(TN, device=rank)
    FN = torch.tensor(FN, device=rank)
    # - aggregate across GPUs
    torch.distributed.all_reduce(TP, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(FP, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(TN, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(FN, op=torch.distributed.ReduceOp.SUM)

    # Compute global metrics (on rank 0)
    if rank == 0:
        precision = TP.item() / (TP.item() + FP.item() + 1e-10)
        recall = TP.item() / (TP.item() + FN.item() + 1e-10)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        AUC_ROC = roc_auc_score(y_true, y_scores)

        metrics_results = {
            "TP": TP.item(),
            "FP": FP.item(),
            "TN": TN.item(),
            "FN": FN.item(),
            "precision": precision,
            "recall": recall,
            "F1": F1,
            "AUC_ROC": AUC_ROC,
        }

        # Print final metrics
        print(f"| Finished testing {model_name}. Metrics: {metrics_results}")

    return fraction_correct_results, metrics_results if rank == 0 else None
