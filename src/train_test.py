# imports
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import datetime
from sklearn.metrics import roc_auc_score
import torch
import pickle
import time
import os
import sys

# defining random generator (used to define the clique size value of each graph in the batch during training)
random_generator = np.random.default_rng()

# custom import
import src.graphs_generation as gen_graphs
from src.utils import save_model, save_resume_progress, load_resume_progress, save_temp_checkpoint, load_temp_checkpoint, get_slurm_time_limit_seconds, get_slurm_elapsed_seconds

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
    Leaky integrator early stopper based on the validation loss of the current clique size. At each validation step, the running mean of the validation loss is updated.
    As long as this running mean loss decreases, training continues.
    If this running mean loss does not decrease significantly / is below the exit value for "patience" consecutive steps, training is interrupted.

    Attributes:
        running_mean_val_loss (float): The running mean of the validation loss.
        alpha (float): The leak rate of the integrator.
        patience (int): Number of steps with subsequent "validation loss increase" / "validation loss under exit value" before stopping the training.

        # INCREASE LOSS STOPPING:
        min_delta (float): Minimum deviation in the running mean to qualify as significant.
        val_increase_counter (int): Counter that increments if the loss increases by a significant amount (this counter is compared to the patience).

        # EXIT LOSS STOPPING:
        val_exit_loss (float): The validation loss under which training can stop.
        val_exit_counter (int): Counter that increments if the validation loss is below the exit value (this counter is compared to the patience).

        stop_reason (str): The reason for stopping the training.
    """

    def __init__(self, alpha, patience, min_delta, val_exit_loss):

        # Testing input validity:
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha should be a float between 0 and 1.")
        if patience <= 0:
            raise ValueError("patience should be a positive integer.")
        if min_delta < 0:
            raise ValueError("min_delta should be a non-negative float.")
        if val_exit_loss < 0:
            raise ValueError("val_exit_loss should be a non-negative float.")

        # Initializations:
        self.alpha = alpha
        self.patience = patience
        self.running_mean_val_loss = None
        # - Increase loss stopping:
        self.min_delta = min_delta
        self.val_increase_counter = 0
        # - Exit loss stopping:
        self.val_exit_counter = 0
        self.val_exit_loss = val_exit_loss
        # - Stop reason:
        self.stop_reason = None

    def should_stop(self, val_loss):
        """
        Determines whether the training should stop by udpating the running mean of the validation loss and comparing it to the one at the previous step.

        Args:
            val_loss(float): The validation loss observed at the current step (used to update the running mean).

        Returns:
            should_stop (bool): True if the training should stop, False otherwise.
            stop_reason (str): If should_stop is True, the reason for stopping the training.
        """

        # Testing input validity:
        if val_loss < 0:
            raise ValueError(
                "validation loss is negative, check for mistakes in the loss calculation."
            )

        # Updating the running mean loss:
        if self.running_mean_val_loss is None:
            # if this is the first validation step, set the running mean loss to the current validation loss and skip checks of early stopping conditions
            self.running_mean_val_loss = val_loss
        else:
            # update the running mean loss with the leaky integrator formula
            previous_running_mean_val_loss = self.running_mean_val_loss
            self.running_mean_val_loss = (self.alpha * self.running_mean_val_loss) + (
                (1 - self.alpha) * val_loss
            )
            
            # Checking the two stopping conditions:
            # - Increase loss stopping:
            if self.running_mean_val_loss >= previous_running_mean_val_loss - self.min_delta:
                # - if the monitored loss did NOT decrease by a significant amount compared to the previous value, increase counter and check if it is above the patience:
                self.val_increase_counter += 1
                if self.val_increase_counter >= self.patience:
                    self.stop_reason = "no_improvement"
                    return True       
            else:
                # - if the monitored loss decreased by a significant amount, reset counter:
                self.val_increase_counter = 0

            # - Exit loss stopping:
            if self.running_mean_val_loss < self.val_exit_loss:
                # - if the monitored loss is below the exit value, increase counter and check if it is above the patience:
                self.val_exit_counter += 1
                if self.val_exit_counter >= self.patience:
                    self.stop_reason = "min_loss"
                    return True
            else:
                # - if the current loss is above the exit value, reset counter:
                self.val_exit_counter = 0

            # if none of the stopping conditions are met, return False
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
    rank,
    device_id,
    resume=False,
    exp_name_with_time=None,
):
    """
    Trains a model using the specified hyperparameters, saving it as training progresses.
    Training is structured as a curriculum learning task, where the model is trained on graphs with decreasing clique sizes.
    Sketch of training structure:
    FOR (decreasing clique size):
        FOR (decreasing learning rate):
            - initialize early stopper
            FOR (training steps):
                - generate training data
                - forward pass on training data
                - compute loss on training data
                - backward pass and update weights
                - at regular intervals (save_step):
                    - generate validation data
                    - save errors and print to Tensorboard
                - check if checkpointing condition is met -> if yes, save model
                - check if early stopping condition is met -> if yes, stop training with current learning rate (move to lower lr/next clique size)
            END FOR
            - print "sparse" vertical bar in the Tensorboard plot to separate learning rates
        END FOR
        - print "thick" vertical bar in the Tensorboard plot to separate clique sizes
    END FOR

    Args:
        model (torch.nn.Module): The loaded model.
        training_parameters (dict): A dictionary containing all hyperparameters for training (they are read from configuration file).
        graph_size (int): The size of the graph.
        p_correction_type (str): The type of p-correction.
        writer: The Tensorboard writer.
        model_name (str): The name of the model.
        results_dir (str): The directory where the best model will be saved.
        world_size (int): Integer indicating the number of processes.
        rank: (int): The rank of the process.
        device_id: (int): The id of the device used by the process (in this context, each process uses a single GPU).
        resume (bool): whether training is resumed or not (see "Resume training explanation" below for detailed explanation).
        exp_name_with_time (str): if resume is True, this should be the name of the experiment folder (including date and time) from which training is resumed.

    Raises:
        ValueError: If the model is not provided, training_parameters is not a dictionary,
            graph_size is not a positive integer, p_correction_type is not a string, or writer is not provided.
    
    Resume training explanation
    The function can be called in two modes (defined by the "resume" flag):
    - "resume" == False: in this case, training is performed normally. The script contains a timer that triggers saving of a yaml file containing the current position in the training loops
     (at which clique size, learning rate and training step the script is) as well as a temporary model and optimizer checkpoints;
     - "resume" == True: in this case, training restarts from the point where it was interrupted. More specifically, the script:
        1. Reads the yaml file to that indicates where training was interrupted, and loads the checkpointed model; 
        2. Skips all completed clique sizes and learning rates;
        3. For the interrupted learning rate, skips completed steps and resumes from the last saved step (the last one to be saved to Tensorboard, so that logging continues normally);
        4. Model and optimizer states are restored from the checkpoint, continuing training;
        5. The early-stopper is re-initialized for the resumed learning rate (and for each new learning rate), so it is tracked from the current (resumed) point onwards.
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

    # Timer setup from SLURM
    training_loop_start_time = time.time()
    MAX_SECONDS = get_slurm_time_limit_seconds() - 300  # 5 min buffer
    
    # Resume logic: load progress and temp checkpoint model if resume
    progress = None
    temp_ckpt = None
    if resume and rank == 0:
        progress = load_resume_progress(results_dir, model_name, graph_size)
        temp_ckpt = load_temp_checkpoint(results_dir, model_name, graph_size, map_location={f"cuda:%d" % 0: f"cuda:%d" % device_id})
    # Broadcast progress and temp_ckpt info to all ranks
    progress_bytes = pickle.dumps(progress) if progress is not None else b''    # serialization of 'progress' object to bytes
    progress_size = torch.tensor([len(progress_bytes)], device=device_id)   # defining size of tensor to receive 'progress'
    torch.distributed.broadcast(progress_size, src=0)   # broadcasting receiving tensor on all devices
    if progress_size.item() > 0:
        # if progress data is present ("resume" case), fill tensor with serialized bytes
        progress_bytes_tensor = torch.zeros(progress_size.item(), dtype=torch.uint8, device=device_id)
        if rank == 0:
            progress_bytes_tensor[:] = torch.tensor(list(progress_bytes), dtype=torch.uint8, device=device_id)
        torch.distributed.broadcast(progress_bytes_tensor, src=0)
        if rank != 0:
            progress = pickle.loads(bytes(progress_bytes_tensor.tolist()))
    else:
        progress = None
    # (temp_ckpt is only needed on rank 0 for loading model/optimizer)

    # - INPUT TRANSFORMATION FLAGS:
    input_magnification = True if "CNN" in model_name else False

    # - NUMBER OF TRAINING STEPS:
    # reading number of training steps
    num_training_steps = int(training_parameters["num_training_steps"])

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
    # - calculating local batch sizes for training and validation data:
    # Checking divisibility of training batch size by world size
    if training_parameters["num_train"] % world_size != 0:
        raise ValueError(
            f"Training batch size of {training_parameters['num_train']} is not evenly divisible by world_size={world_size}. "
            f"Each rank requires an equal share of the data for DDP. Please adjust 'num_train' to be divisible by {world_size}."
        )    
    local_batch_size_train = training_parameters["num_train"] // world_size
    # Checking divisibility of validation batch size by world size
    if training_parameters["num_val"] % world_size != 0:
        raise ValueError(
            f"Validation batch size of {training_parameters['num_val']} is not evenly divisible by world_size={world_size}. "
            f"Each rank requires an equal share of the data for DDP. Please adjust 'num_val' to be divisible by {world_size}."
        )
    local_batch_size_val = training_parameters["num_val"] // world_size # batch size for validation data (both standard validation and validation on all task versions)    

    # Calculating array of clique sizes for all training curriculum:
    clique_sizes = np.linspace(
        max_clique_size,
        min_clique_size,
        num=training_parameters["clique_training_levels"],
    ).astype(int)

    # Resume: determine which clique sizes/lrs/steps to skip
    resume_clique_idx = 0
    resume_lr_idx = 0
    resume_step = 0
    if resume and progress is not None:
        resume_clique_idx = progress.get('clique_idx', 0)
        resume_lr_idx = progress.get('lr_idx', 0)
        resume_step = progress.get('step', 0)

    # initializing checkpointer (triggers model saving when mean validation loss is lower than the minimum seen so far):
    checkpointer = Checkpointer()

    # Notify start of training (only rank 0 logs)
    if rank == 0:
        print(f"| Started training {model_name}...")

    # Loop for decreasing clique sizes
    for i, current_clique_size in enumerate(clique_sizes):
        if resume and i < resume_clique_idx:
            continue  # skip completed clique sizes

        # Defining clique list for current clique size value:
        clique_size_list = clique_sizes[: i + 1]
        if rank == 0:
            print("||| Minimum clique size is now: ", current_clique_size)
            print("||| List of available clique sizes is now: ", clique_size_list)

        # Loop for decreasing learning rate
        for j, learning_rate in enumerate(training_parameters["learning_rates"]):
            if resume and i == resume_clique_idx and j < resume_lr_idx:
                continue  # skip completed lrs
            if rank == 0:
                print("||||| Learning rate is now: ", float(learning_rate))

            # initializing early stopper (triggers passage to following learning rate)
            early_stopper = EarlyStopper(
                alpha=training_parameters["alpha"],
                patience=training_parameters["patience"],
                min_delta=training_parameters["min_delta"],
                val_exit_loss=training_parameters["val_exit_loss"],
            )

            # reading optimizer and learning rate
            if training_parameters["optimizer"] == "Adam":
                optim = torch.optim.Adam(
                    model.parameters(), lr=float(learning_rate)
                )
            elif training_parameters["optimizer"] == "AdamW":
                optim = torch.optim.AdamW(
                    model.parameters(), lr=float(learning_rate)
                )
            elif training_parameters["optimizer"] == "SGD":
                optim = torch.optim.SGD(
                    model.parameters(),
                    lr=float(learning_rate),
                    momentum=0.9,  # default value is zero
                )
            else:
                raise ValueError("Optimizer not found")

            # Resume: load optimizer/model state if needed
            if resume and i == resume_clique_idx and j == resume_lr_idx and temp_ckpt is not None and rank == 0:
                model.load_state_dict(temp_ckpt['model_state_dict'])
                optim.load_state_dict(temp_ckpt['optimizer_state_dict'])
                saved_steps = temp_ckpt['step_info'].get('saved_steps', 0)
            elif not (resume and i == resume_clique_idx and j == resume_lr_idx):
                saved_steps = 0

            # Training steps loop:
            for training_step in range(training_parameters["num_training_steps"] + 1):
                early_stop = False  # flag for early stopping                
                # skip completed steps (NOTE: if resuming, early stopping is reset and running mean loss is recomputed from the current step onwards for the current learning rate; for subsequent learning rates, early stopper is always re-initialized)
                if resume and i == resume_clique_idx and j == resume_lr_idx and training_step < resume_step:
                    continue  
                # Creating placeholders to receive subset of training data
                # - Placeholder for graphs
                if input_magnification:
                    local_tensor_graphs_train = torch.zeros((local_batch_size_train, 1, 2400, 2400), device=device_id)
                else:
                    local_tensor_graphs_train = torch.zeros((local_batch_size_train, 1, graph_size, graph_size), device=device_id)
                # - Placeholder for labels
                local_tensor_labels_train = torch.zeros((local_batch_size_train), device=device_id)
                    
                # Generating training data on CPU (full batch) on rank 0 and scattering relevant data to separate GPUs
                if rank == 0:                
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
                    
                    # Create separate scatter lists for graphs and labels:
                    scatter_list_graphs_train = [torch.Tensor(full_train_data[0][i * local_batch_size_train:(i + 1) * local_batch_size_train]).to(device_id) for i in range(world_size)]
                    scatter_list_labels_train = [torch.Tensor(full_train_data[1][i * local_batch_size_train:(i + 1) * local_batch_size_train]).to(device_id) for i in range(world_size)]                    
                else:
                    full_train_data = None
                    scatter_list_graphs_train = None
                    scatter_list_labels_train = None

                # Scatter the lists to each process
                # - scattering graphs:
                torch.distributed.scatter(local_tensor_graphs_train, scatter_list=scatter_list_graphs_train, src=0) 
                # - scattering labels:
                torch.distributed.scatter(local_tensor_labels_train, scatter_list=scatter_list_labels_train, src=0)
                
                # Barrier to ensure all processes reach this point
                torch.distributed.barrier()                

                # Forward pass on training data (after this, train_loss is the loss for the local batch)
                train_pred = model(local_tensor_graphs_train).squeeze()
                train_loss = criterion(
                    train_pred.type(torch.float),
                    torch.Tensor(local_tensor_labels_train).type(torch.float),
                )

                # Backward pass
                train_loss.backward()  # DDP GRADIENT SYNCHRONIZATION HAPPENS HERE
                optim.step()
                optim.zero_grad(set_to_none=True)

                # Free up memory from training data
                del full_train_data, local_tensor_graphs_train, local_tensor_labels_train, scatter_list_graphs_train, scatter_list_labels_train
                torch.cuda.empty_cache()

                # Timer: save temp checkpoint and progress if time is almost up
                elapsed = get_slurm_elapsed_seconds(training_loop_start_time)
                if (MAX_SECONDS - elapsed) < 150:   # less than 2.5 min left
                    if rank == 0:  
                        # Save temp checkpoint and progress
                        step_info = {
                            'clique_idx': i,    # index of current clique size in curriculum
                            'lr_idx': j,    # index of current learning rate in curriculum
                            'step': training_step,  # current training step within the current learning rate
                            'saved_steps': saved_steps  # number of times results have been saved to TensorBoard (used for correct x-axis continuation)
                        }
                        save_temp_checkpoint(model, optim, step_info, results_dir, model_name, graph_size)
                        save_resume_progress(step_info, results_dir, model_name, graph_size)
                        print(f"[RANK 0] Saved temp checkpoint and progress at step {training_step} (time left: {MAX_SECONDS - elapsed}s)")
                    # Synchronize all ranks before exit
                    torch.distributed.barrier()
                    torch.distributed.destroy_process_group()
                    sys.exit(0)                    

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
                        train_loss_tensor = torch.tensor(
                            train_loss.item(), device=device_id
                        )
                        torch.distributed.all_reduce(
                            train_loss_tensor, op=torch.distributed.ReduceOp.SUM
                        )
                        # Storing global training loss (only rank 0 updates the dictionary)
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
                        # Split validation data across GPUs:
                        # Creating placeholders to receive subset of stdval data
                        # - Placeholder for graphs
                        if input_magnification:
                            local_tensor_graphs_stdval = torch.zeros((local_batch_size_val, 1, 2400, 2400), device=device_id)
                        else:
                            local_tensor_graphs_stdval = torch.zeros((local_batch_size_val, 1, graph_size, graph_size), device=device_id)
                        # - Placeholder for labels
                        local_tensor_labels_stdval = torch.zeros((local_batch_size_val), device=device_id)
                                                
                        # Generating standard validation data on CPU (full batch) on rank 0 and scattering relevant data to separate GPUs
                        if rank == 0:
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
                            
                            # Create separate scatter lists for graphs and labels:
                            scatter_list_graphs_stdval = [torch.Tensor(full_stdval_data[0][i * local_batch_size_val:(i + 1) * local_batch_size_val]).to(device_id) for i in range(world_size)]
                            scatter_list_labels_stdval = [torch.Tensor(full_stdval_data[1][i * local_batch_size_val:(i + 1) * local_batch_size_val]).to(device_id) for i in range(world_size)]                    
                        else:
                            full_stdval_data = None
                            scatter_list_graphs_stdval = None
                            scatter_list_labels_stdval = None                            

                        # Scatter the lists to each process
                        # - scattering graphs:
                        torch.distributed.scatter(local_tensor_graphs_stdval, scatter_list=scatter_list_graphs_stdval, src=0) 
                        # - scattering labels:
                        torch.distributed.scatter(local_tensor_labels_stdval, scatter_list=scatter_list_labels_stdval, src=0)
                        
                        # Barrier to ensure all processes reach this point
                        torch.distributed.barrier()   

                        # Compute loss on standard validation set:
                        stdval_pred = model(local_tensor_graphs_stdval).squeeze()
                        stdval_loss = criterion(
                            stdval_pred.type(torch.float),
                            torch.Tensor(local_tensor_labels_stdval).type(torch.float),
                        )

                        # Aggregate validation loss across GPUs:
                        stdval_loss_tensor = torch.tensor(
                            stdval_loss.item(), device=device_id
                        )
                        torch.distributed.all_reduce(
                            stdval_loss_tensor, op=torch.distributed.ReduceOp.SUM
                        )
                        global_stdval_loss = stdval_loss_tensor.item() / world_size
                        
                        # Check early stopping condition (it is based on the validation loss for the currently trained clique size):
                        early_stop = early_stopper.should_stop(global_stdval_loss)

                        if rank == 0:
                            # storing losses in the training and validation losses dictionary:
                            # - "standard validation" loss
                            train_val_dict[f"stdval-loss-{current_clique_size}"] = (
                                global_stdval_loss
                            )
                            # - "running mean validation" loss (used for early stopping)
                            train_val_dict["runningmean-stdval-loss"] = (early_stopper.running_mean_val_loss)

                        # Free up memory for validation data
                        del full_stdval_data, local_tensor_graphs_stdval, local_tensor_labels_stdval, scatter_list_graphs_stdval, scatter_list_labels_stdval
                        torch.cuda.empty_cache()

                        # VALIDATING MODEL ON ALL TASK VERSIONS:
                        for current_clique_size_val in clique_sizes:

                            # Creating placeholders to receive subset of training data
                            # - Placeholder for graphs
                            if input_magnification:
                                local_tensor_graphs_val = torch.zeros((local_batch_size_val, 1, 2400, 2400), device=device_id)
                            else:
                                local_tensor_graphs_val = torch.zeros((local_batch_size_val, 1, graph_size, graph_size), device=device_id)
                            # - Placeholder for labels
                            local_tensor_labels_val = torch.zeros((local_batch_size_val), device=device_id)

                            # Split validation data across GPUs:
                            if rank == 0:
                                
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

                                # Create separate scatter lists for graphs and labels:
                                scatter_list_graphs_val = [torch.Tensor(full_val_data[0][i * local_batch_size_val:(i + 1) * local_batch_size_val]).to(device_id) for i in range(world_size)]
                                scatter_list_labels_val = [torch.Tensor(full_val_data[1][i * local_batch_size_val:(i + 1) * local_batch_size_val]).to(device_id) for i in range(world_size)]                    
                            else:
                                full_val_data = None
                                scatter_list_graphs_val = None
                                scatter_list_labels_val = None
                                
                            # Scatter the lists to each process
                            # - scattering graphs:
                            torch.distributed.scatter(local_tensor_graphs_val, scatter_list=scatter_list_graphs_val, src=0) 
                            # - scattering labels:
                            torch.distributed.scatter(local_tensor_labels_val, scatter_list=scatter_list_labels_val, src=0)
                            
                            # Barrier to ensure all processes reach this point
                            torch.distributed.barrier()                                   
                                
                            # Compute loss on validation set:
                            val_pred = model(local_tensor_graphs_val).squeeze()
                            val_loss = criterion(
                                val_pred.type(torch.float),
                                torch.Tensor(local_tensor_labels_val).type(torch.float),
                            )

                            # Aggregate the validation loss across GPUs
                            val_loss_tensor = torch.tensor(
                                val_loss.item(), device=device_id
                            )
                            torch.distributed.all_reduce(
                                val_loss_tensor, op=torch.distributed.ReduceOp.SUM
                            )

                            # updating dictionary with validation losses for all task versions:
                            if rank == 0:
                                global_val_loss = val_loss_tensor.item() / world_size
                                complete_val_dict[f"val-loss-{current_clique_size_val}"] = (
                                    global_val_loss
                                )

                            # Free up memory for validation data
                            del full_val_data, local_tensor_graphs_val, local_tensor_labels_val, scatter_list_graphs_val, scatter_list_labels_val
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

                # Check if early stopping condition is met
                if early_stop:
                    if rank == 0:
                        # Print the reason for early stopping
                        if early_stopper.stop_reason == "min_loss":
                            print(
                                f"||||||| Early stopping triggered: standard validation loss was below the exit value for {early_stopper.patience} consecutive validation steps."
                            )
                        elif early_stopper.stop_reason == "no_improvement":
                            print(
                                f"||||||| Early stopping triggered: standard validation loss did not decrease for {early_stopper.patience} consecutive validation steps."
                            )
                        else:
                            print(
                                "||||||| Early stopping triggered for unknown reason. Check for mistakes in the code."
                            )
                        # Print the training step at which early stopping was triggered
                        print(
                            f"||||||| Breaking out at training step number {int(training_step)} out of {num_training_steps}."
                        )
                    # interrupting training for current learning rate
                    break
            
            # When this runs, training steps loop has finished (either due to early stopping or reaching the maximum number of training steps)
            if rank == 0:
                # 1. Tensorboard logging: printing a vertical bar of 10 points in the plot, to separate the learning rates
                # - defining y values for the vertical lines:
                spacing_values = np.linspace(0, 1.5, 10)
                # - dictionary with scalar values for the vertical lines:
                scalar_values = {
                    f'vert-line-{round(value,2)}_{current_clique_size}_{float(learning_rate)}': value
                    for value in spacing_values
                }
                # - add the scalars to both writers (only from rank 0 process):
                writer.add_scalars(
                    f'{model_name}_train-stdval-meanval-losses', scalar_values, saved_steps
                )
                writer.add_scalars(
                    f'{model_name}_validation-losses', scalar_values, saved_steps
                )    
                # 2. Printing a message to indicate the end of training for the current learning rate:
                print("||||| Completed training for learning rate = ", float(learning_rate))
                print(
                    "||||| ---------------------------------------------------------------------------------"
                )

        # When this runs, learning rate loop has finished (all learning rates have been used for the current clique size)
        if rank == 0:
            # 1. Tensorboard logging: printing a vertical bar of 20 points in the plot, to separate the clique size values
            # - defining y values for the vertical lines:
            spacing_values = np.linspace(0, 1.5, 20)
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
        print(f"| Finished training {model_name} at {datetime.datetime.now()}.")
    # - close the writer (opened on all GPUs):
    writer.close()


# TESTING FUNCTION:
def test_model(
    model,
    testing_parameters,
    graph_size,
    p_correction_type,
    model_name,
    world_size,
    rank,
    device_id,
):
    """
    Test the given model.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        testing_parameters (dict): A dictionary containing parameters for testing.
        graph_size (int): The size of the graph.
        p_correction_type (str): The type of p-correction.
        model_name (str): The name of the model.
        world_size (int): Integer indicating the number of processes.
        rank: (int): The rank of the process.
        device_id: (int): The id of the device used by the process (in this context, each process uses a single GPU).

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
    
    # Checking divisibility of test batch size by world size
    if testing_parameters["num_test"] % world_size != 0:
        raise ValueError(
            f"Batch size of {testing_parameters['num_test']} is not evenly divisible by world_size={world_size}. "
            f"Each rank requires an equal share of the data for DDP. Please adjust 'num_test' to be divisible by {world_size}."
        )
    local_batch_size_test = testing_parameters["num_test"] // world_size

    # Calculate array of clique sizes for all test curriculum
    # NOTE: if max clique size is smaller than the the number of test levels, use max clique size as the number of test levels
    clique_sizes = np.linspace(
        max_clique_size,
        1,
        num=min(max_clique_size, testing_parameters["clique_testing_levels"]),
    ).astype(int)

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

            # Creating placeholders to receive subset of training data
            # - Placeholder for graphs
            if input_magnification:
                local_tensor_graphs_test = torch.zeros((local_batch_size_test, 1, 2400, 2400), device=device_id)
            else:
                local_tensor_graphs_test = torch.zeros((local_batch_size_test, 1, graph_size, graph_size), device=device_id)
            # - Placeholder for labels
            local_tensor_labels_test = torch.zeros((local_batch_size_test), device=device_id)

            if rank == 0:
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
                
                # Create separate scatter lists for graphs and labels:
                scatter_list_graphs_test = [torch.Tensor(full_test_data[0][i * local_batch_size_test:(i + 1) * local_batch_size_test]).to(device_id) for i in range(world_size)]
                scatter_list_labels_test = [torch.Tensor(full_test_data[1][i * local_batch_size_test:(i + 1) * local_batch_size_test]).to(device_id) for i in range(world_size)]                    
            else:
                full_test_data = None
                scatter_list_graphs_test = None
                scatter_list_labels_test = None
            
            # Scatter the lists to each process
            # - scattering graphs:
            torch.distributed.scatter(local_tensor_graphs_test, scatter_list=scatter_list_graphs_test, src=0) 
            # - scattering labels:
            torch.distributed.scatter(local_tensor_labels_test, scatter_list=scatter_list_labels_test, src=0)
            
            # Barrier to ensure all processes reach this point
            torch.distributed.barrier()  

            # Perform prediction on test data
            soft_output = model(local_tensor_graphs_test).squeeze()

            # Update global metrics for AUC-ROC
            y_scores.extend(soft_output.cpu().tolist())
            y_true.extend(local_tensor_labels_test.cpu().tolist())

            # Convert soft predictions to hard predictions
            hard_output = (soft_output > 0.5).float()

            # Compute metrics
            TP += ((hard_output == 1) & (local_tensor_labels_test == 1)).sum().item()
            FP += ((hard_output == 1) & (local_tensor_labels_test == 0)).sum().item()
            TN += ((hard_output == 0) & (local_tensor_labels_test == 0)).sum().item()
            FN += ((hard_output == 0) & (local_tensor_labels_test == 1)).sum().item()

            # Calculate fraction correct for current test iteration
            fraction_correct = (hard_output == local_tensor_labels_test).float().mean().item()
            fraction_correct_list.append(fraction_correct)

            # Free memory after this iteration
            del full_test_data, local_tensor_graphs_test, local_tensor_labels_test, scatter_list_graphs_test, scatter_list_labels_test
            torch.cuda.empty_cache()

        # After all test iterations for the current clique size:
        # - calculate total fraction correct for each rank
        total_fraction_correct = torch.tensor(
            sum(fraction_correct_list) / len(fraction_correct_list),
            device=device_id,
        )
        # - aggregate fraction correct across GPUs
        torch.distributed.all_reduce(
            total_fraction_correct, op=torch.distributed.ReduceOp.SUM
        )
        total_fraction_correct /= world_size

        # Store aggregated results (only rank 0 updates dictionary)
        if rank == 0:
            fraction_correct_results[current_clique_size] = round(
                total_fraction_correct.item(), 2
            )
            print(
                f"||| Completed testing for clique = {current_clique_size}. "
                f"Average fraction correct = {fraction_correct_results[current_clique_size]}"
            )
            print("|||===========================================================")

    # - retrieve TP, FP, TN and FN from each GPU
    TP = torch.tensor(TP, device=device_id)
    FP = torch.tensor(FP, device=device_id)
    TN = torch.tensor(TN, device=device_id)
    FN = torch.tensor(FN, device=device_id)
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
        num_params = sum(
            p.numel() for p in model.parameters()
        )  # storing total number of parameters
        
        # DEBUG:
        print(f"Total number of parameters: {num_params}")

        metrics_results = {
            "TP": TP.item(),
            "FP": FP.item(),
            "TN": TN.item(),
            "FN": FN.item(),
            "precision": precision,
            "recall": recall,
            "F1": F1,
            "AUC_ROC": AUC_ROC,
            "total_params": num_params,
        }

        # Print final metrics
        print(f"| Finished testing {model_name} at {datetime.datetime.now()}. Metrics: {metrics_results}")

    return fraction_correct_results, metrics_results if rank == 0 else None
