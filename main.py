import datetime
import os
import unittest
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# custom imports
from src.utils import (
    load_config,
    load_model,
    save_exp_config,
    save_partial_time,
    save_test_results,
)
from src.train_test import (
    train_model,
    test_model,
)
from src.tensorboard_save import tensorboard_save_images

# loading experiment configuration file:
config = load_config(
    os.path.join("docs", "vit_exp_config.yml")
)  # CHANGE THIS TO PERFORM DIFFERENT EXPERIMENTS


# Defining tests:
def tests():

    # Define the directory where test files are located
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tests")

    # Discover and run all tests in the `tests` directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=test_dir, pattern="test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests and capture the result
    result = test_runner.run(test_suite)

    # Stop the main script if tests fail
    if not result.wasSuccessful():
        print("Some tests failed. Aborting experiment.")
        sys.exit(1)  # Exit with error code if tests failed
    else:
        print("All tests passed. Proceeding with the experiment.")

    # - making sure processes are synchronized on all devices before moving on
    torch.distributed.barrier()


# Defining full experiment:
def full_exp():

    # DDP:
    rank = (
        torch.distributed.get_rank()
    )  # identifies processes (in this context, one process per GPU)
    device_id = rank % torch.cuda.device_count()
    print(f"Running full experiment on device id: {device_id}.")
    world_size = torch.cuda.device_count()

    # storing starting time of the experiment in string format:
    start_time = datetime.datetime.now()
    start_time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # storing current directory:
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # storing the name of the experiment
    exp_name_with_time = f"{config['exp_name']}_{start_time_string}"

    # creating folder in "results/data" folder to save the results of the whole experiment
    experiment_results_dir = os.path.join(
        current_dir,
        "results",
        "data",
        exp_name_with_time,
    )
    if rank == 0:
        os.makedirs(experiment_results_dir)

    # creating experiment folder in "runs" folder
    experiment_runs_dir = os.path.join(
        current_dir,
        "runs",
        exp_name_with_time,
    )

    # looping over the different graph sizes in the experiment:
    for graph_size in config["graph_size_values"]:

        # printing separation line and graph size
        if rank == 0:
            print("-----------------------------------")
            print(f" GRAPH SIZE: {graph_size}")

        # create a new directory in "runs/" for each graph size value
        Nvalue_runs_dir = os.path.join(experiment_runs_dir, f"N{graph_size}")
        # create a new writer for each graph size value and point it to the correct log directory
        writer = SummaryWriter(log_dir=Nvalue_runs_dir)

        # saving input images to tensorboard:
        if rank == 0:
            tensorboard_save_images(
                writer,
                graph_size,
                config["p_correction_type"],
                num_images=10,
            )

        # inside experiment results folder, create a new directory for each graph size value:
        graph_size_results_dir = os.path.join(experiment_results_dir, f"N{graph_size}")

        if rank == 0:
            os.makedirs(graph_size_results_dir)

        # loading, training, and testing models:
        for model_specs in config["models"]:

            # creating model subfolder in current graph size folder:
            model_results_dir = os.path.join(
                graph_size_results_dir,
                model_specs["model_name"],
            )
            if rank == 0:
                os.makedirs(model_results_dir)
                # printing model name
                print(model_specs["model_name"])

            # loading model:
            model = load_model(
                model_specs,
                graph_size,
                config["training_parameters"]["num_train"], # used for torchinfo summary
                # DDP:
                world_size,
                rank,
                device_id,
            )  
            # - making sure processes are synchronized on all devices
            torch.distributed.barrier()                             

            # put model in training mode
            model.train()

            # training model and visualizing training progression on Tensorboard
            train_model(
                model,
                config["training_parameters"],
                graph_size,
                config["p_correction_type"],
                writer,
                model_specs["model_name"],
                model_results_dir,
                # DDP:
                world_size,
                rank,
                device_id,
            )

            # load the best model from the training process on all ranks
            # - defining file name and path:
            file_path = os.path.join(
                model_results_dir,
                f"{model_specs['model_name']}_N{graph_size}_trained.pth",
            )
            # - making sure processes are synchronized on all devices
            torch.distributed.barrier()
            # - configuring map location:
            map_location = {"cuda:%d" % 0: "cuda:%d" % device_id}

            # - loading the model:
            state_dict = torch.load(file_path, map_location=map_location)
            model.load_state_dict(state_dict)

            # - putting the model in evaluation mode before starting testing:
            model.eval()

            # testing best model
            fraction_correct_results, metrics_results = test_model(
                model,
                config["testing_parameters"],
                graph_size,
                config["p_correction_type"],
                model_specs["model_name"],
                # DDP:
                world_size,
                rank,
                device_id,
            )
            # - making sure processes are synchronized on all devices
            torch.distributed.barrier()

            # saving test results as csv file
            if rank == 0:
                save_test_results(
                    fraction_correct_results,
                    metrics_results,
                    model_specs["model_name"],
                    graph_size,
                    model_results_dir,
                )

            # TODO: implement GradCAM for CNN model + Attention Visualization for ViT model (otherwise performed on saved model after training is completed)

            # deleting model from device to free up memory:
            del model
            torch.cuda.empty_cache()

        if rank == 0:

            # Saving .yml file with time elapsed from the start of the experiment (to calculate the time needed for each graph size value):
            current_time = datetime.datetime.now()
            save_partial_time(
                graph_size,
                graph_size_results_dir,
                exp_name_with_time,
                start_time,
                current_time,
            )

        # synchronizing all processes before moving on with training:
        torch.distributed.barrier()

    # saving copy of the configuration file in the experiment folder, adding the time elapsed from the start of the experiment:
    end_time = datetime.datetime.now()
    if rank == 0:
        save_exp_config(
            config,
            experiment_results_dir,
            exp_name_with_time,
            start_time,
            end_time,
        )


# CALLING FUNCTION TO RUN EXPERIMENT:
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # DDP (here, using one process per GPU):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")  # process group initialization

    # running tests:
    tests()

    # running exp:
    full_exp()

    # DDP:
    torch.distributed.destroy_process_group()
