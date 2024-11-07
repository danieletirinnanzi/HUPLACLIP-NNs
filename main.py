import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter

# custom imports
from src.utils import (
    load_config,
    load_model,
    save_exp_config,
    save_partial_time,
    save_test_results,
    save_features,
)
from src.train_test import (
    train_model,
    test_model,
)
from tests.run_tests import run_all_tests
from src.tensorboard_save import tensorboard_save_images

# defining device and cleaning cache:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# loading experiment configuration file:
config = load_config(os.path.join("docs", "cnn_exp_config.yml"))   # CHANGE THIS TO PERFORM DIFFERENT EXPERIMENTS

# running all tests before running the experiment:
run_all_tests()

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
    print("-----------------------------------")
    print(f" GRAPH SIZE: {graph_size}")

    # create a new directory in "runs/" for each graph size value
    Nvalue_runs_dir = os.path.join(experiment_runs_dir, f"N{graph_size}")
    # create a new writer for each graph size value and point it to the correct log directory
    writer = SummaryWriter(log_dir=Nvalue_runs_dir)

    # saving images to tensorboard (no input transformation, like MLP input):
    tensorboard_save_images(
        writer,
        graph_size,
        config["p_correction_type"],
        num_images=10,
    )

    # inside experiment results folder, create a new directory for each graph size value:
    graph_size_results_dir = os.path.join(experiment_results_dir, f"N{graph_size}")
    os.makedirs(graph_size_results_dir)

    # loading, training, and testing models:
    for model_specs in config["models"]:

        # creating model subfolder in current graph size folder:
        model_results_dir = os.path.join(
            graph_size_results_dir,
            model_specs["model_name"],
        )
        os.makedirs(model_results_dir)

        # printing model name
        print(model_specs["model_name"])

        # loading model
        model = load_model(
            model_specs,
            graph_size,
            device,
        )

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
        )

        # load the best model from the training process
        # - defining file name and path:
        file_path = os.path.join(
            model_results_dir,
            f"{model_specs['model_name']}_N{graph_size}_trained.pth",
        )
        # - loading the model:
        model.load_state_dict(torch.load(file_path))
        # - sending the model to the device:
        model.to(device)
        # - putting the model in evaluation mode:
        model.eval()

        # testing best model
        fraction_correct_results, metrics_results = test_model(
            model,
            config["testing_parameters"],
            graph_size,
            config["p_correction_type"],
            model_specs["model_name"],
        )

        # saving test results as csv file
        save_test_results(
            fraction_correct_results,
            metrics_results,
            model_specs["model_name"],
            graph_size,
            model_results_dir,
        )

        # TODO: ADAPT TO NEW CNNs
        # # when possible, saving the features extracted by the model:
        # if model_specs["model_name"] in [
        #     "CNN_small_1",
        #     "CNN_small_2",
        #     "CNN_medium_1",
        #     "CNN_medium_2",
        #     "CNN_large_1",
        #     "CNN_large_2",
        # ]:
        #     save_features(
        #         model,
        #         model_specs["model_name"],
        #         graph_size,
        #         config["p_correction_type"],
        #         model_results_dir,
        #         device,
        #     )

        # deleting model from device to free up memory:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # At the end of each graph size value, saving .yml file with time elapsed from the start of the experiment (to calculate the time needed for each graph size value):
    current_time = datetime.datetime.now()
    save_partial_time(
        graph_size,
        graph_size_results_dir,
        exp_name_with_time,
        start_time,
        current_time,
    )

# saving copy of the configuration file in the experiment folder, adding the time elapsed from the start of the experiment:
end_time = datetime.datetime.now()
save_exp_config(
    config,
    experiment_results_dir,
    exp_name_with_time,
    start_time,
    end_time,
)
