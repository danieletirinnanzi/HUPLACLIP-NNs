import datetime
import os
import torch

from torch.utils.tensorboard import SummaryWriter

# custom imports
from src.utils import load_config
from src.utils import load_model
from src.utils import save_exp_config
from src.utils import save_test_results
from src.utils import save_trained_model
from src.utils import save_features
from src.train_test import train_model
from src.train_test import test_model
from src.tensorboard_save import (
    tensorboard_save_images,
    # # MODEL SAVING (not working):
    # ModelsWrapper,
    # tensorboard_save_models,
)

# loading experiment configuration file:
config = load_config(os.path.join("docs", "nndl_exp_config.yml"))

# storing starting time of the experiment in string format:
start_time = datetime.datetime.now()
start_time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S")

# storing current directory:
current_dir = os.path.dirname(os.path.realpath(__file__))

# storing the name of the experiment
exp_name_with_time = f"{config['exp_name']}_{start_time_string}"

# creating folder in "results/data" folder to save the results of the whole experiment
experiment_results_dir = os.path.join(
    current_dir, "results", "data", exp_name_with_time
)
os.makedirs(experiment_results_dir)

# creating experiment folder in "runs" folder
experiment_runs_dir = os.path.join(current_dir, "runs", exp_name_with_time)

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
    # os.makedirs(graph_size_results_dir)

    # loading, training, and testing models:
    for model_specs in config["models"]:

        # printing model name
        print(model_specs["model_name"])

        # loading model
        model = load_model(model_specs, graph_size)

        # put model in training mode
        model.train()

        # training model and visualizing training progression on Tensorboard
        trained_model = train_model(
            model,
            config["training_parameters"],
            graph_size,
            config["p_correction_type"],
            writer,
            model_specs["model_name"],
        )

        # put trained model in evaluation mode
        trained_model.eval()

        # testing trained model
        test_results = test_model(
            trained_model,
            config["testing_parameters"],
            graph_size,
            config["p_correction_type"],
            model_specs["model_name"],
        )

        # creating model subfolder in current graph size folder:
        model_results_dir = os.path.join(
            graph_size_results_dir, model_specs["model_name"]
        )
        os.makedirs(model_results_dir)

        # - saving test results as csv file
        save_test_results(
            test_results,
            model_specs["model_name"],
            graph_size,
            model_results_dir,
        )

        # - saving the trained model (will not be synched with git due to size)
        save_trained_model(
            trained_model,
            model_specs["model_name"],
            graph_size,
            model_results_dir,
        )

        # - for CNN, VGG and ResNet, saving the features extracted by the model in the results folder:
        if model_specs["model_name"] in [
            "CNN_small",
            "CNN_medium",
            "CNN_large",
            "VGG16scratch",
            "VGG16pretrained",
            "ResNet50scratch",
            "ResNet50pretrained",
            "GoogLeNetscratch",
            "GoogLeNetpretrained",
        ]:
            save_features(
                trained_model,
                model_specs["model_name"],
                graph_size,
                config["p_correction_type"],
                model_results_dir,
            )

# saving copy of the configuration file in the experiment folder just created (to keep track of the experiment settings), adding an indication regarding the elapsed time from the start of the experiment:
end_time = datetime.datetime.now()
save_exp_config(
    config, experiment_results_dir, exp_name_with_time, start_time, end_time
)


# -----------------------------DO NOT UNCOMMENT THIS, NOT WORKING

# # SAVING MODELS (NOT WORKING)
# # create empty dictionary to store models (used to store models for tensorboard saving, not working)
# models_dict = {}

# # saving model to dictionary (used to store models for tensorboard saving, not working)
# models_dict[model_specs["model_name"]] = model

# # saving single model to tensorboard (not working for VGG16, model name should be passed instead of model itself):
# # Got error: "RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[1, 1, 300, 300] to have 3 channels, but got 1 channels instead"
# tensorboard_save_models(writer, model, config["graph_size"])


# # saving all models to tensorboard (not working):
# # - creating wrapper class:
# models_wrapper = ModelsWrapper(models_dict)
# tensorboard_save_models(
#     writer,
#     models_wrapper,
#     config["graph_size"],
#     config["models"],
# )
