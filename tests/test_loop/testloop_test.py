import torch
import os
import csv
import pandas as pd

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

from src.utils import load_config
from src.train_test import test_model
from src.models import (
    MLP,
    CNN,
    # CNN_rudy,
    # VGG16_scratch,
    # VGG16_pretrained,
    # ResNet50_scratch,
    # ResNet50_pretrained,
    # GoogLeNet_scratch,
    # GoogLeNet_pretrained,
    FlexiViT_scratch,
    FlexiViT_pretrained,
)

# TO RUN as module: FROM HOME DIRECTORY -> python -m tests.test_loop.testloop_test

# Setting device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

config = load_config(os.path.join("docs", "grid_exp_config.yml"))  # CHANGE THIS

model_name = "CNN_small_2"  # CHANGE THIS

if "CNN" in model_name:
    # reading model configuration:
    model_config = [
        model for model in config["models"] if model["model_name"] == model_name
    ][0]
    # creating the model:
    model = CNN(config["graph_size_values"][0], model_config["architecture"])
    # # loading state dictionary:
    # # - from \tests folder
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             current_dir,
    #             "..",
    #             "train_loop",
    #             "mock_results",
    #             model_name,
    #             f"{model_name}_N{config['graph_size_values'][0]}_trained.pth",
    #         )
    #     )
    # )
    
    # - from \results folder
    model.load_state_dict(
        torch.load(
            os.path.join(
                current_dir,
                "..",
                "..",
                "results",
                "data",
                "grid_exp_2024-10-16_20-04-02",
                "N100",
                model_name,
                f"{model_name}_N{config['graph_size_values'][0]}_trained.pth",
            )
        )
    )
elif "ViT" in model_name:
    # Initialize the model (choosing lower graph size for testing)
    model = FlexiViT_pretrained(config["graph_size_values"][2])
    # # loading state dictionary:
    # # - from \tests folder
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             current_dir,
    #             "..",
    #             "train_loop",
    #             "mock_results",
    #             model_name,
    #             f"{model_name}_N{config['graph_size_values'][0]}_trained.pth",
    #         )
    #     )
    # )
    
    # # - from \results folder
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             current_dir,
    #             "..",
    #             "..",
    #             "results",
    #             "data",
    #             "grid_exp_2024-10-16_20-04-02",
    #             "N100",
    #             model_name,
    #             f"{model_name}_N{config['graph_size_values'][0]}_trained.pth",
    #         )
    #     )
    # )


# Sending model to device:
model.to(device)

# Put the model in eval mode
model.eval()

# Test the model
fraction_correct_results, metrics_results = test_model(
    model,
    config["testing_parameters"],
    config["graph_size_values"][0],
    config["p_correction_type"],
    model_name,
)

# Saving accuracy results in .csv file:
# - defining file name and path:
file_path = os.path.join(
    current_dir, f"{model_name}_N{config['graph_size_values'][0]}_fraction_correct.csv"
)
# - saving the dictionary as a .csv file:
with open(file_path, "w") as file:
    writer = csv.writer(file)
    writer.writerow(["clique size", "fraction correct"])  # Add column labels
    for key, value in fraction_correct_results.items():
        writer.writerow([key, value])

# Saving metrics results in .csv file:
# - defining file name and path:
file_path = os.path.join(
    current_dir, f"{model_name}_N{config['graph_size_values'][2]}_metrics.csv"
)
# - saving the dictionary as a .csv file:
pd.DataFrame([metrics_results]).to_csv(file_path, index=False)

print(f"- Results saved successfully in {current_dir}.")
