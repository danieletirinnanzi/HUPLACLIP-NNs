import yaml
import torch
import csv

# custom imports
from .models import Models

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
def load_model(model_name, graph_size, hyperparameters):

    # - building requested model
    match model_name:
        case "MLP":
            model = Models.mlp(graph_size, hyperparameters)
        case "CNN":
            model = Models.cnn(graph_size, hyperparameters)
        case "VGG16":
            model = Models.vgg16()
        case _:
            raise ValueError("Model not found")

    # - sending model to device:
    model.to(device)
    print("- Model loaded successfully.")
    return model


# Save results in csv file:
def save_test_results(test_results_dict, model_name, graph_size, results_dir):

    # START OF TESTS:

    # Check if test_results_dict is passed correctly
    if not isinstance(test_results_dict, dict):
        raise ValueError("test_results_dict should be a dictionary")

    # Check if model_name is passed correctly
    if not model_name:
        raise ValueError("Model name not provided")

    # Check if graph_size is passed correctly
    if not graph_size:
        raise ValueError("Graph size not provided")

    # Check if directory is passed correctly
    if not results_dir:
        raise ValueError("Results directory not provided")

    # END OF TESTS

    # Saving .csv file in "results" folder (already exists, created in main.py)
    # - defining file name:
    file_name = f"{model_name}_N{graph_size}_results"
    # - defining file path:
    file_path = f"{results_dir}/{file_name}.csv"

    # - saving the dictionary as a .csv file:
    with open(file_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["clique size", "fraction correct"])  # Add column labels
        for key, value in test_results_dict.items():
            writer.writerow([key, value])

    print(f"- Results saved successfully in {file_path}.")


# Save trained model as .pth file:
def save_trained_model(trained_model, model_name, graph_size, results_dir):

    # START OF TESTS

    # Check if trained_model is passed correctly (NOTE: make more stringent checks if needed)
    if not trained_model:
        raise ValueError("Trained model not provided")

    # Check if model_name is passed correctly
    if not model_name:
        raise ValueError("Model name not provided")

    # Check if graph_size is passed correctly
    if not graph_size:
        raise ValueError("Graph size not provided")

    # Check if directory is passed correctly
    if not results_dir:
        raise ValueError("Results directory not provided")

    # END OF TESTS

    # Saving .pth file in "results" folder (already exists, created in main.py)

    # - defining file name:
    file_name = f"{model_name}_N{graph_size}_trained"
    # - defining file path:
    file_path = f"{results_dir}/{file_name}.pth"

    # - saving the trained model as a .pth file:
    torch.save(trained_model.state_dict(), file_path)

    print(f"- Trained model saved successfully in {file_path}.")
