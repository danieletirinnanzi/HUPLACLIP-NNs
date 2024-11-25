import yaml
import torch
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP

# custom imports
from .models import (
    MLP,
    CNN,
    ViT_scratch,
    ViT_pretrained,
)
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


# -----------------------------------------
# Loading experiment configuration file:
def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    print("Configuration file loaded successfully.")
    return config


# Loading model based on model name:
def load_model(model_specs, graph_size, device, local_rank=None):
    """
    Load and initialize a model based on the provided specifications.

    Args:
        model_specs (dict): Model specifications (e.g., name, architecture).
        graph_size (int): Size of the graph input to the model.
        device (torch.device): Device to load the model onto.
        local_rank (int, optional): GPU index for DDP. Required for multi-GPU setups.

    Returns:
        torch.nn.Module: The initialized model.
    """    
    model_name = model_specs["model_name"]

    # Build the requested model
    match model_name:
        case "MLP":
            model = MLP(graph_size, model_specs["architecture"])
        case "CNN_large":
            model = CNN(graph_size, model_specs["architecture"])
        case "ViTscratch":
            model = ViT_scratch(graph_size)
        case "ViTpretrained":
            model = ViT_pretrained(graph_size)
        case _:
            raise ValueError("Model not found")

    # Move model to device
    model.to(device)
    
    # DDP: wrap the model if using multi-GPU    
    if local_rank is not None:
        # Wrap the model in DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"- {model_name} Model loaded successfully on GPU {local_rank}.")
    else: 
        print(f"- {model_name} Model loaded successfully on {device}.")
        
    return model


# save time needed for training the completed graph size in a .yml file:
def save_partial_time(
    graph_size, graph_size_results_dir, exp_name_with_time, start_time, current_time
):
    """
    Save the time needed to train up to the graph size just completed in a .yml file.

    Args:
    graph_size (int): The size of the graphs that has been trained.
    graph_size_results_dir (str): The directory corresponding to the current graph size, where the results have been saved.
    exp_name_with_time (str): The name of the experiment with starting time.
    start_time (datetime): The time when the experiment started.
    current_time (datetime): The time at the end of the training of the current graph size.

    Returns:
    None
    """
    elapsed_time_file_path = os.path.join(
        graph_size_results_dir, f"{exp_name_with_time}_{graph_size}_elapsed_time.yml"
    )
    # Calculate the elapsed time since the start of the experiment
    elapsed_time = current_time - start_time
    # Convert the elapsed time to a string format
    elapsed_time_str = str(elapsed_time)
    # Saving the elapsed time in a .yml file
    with open(elapsed_time_file_path, "w") as file:
        yaml.dump(elapsed_time_str, file)


# save experiment configuration file in results folder:
def save_exp_config(config, results_dir, exp_name_with_time, start_time, end_time):
    """
    Save a copy of the configuration file in the experiment folder, adding a line where the time needed for the whole experiment is reported.

    Args:
    config (dict): The configuration settings.
    results_dir (str): The directory where the results will be saved.
    exp_name_with_time (str): The name of the experiment with starting time.
    start_time (datetime): The time when the experiment started.
    end_time (datetime): The time when the experiment ended.

    Returns:
    None
    """
    config_file_path = os.path.join(results_dir, f"{exp_name_with_time}_config.yml")
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Convert the elapsed time to a string format
    elapsed_time_str = str(elapsed_time)
    # adding elapsed time to the configuration dictionary and saving it to the file
    config["elapsed_time"] = elapsed_time_str
    with open(config_file_path, "w") as file:
        yaml.dump(config, file)


# Save test results to .csv files:
def save_test_results(
    fraction_correct_results, metrics_results, model_name, graph_size, results_dir
):
    """
    Save the test results to .csv files.

    Args:
        fraction_correct_results (dict): A dictionary containing the fraction correct results.
        metrics_results (dict): A dictionary containing the metrics results.
        model_name (str): The name of the model.
        graph_size (int): The size of the graph.
        results_dir (str): The directory where the results will be saved.

    Raises:
        ValueError: If the input arguments are not passed correctly.

    Returns:
        None
    """

    # START OF TESTS:

    # Check if fraction_correct_results is passed correctly
    if not isinstance(fraction_correct_results, dict):
        raise ValueError("fraction_correct_results should be a dictionary")

    # Check if metrics_results is passed correctly
    if not isinstance(metrics_results, dict):
        raise ValueError("metrics_results should be a dictionary")

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

    # Saving accuracy results in .csv file:
    # - defining file name and path:
    file_path = os.path.join(
        results_dir, f"{model_name}_N{graph_size}_fraction_correct.csv"
    )
    # - saving the dictionary as a .csv file:
    with open(file_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["clique size", "fraction correct"])  # Add column labels
        for key, value in fraction_correct_results.items():
            writer.writerow([key, value])

    # Saving metrics results in .csv file:
    # - defining file name and path:
    file_path = os.path.join(results_dir, f"{model_name}_N{graph_size}_metrics.csv")
    # - saving the dictionary as a .csv file:
    pd.DataFrame([metrics_results]).to_csv(file_path, index=False)

    print(f"- {model_name} Results saved successfully in {results_dir}.")


# Save trained model as .pth file:
def save_model(model, model_name, graph_size, results_dir):

    # START OF TESTS

    # Check if model is passed correctly (NOTE: make more stringent checks if needed)
    if not model:
        raise ValueError("Model not provided")

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

    # Saving .pth file in "results" folder (folder already exists, created in main.py)

    # - defining file name and path:
    file_path = os.path.join(results_dir, f"{model_name}_N{graph_size}_trained.pth")

    # - saving the trained model as a .pth file:
    torch.save(model.state_dict(), file_path)

    print(f"- {model_name} Trained Model saved successfully in {file_path}.")


def save_features(model, model_name, graph_size, p_correction, results_dir, device):
    """
    Save the features extracted from a trained model and save the corresponding image in results folder.

    Args:
        model (torch.nn.Module): The trained model used for feature extraction.
        model_name (str): The name of the model.
        graph_size (int): The size of the graph.
        p_correction (str): The p-correction type.
        results_dir (str): The directory where the features image will be saved.
        device (torch.device): The device where the model is stored.

    Returns:
        None
    """

    # # Uncomment this to visualize the names of the nodes in the graph (also print model to see the names of the nodes):
    # names = get_graph_node_names(model)
    # print(names)

    import src.graphs_generation as graphs_gen

    #  generate single graph with clique (70% of graph size, can be modified)
    graph = graphs_gen.generate_batch(
        1, graph_size, [int(0.7 * graph_size)], p_correction, True, p_clique=1
    )[0]
    graph = graph.to(device)

    # Defining layers to extract features from:
    # - CNN features:
    if "CNN" in model_name:
        # differentiating CNN versions:
        if model_name == "CNN_small_1":
            # creating features extractor with relevant node names:
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                    "model.4.0": "feat5",
                },
            )
        elif model_name == "CNN_small_2":
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                    "model.4.0": "feat5",
                    "model.5.0": "feat6",
                    "model.6.0": "feat7",
                },
            )
        elif model_name == "CNN_large_1":
            # creating features extractor with relevant node names:
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                },
            )
        elif model_name == "CNN_large_2":
            # creating features extractor with relevant node names:
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                    "model.4.0": "feat5",
                    "model.5.0": "feat6",
                },
            )
        elif model_name == "CNN_medium_1":
            # creating features extractor with relevant node names:
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                },
            )
        elif model_name == "CNN_medium_2":
            # creating features extractor with relevant node names:
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                    "model.4.0": "feat5",
                    "model.5.0": "feat6",
                },
            )
        else:
            raise ValueError("CNN Model not found. Model name might be incorrect.")

    # TODO: Visualization of features for ViTs?

    else:
        raise ValueError("Model not found. Model name might be incorrect.")

    # performing prediction on the single graph:
    out = model(graph)
    # Putting input as first element in the dictionary, before the features:
    out = {"input": graph, **out}

    # Visualizing the input image and the saved features contained in "out" dictionary:
    n_plots = len(out)
    # - Create a figure with appropriate number of subplots
    fig, axs = plt.subplots(1, n_plots, figsize=(20, 5))

    # - Iterate over the feature maps and add them in places
    for i, (name, feature_map) in enumerate(out.items()):
        # Select the first feature map
        feature_map = feature_map[0, 0, :, :].detach().cpu().numpy()

        # normalizing the feature map
        epsilon = 1e-10  # to avoid division by zero
        feature_map = (feature_map - feature_map.min()) / (
            feature_map.max() - feature_map.min() + epsilon
        )

        # Plot the feature map
        axs[i].imshow(feature_map, cmap="gray")
        axs[i].set_title(name)

    plt.tight_layout()

    # - Defining file path:
    file_path = os.path.join(results_dir, f"{model_name}_features_N{graph_size}.png")
    plt.savefig(file_path, dpi=300)

    # deleting temporary variables from memory to save space:
    del graph
    torch.cuda.empty_cache()

    print(f"- Features image saved successfully in {file_path}.")
