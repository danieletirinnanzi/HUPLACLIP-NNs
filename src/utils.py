import yaml
import torch
import csv
import os
import matplotlib.pyplot as plt

# custom imports
from .models import (
    MLP,
    CNN,
    VGG16_scratch,
    VGG16_pretrained,
    ResNet50_scratch,
    ResNet50_pretrained,
    GoogLeNet_scratch,
    GoogLeNet_pretrained,
    ViT_scratch,
    ViT_pretrained,
)
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

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
def load_model(model_specs, graph_size):

    model_name = model_specs["model_name"]

    # - building requested model
    match model_name:
        case "MLP":
            model = MLP(graph_size, model_specs["architecture"])
        case "CNN_small" | "CNN_medium" | "CNN_large":
            model = CNN(graph_size, model_specs["architecture"])
        case "VGG16scratch":
            model = VGG16_scratch()
        case "VGG16pretrained":
            model = VGG16_pretrained()
        case "RESNET50scratch":
            model = ResNet50_scratch()
        case "RESNET50pretrained":
            model = ResNet50_pretrained()
        case "GoogLeNetscratch":
            model = GoogLeNet_scratch()
        case "GoogLeNetpretrained":
            model = GoogLeNet_pretrained()
        case "VITscratch":
            model = ViT_scratch(graph_size)
        case "VITpretrained":
            model = ViT_pretrained(graph_size)

        # ADDITIONAL MODELS CAN BE ADDED HERE

        case _:
            raise ValueError("Model not found")

    # - sending model to device:
    model.to(device)
    print("- Model loaded successfully.")
    return model


# save experiment configuration file in results folder:
def save_exp_config(config, results_dir, exp_name_with_time, start_time, end_time):
    """
    Save a copy of the configuration file in the experiment folder, adding a line where the time needed for the whole experiment is reported.

    Args:
    config (dict): The configuration settings.
    results_dir (str): The directory where the results will be saved.
    exp_name_with_time (str): The name of the experiment with time.
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


def save_features(trained_model, model_name, graph_size, p_correction, results_dir):
    """
    Save the features extracted from a trained model and save the corresponding image.

    Args:
        trained_model (torch.nn.Module): The trained model used for feature extraction.
        model_name (str): The name of the model.
        graph_size (int): The size of the graph.
        p_correction (str): The p-correction type.
        results_dir (str): The directory where the features image will be saved.

    Returns:
        None
    """

    # NOTE: add visualization of features also when graph has no clique?

    # # Uncomment this to visualize the names of the nodes in the graph:
    # names = get_graph_node_names(trained_model)
    # print(names)

    from src.graphs_generation import generate_graphs

    # Defining layers to extract features from:
    # - CNN features:
    if model_name == "CNN":
        # creating features extractor with relevant node names:
        trained_model = create_feature_extractor(
            trained_model,
            {
                "0.0": "feat1",
                "1.0": "feat2",
                "2.0": "feat3",
                "3.0": "feat4",
                "4.0": "feat5",
                "5.0": "feat6",
                "6.0": "feat7",
            },
        )
        #  generate graph with clique (70% of graph size, can be modified)
        graph = generate_graphs(
            1, graph_size, int(0.7 * graph_size), p_correction, False, True, p_clique=1
        )[0]
        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    # - VGG features:
    if model_name == "VGG16":
        # creating features extractor with relevant node names:
        trained_model = create_feature_extractor(
            trained_model,
            {
                "features.2": "feat2",
                "features.7": "feat7",
                "features.12": "feat12",
                "features.17": "feat17",
                "features.21": "feat21",
                "features.26": "feat26",
                "features.28": "feat28",
            },
        )
        #  generate graph with clique (70% of graph size, can be modified)
        graph = generate_graphs(
            1, graph_size, int(0.7 * graph_size), p_correction, True, False, p_clique=1
        )[0]
        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    # - ResNet features:
    if model_name == "RESNET50":
        # creating features extractor with relevant node names:
        trained_model = create_feature_extractor(
            trained_model,
            {
                "layer1.0.conv1": "feature1.0",
                "layer1.2.conv3": "feature1.2",
                "layer2.0.conv1": "feature2.0",
                "layer2.3.conv3": "feature2.3",
                "layer3.0.conv1": "feature3.0",
                "layer3.5.conv3": "feature3.5",
                "layer4.2.conv3": "feature4.2",
            },
        )
        #  generate graph with clique (70% of graph size, can be modified)
        graph = generate_graphs(
            1, graph_size, int(0.7 * graph_size), p_correction, True, False, p_clique=1
        )[0]
        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    # Putting input as first element in the dictionary, before the features:
    out = {"input": graph, **out}

    # Visualizing the input image and the 4 features:
    # - Create a figure with 8 subplots
    fig, axs = plt.subplots(1, 8, figsize=(20, 5))

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

    print(f"- Features image saved successfully in {file_path}.")
