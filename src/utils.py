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
        case "ResNet50scratch":
            model = ResNet50_scratch()
        case "ResNet50pretrained":
            model = ResNet50_pretrained()
        case "GoogLeNetscratch":
            model = GoogLeNet_scratch()
        case "GoogLeNetpretrained":
            model = GoogLeNet_pretrained()
        case "ViTscratch":
            model = ViT_scratch(graph_size, model_specs["architecture"])
        case "ViTpretrained":
            model = ViT_pretrained(graph_size, model_specs["architecture"])

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
    Save the features extracted from a trained model and save the corresponding image in results folder.

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

    # # Uncomment this to visualize the names of the nodes in the graph (also print model to see the names of the nodes):
    # names = get_graph_node_names(trained_model)
    # print(names)

    from src.graphs_generation import generate_graphs

    #  generate graph with clique (70% of graph size, can be modified)
    graph = generate_graphs(
        1, graph_size, int(0.7 * graph_size), p_correction, True, p_clique=1
    )[0]

    # Defining layers to extract features from:
    # - CNN features:
    if "CNN" in model_name:
        # differentiating 3 CNN versions:
        if model_name == "CNN_small":
            # creating features extractor with relevant node names:
            trained_model = create_feature_extractor(
                trained_model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                    "model.4.0": "feat5",
                    "model.5.0": "feat6",
                    "model.6.0": "feat7",
                    "model.7.0": "feat8",
                },
            )
        elif model_name == "CNN_medium":
            # creating features extractor with relevant node names:
            trained_model = create_feature_extractor(
                trained_model,
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
        elif model_name == "CNN_large":
            # creating features extractor with relevant node names:
            trained_model = create_feature_extractor(
                trained_model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                    "model.3.0": "feat4",
                },
            )
        else:
            raise ValueError("CNN Model not found. Model name might be incorrect.")

        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    # - VGG features:
    elif "VGG16" in model_name:
        # creating features extractor with relevant node names:
        trained_model = create_feature_extractor(
            trained_model,
            {
                "model.features.2": "conv2",
                "model.features.7": "conv4",
                "model.features.12": "conv6",
                "model.features.17": "conv8",
                "model.features.21": "conv10",
                "model.features.26": "conv12",
                "model.features.28": "conv15",
            },
        )
        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    # - ResNet features:
    elif "ResNet50" in model_name:
        # creating features extractor with relevant node names:
        trained_model = create_feature_extractor(
            trained_model,
            {
                "model.layer1.0.conv1": "layer1.0_conv1",
                "model.layer1.2.conv3": "layer1.2_conv3",
                "model.layer2.0.conv1": "layer2.0_conv1",
                "model.layer2.3.conv3": "layer2.3_conv3",
                "model.layer3.0.conv1": "layer3.0_conv1",
                "model.layer3.5.conv3": "layer3.5_conv3",
                "model.layer4.2.conv3": "layer4.2_conv3",
            },
        )
        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    elif "GoogLeNet" in model_name:
        # creating features extractor with relevant node names:
        trained_model = create_feature_extractor(
            trained_model,
            {
                "model.conv1.conv": "conv1",
                "model.conv2.conv": "conv2",
                "model.conv3.conv": "conv3",
                "model.inception3a.branch2.0.conv": "inception3a_branch2",
                "model.inception4a.branch2.0.conv": "inception4a_branch2",
                "model.inception4c.branch2.0.conv": "inception4c_branch2",
                "model.inception4d.branch2.0.conv": "inception4d_branch2",
                "model.inception5a.branch2.0.conv": "inception5a_branch2",
            },
        )
        # performing prediction on the single graph:
        if device == "cuda":
            out = trained_model(graph.cuda())
        else:
            out = trained_model(graph)

    else:
        raise ValueError("Model not found. Model name might be incorrect.")

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

    print(f"- Features image saved successfully in {file_path}.")
