import yaml
import torch
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

# custom imports
from .models import (
    MLP,
    CNN,
    # CNN_rudy,
    # VGG16_scratch,
    # VGG16_pretrained,
    # ResNet50_scratch,
    # ResNet50_pretrained,
    # GoogLeNet_scratch,
    # GoogLeNet_pretrained,
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
def load_model(model_specs, graph_size, device):

    model_name = model_specs["model_name"]

    # - building requested model
    match model_name:
        case "MLP":
            model = MLP(graph_size, model_specs["architecture"])
        case (
            "CNN_small_1"
            | "CNN_small_2"
            | "CNN_medium_1"
            | "CNN_medium_2"
            | "CNN_large_1"
            | "CNN_large_2"
        ):
            model = CNN(graph_size, model_specs["architecture"])
        # case "CNN_rudy":
        #     model = CNN_rudy(graph_size, model_specs["architecture"])
        # case "VGG16scratch":
        #     model = VGG16_scratch()
        # case "VGG16pretrained":
        #     model = VGG16_pretrained()
        # case "ResNet50scratch":
        #     model = ResNet50_scratch()
        # case "ResNet50pretrained":
        #     model = ResNet50_pretrained()
        # case "GoogLeNetscratch":
        #     model = GoogLeNet_scratch()
        # case "GoogLeNetpretrained":
        #     model = GoogLeNet_pretrained()
        case "ViTscratch":
            model = ViT_scratch(graph_size)
        case "ViTpretrained":
            model = ViT_pretrained(graph_size)

        # ADDITIONAL MODELS CAN BE ADDED HERE

        case _:
            raise ValueError("Model not found")

    # - sending model to device:
    model.to(device)
    print(f"- {model_name} Model loaded successfully.")
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

    # TODO: adapt feature extraction to new models (names defined in "models.py")

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
        if "1" in model_name:
            # creating features extractor with relevant node names:
            model = create_feature_extractor(
                model,
                {
                    "model.0.0": "feat1",
                    "model.1.0": "feat2",
                    "model.2.0": "feat3",
                },
            )
        elif "2" in model_name:
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
        # elif "rudy" in model_name:
        #     # creating features extractor with relevant node names:
        #     model = create_feature_extractor(
        #         model,
        #         {
        #             "model.1.0": "feat1",
        #             "model.4.conv1": "feat_res_1_1",
        #             "model.4.conv2": "feat_res_1_2",
        #             "model.5.0": "feat2",
        #             "model.8.conv1": "feat_res_2_1",
        #             "model.8.conv2": "feat_res_2_2",
        #             "model.9.0": "feat3",
        #         },
        #     )
        else:
            raise ValueError("CNN Model not found. Model name might be incorrect.")

    # # - VGG features:
    # elif "VGG16" in model_name:
    #     # creating features extractor with relevant node names:
    #     trained_model = create_feature_extractor(
    #         trained_model,
    #         {
    #             "model.features.2": "conv2",
    #             "model.features.7": "conv4",
    #             "model.features.12": "conv6",
    #             "model.features.17": "conv8",
    #             "model.features.21": "conv10",
    #             "model.features.26": "conv12",
    #             "model.features.28": "conv15",
    #         },
    #     )

    # # - ResNet features:
    # elif "ResNet50" in model_name:
    #     # creating features extractor with relevant node names:
    #     trained_model = create_feature_extractor(
    #         trained_model,
    #         {
    #             "model.layer1.0.conv1": "layer1.0_conv1",
    #             "model.layer1.2.conv3": "layer1.2_conv3",
    #             "model.layer2.0.conv1": "layer2.0_conv1",
    #             "model.layer2.3.conv3": "layer2.3_conv3",
    #             "model.layer3.0.conv1": "layer3.0_conv1",
    #             "model.layer3.5.conv3": "layer3.5_conv3",
    #             "model.layer4.2.conv3": "layer4.2_conv3",
    #         },
    #     )

    # elif "GoogLeNet" in model_name:
    #     # creating features extractor with relevant node names:
    #     trained_model = create_feature_extractor(
    #         trained_model,
    #         {
    #             "model.conv1.conv": "conv1",
    #             "model.conv2.conv": "conv2",
    #             "model.conv3.conv": "conv3",
    #             "model.inception3a.branch2.0.conv": "inception3a_branch2",
    #             "model.inception4a.branch2.0.conv": "inception4a_branch2",
    #             "model.inception4c.branch2.0.conv": "inception4c_branch2",
    #             "model.inception4d.branch2.0.conv": "inception4d_branch2",
    #             "model.inception5a.branch2.0.conv": "inception5a_branch2",
    #         },
    #     )

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

    print(f"- Features image saved successfully in {file_path}.")
