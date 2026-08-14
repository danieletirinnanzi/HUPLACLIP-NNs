import sys
import torch
import os
import yaml
import numpy as np
import pandas as pd
import morans_I
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from src.models import (
    MLP,
    CNN,
    ViT_scratch,
    ViT_pretrained,
)
import src.graphs_generation as graphs_gen
from src.input_transforms import magnify_input

# DESCRIPTION:
# This code recapitulates the Moran analysis performed on human data. For each model, the code:
# - loads the trained model
# - tests the loaded model in the K range manually specified (this is set so that all models are tested in the same K range, spanning from random to easy instances)
# - test graphs are both with and without clique. The script stores the model's soft and hard output (can be correct/incorrect) and the input's Moran's I
# - saves the results in a csv file
# This is designed to be compatible with execution on a normal laptop (not on a cluster using DistributedDataParallel)

# This will be usually run on regular laptop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading configuration:
config_path = os.path.join(os.getcwd(), "moran_testing_config.yml")
with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)
    print("Configuration file loaded successfully.")

# PARAMETERS:
# - Define the K range for testing (indicating min and max K values):
K_range_dict = {
    100: {"min": 1, "max": 50},
    150: {"min": 1, "max": 70},
    200: {"min": 1, "max": 90},
    300: {"min": 1, "max": 130},
    400: {"min": 1, "max": 150},
    480: {"min": 1, "max": 180},
    600: {"min": 1, "max": 230},
    800: {"min": 1, "max": 270},
    1200: {"min": 1, "max": 500},
}
                                                                                                                                                        
# HELPER FUNCTION: evaluate model on one K (runs the num_iterations loop, returns concatenated df for current batch)
def evaluate_K(K_value):
    all_graphs = []
    all_labels = []
    all_morans_I_results = []
    all_soft_outputs = []
    num_iterations = config["num_iterations"]
    batch_size = config["batch_size"]
    iter_size = batch_size // num_iterations
    assert iter_size == 10, f"Iter size should be 10, it is: {iter_size}"

    for iter_idx in range(num_iterations):
        clique_size_array = graphs_gen.generate_batch_clique_sizes(np.array([K_value]), iter_size)
        graphs, labels = graphs_gen.generate_batch(
            iter_size,
            graph_size,
            clique_size_array,
            config["p_correction_type"],
            input_magnification=False,
            p_clique=0.5
        )
        labels = np.array(labels)
        adj_matrices = graphs[:, 0].cpu().numpy()
        assert adj_matrices.shape == (iter_size, graph_size, graph_size)
        morans_I_results = np.array([morans_I.morans_I_numba(adj, config["max_radius"]) for adj in adj_matrices])
        del adj_matrices
        if model_name == "CNN":
            # magnify the graphs to 2400x2400 after calculating Moran's I
            magnified = torch.zeros((iter_size, 1, 2400, 2400), dtype=graphs.dtype)
            for jj in range(iter_size):
                magnified[jj, 0] = magnify_input(graphs[jj].squeeze(0)) # default output is 2400x2400
            graphs = magnified
        with torch.no_grad():
            graphs_tensor = graphs.clone().detach().to(device) # added to avoid userwarning (copy from tensor)
            soft_outputs = model(graphs_tensor).squeeze().cpu().numpy()

        all_graphs.append(graphs)
        all_labels.append(labels)
        all_morans_I_results.append(morans_I_results)
        all_soft_outputs.append(soft_outputs)

        # free intermediate memory
        del graphs, labels, morans_I_results, graphs_tensor, soft_outputs
        torch.cuda.empty_cache()

    graphs_cat = np.concatenate(all_graphs, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    morans_I_results_cat = np.concatenate(all_morans_I_results, axis=0)
    soft_outputs_cat = np.concatenate(all_soft_outputs, axis=0)
    batch_size_actual = graphs_cat.shape[0]
    assert batch_size_actual == config["batch_size"]
    morans_I_cols = [f"morans_I_lambda_{r}" for r in range(1, config["max_radius"] + 1)]
    morans_I_df = pd.DataFrame(morans_I_results_cat[:, 1:], columns=morans_I_cols)

    hard_outputs = (soft_outputs_cat > 0.5).astype(int)
    labels_int = labels_cat.astype(int)
    correct = (hard_outputs == labels_int)

    batch_df = pd.DataFrame({
        "model": model_name,
        "N": graph_size,
        "K": K_value,
        "soft_output": soft_outputs_cat,
        "hard_output": hard_outputs,    # 1 if model predicts clique, 0 otherwise
        "label": labels_cat,
        "correct": correct,
    })
    batch_df = pd.concat([batch_df, morans_I_df], axis=1)
    return batch_df


# TEST LOOP
for graph_size in config["graph_sizes"]:
    
    print("| Graph size = ", graph_size)
    
    if graph_size == 1000:
        continue

    for model_specs in config["models"]:     
                   
        model_name = model_specs["model_name"]
        if model_name == "Humans":
            continue

        # - loading requested model
        match model_name:
            case "MLP":
                model = MLP(graph_size, model_specs["architecture"])
            case "CNN":
                model = CNN(graph_size, model_specs["architecture"])
            case "ViTscratch":
                model = ViT_scratch(graph_size)
            case "ViTpretrained":
                model = ViT_pretrained(graph_size)
            case _:
                raise ValueError("Model not found")
        # Load weights from saved_models
        weights_path = os.path.join(os.getcwd(), "saved_models",f"N{graph_size}", f"{model_name}_N{graph_size}_trained.pth")
        state_dict = torch.load(weights_path, map_location=device)
        if os.path.exists(weights_path):
            # Remove 'module.' prefix if present (present if model was trained with DDP)
            if any(k.startswith("module.") for k in state_dict.keys()):
                new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                state_dict = new_state_dict
            model.load_state_dict(state_dict)
            print(f"|| Loaded weights from {weights_path}")
        else:
            raise FileNotFoundError(f"|| ERROR: Weights file not found: {weights_path}")
        model.to(device)
        model.eval()        
        print(f"|| {model_name} Model loaded successfully.")
        
        # Start testing in K range defined in K_range_dict (100 values between min and max K, inclusive)
        K_values = np.rint(
            np.linspace(
                K_range_dict[graph_size]["min"],
                K_range_dict[graph_size]["max"],
                100,
                endpoint=True,
            )
        ).astype(int)
        model_results = []
        for K in K_values:
            print("||| Evaluating K: ", K)
            batch_df = evaluate_K(K)
            model_results.append(batch_df)

        # After both phases concatenate and save
        model_results_df = pd.concat(model_results, ignore_index=True)
        save_path = os.path.join(os.getcwd(),"nns_moran_results", "fixed-k-range", f"N{graph_size}",f"{model_name}_N{graph_size}_moran_results_fixed-k-range.csv")
        model_results_df.to_csv(save_path, index=False)
        print(f"|| Results for model {model_name} saved to {save_path}")