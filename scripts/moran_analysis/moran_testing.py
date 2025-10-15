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

# This will be usually run on regular laptop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading configuration:
config_path = os.path.join(os.getcwd(), "moran_testing_config.yml")
with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)
    print("Configuration file loaded successfully.")

# defining grid to be tested:
K_grid = "finer" # or "finer"

for graph_size in config["graph_sizes"]:
    
    print("| Graph size = ", graph_size)
    
    if graph_size == 1000:
        continue
    # for each N value, defining grid of K values where testing will occur:
    # - for each N value, range [ minK0 - 10; maxk0 + 10]
    if K_grid == "common":
        # Find empirical K0s for this graph_size
        empirical_K0_entry = next(e for e in config["empirical_K0s"] if e["N"] == graph_size)
        K0_values = list(empirical_K0_entry["values"].values())
        print("| K0 values are: ", K0_values)
        min_K0 = min(K0_values)
        max_K0 = max(K0_values)
        # in the cases where CNN failed, excluding maximum K0 value:
        if graph_size in [200, 300, 400]:
            max_K0 = sorted(K0_values)[-2]    
        K_range = np.linspace(round(min_K0) - 10, round(max_K0) + 10, 8, dtype=int)
        print(f"| K range: {K_range[0]} to {K_range[-1]}")            
    for model_specs in config["models"]:        
        model_name = model_specs["model_name"]
        if model_name == "Humans":
            continue
        model_results = []
        # - building requested model
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

        if K_grid == "finer":
            # Find empirical K0 of this model
            empirical_K0_entry = next(e for e in config["empirical_K0s"] if e["N"] == graph_size)
            K0_value = empirical_K0_entry["values"][model_name]
            print(f"| K0 value for {model_name} is: {K0_value}")
            # in the cases where CNN failed, excluding maximum K0 value:
            if (graph_size in [200, 300, 400] and model_name == "CNN"):
                continue 
            K_range = np.arange(round(K0_value) - 3, round(K0_value) + 4, dtype=int)
            print(f"| K range: {K_range}")                
        
        for K_value in K_range:
            
            print("||| K value is: ", K_value)
            
            # Split the batch into num_iterations
            # NOTE: needed for "CNN" for memory constraints, but applied to all models for consistency
            num_iterations = config["num_iterations"]
            batch_size = config["batch_size"]
            iter_size = batch_size // num_iterations
            assert iter_size == 10, f"Iter size should be 10, it is: {iter_size}"
            all_graphs = []
            all_labels = []
            all_morans_I_results = []
            all_soft_outputs = []

            for iter_idx in range(num_iterations):
                # Generate a subset of the batch
                clique_size_array = graphs_gen.generate_batch_clique_sizes(
                    np.array([K_value]),
                    iter_size,
                )
                graphs, labels = graphs_gen.generate_batch(
                    iter_size,
                    graph_size,
                    clique_size_array,
                    config["p_correction_type"],
                    input_magnification=True if model_name == "CNN" else False,
                    p_clique=1
                )
                labels = np.array(labels)
                adj_matrices = graphs[:, 0].cpu().numpy()
                if model_name == "CNN":
                    assert adj_matrices.shape == (iter_size, 2400, 2400), f"adj_matrices shape {adj_matrices.shape} does not match expected ({iter_size}, {graph_size}, {graph_size})"
                else:
                    assert adj_matrices.shape == (iter_size, graph_size, graph_size), f"adj_matrices shape {adj_matrices.shape} does not match expected ({iter_size}, {graph_size}, {graph_size})"                    
                # Moran's I
                morans_I_results = np.array([morans_I.morans_I_numba(adj, config["max_radius"]) for adj in adj_matrices])
                # Model predictions
                with torch.no_grad():
                    graphs_tensor = torch.tensor(graphs, dtype=torch.float32, device=device)
                    soft_outputs = model(graphs_tensor).squeeze().cpu().numpy()
                # Collect all
                all_graphs.append(graphs)
                all_labels.append(labels)
                all_morans_I_results.append(morans_I_results)
                all_soft_outputs.append(soft_outputs)
                # Delete unused variables from memory
                del graphs, labels, adj_matrices, morans_I_results, graphs_tensor, soft_outputs
                torch.cuda.empty_cache()

            # Concatenate all iterations
            graphs = np.concatenate(all_graphs, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            morans_I_results = np.concatenate(all_morans_I_results, axis=0)
            soft_outputs = np.concatenate(all_soft_outputs, axis=0)
            batch_size = graphs.shape[0]
            assert batch_size == config["batch_size"], f"Number of generated graphs ({batch_size}) does not match config batch size ({config['batch_size']})"
            # Prepare DataFrame columns for Moran's I
            morans_I_cols = [f"morans_I_lambda_{r}" for r in range(1, config["max_radius"] + 1)]
            morans_I_df = pd.DataFrame(morans_I_results[:, 1:], columns=morans_I_cols)  # skip radius 0

            # Hard predictions (threshold 0.5)
            hard_outputs = (soft_outputs > 0.5).astype(int)
            labels_int = labels.astype(int)
            correct = (hard_outputs == labels_int)

            batch_df = pd.DataFrame({
                "model": model_name,
                "N": graph_size,
                "K": K_value,
                "soft_output": soft_outputs,
                "hard_output": hard_outputs,
                "label": labels,
                "correct": correct,
            })
            batch_df = pd.concat([batch_df, morans_I_df], axis=1)
            
            # Sanity check print
            print(f"    - Batch DataFrame shape: {batch_df.shape}, correct: {correct.sum()}/{batch_size}")

            model_results.append(batch_df)

        # After all K for this model, concatenate and save
        model_results_df = pd.concat(model_results, ignore_index=True)
        save_path = os.path.join(os.getcwd(),"results" if K_grid == "common" else "results_finer_grid",f"N{graph_size}",f"{model_name}_N{graph_size}_moran_results.csv")
        model_results_df.to_csv(save_path, index=False)
        print(f"|| Results for model {model_name} saved to {save_path}")