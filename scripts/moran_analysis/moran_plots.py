import os
import yaml
import pandas as pd
import seaborn as sns
import colorcet as cc
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Moran's I for K values around K0s of all models, only for graphs with clique (correct/incorrect distribution)
# PLOT SETTINGS:
models_legend = {
    "MLP": {
        "color": sns.color_palette(cc.glasbey, 6)[0],
        "marker": "o",
    },
    "CNN": {
        "color": sns.color_palette(cc.glasbey, 6)[4],
        "marker": "o",
    },
    "ViTscratch": {
        "color": sns.color_palette(cc.glasbey, 6)[1],
        "marker": "o",
    },
    "ViTpretrained": {
        "color": sns.color_palette(cc.glasbey, 6)[2],
        "marker": "o",
    },
    "Humans": {
        "color": sns.color_palette("flare", 9),
        "marker": "*"    
        }
}
lambda_value = 2  # CHANGE THIS TO TEST DIFFERENT LAMBDAS

# loading configuration:
config_path = os.path.join(os.getcwd(), "moran_testing_config.yml")
with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)
    print("Configuration file loaded successfully.")
# Define the range for fraction correct
fraction_correct_range = {
    "min": 0.50,
    "max": 0.90
}

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
# Draw a horizontal dashed gray line at y=0
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.1)

for i, graph_size in enumerate(config["graph_sizes"]):
    print("N = ", graph_size)
    
    for model_specs in config["models"]:
        
        model_name = model_specs["model_name"]
        print("Model name: ", model_name)

        # ANNs were tested at N=1200, Humans at N=1000
        if graph_size == 1000 and model_name != "Humans":
            continue
        elif graph_size == 1200 and model_name == "Humans":
            continue
        # CNN failed at some N values:
        if model_name == "CNN" and graph_size in [200, 300, 400]:
            continue
        # Plot specs:
        if model_name == "Humans":
            color = models_legend[model_name]["color"][i]
        else:
            color = models_legend[model_name]["color"]
        
        # Empirical K0:
        empirical_K0_entry = next(e for e in config["empirical_K0s"] if e["N"] == graph_size)
        # add empirical K0 as vertical dashed line:
        K0_value = empirical_K0_entry["values"][model_name]        

        # ANNs:
        if model_name != "Humans":
            # Load dataframe for current model, N and clique condition
            df_path = os.path.join(os.getcwd(), "results_finer_grid", f"N{graph_size}", f"{model_name}_N{graph_size}_moran_results.csv")
            df = pd.read_csv(df_path)

            # Define K values to include in plot (where fraction correct is within range)         
            K_array = df['K'].unique()
            print("K values: ", K_array)
            K_array_within_range = []
            for K_value in K_array:
                df_sub = df[df['K'] == K_value]
                fraction_correct = df_sub['correct'].mean()
                print(f"Safety check: K={K_value}, fraction_correct={fraction_correct}")
                if fraction_correct_range["min"] <= fraction_correct <= fraction_correct_range["max"]:
                    K_array_within_range.append(K_value)
            K_array = np.array(K_array_within_range)
            print("K values within fraction correct range: ", K_array)
            # Select correct/incorrect trials within range
            df_sub_correct = df[(df['K'].isin(K_array)) & (df['correct'] == True)]
            df_sub_incorrect =  df[(df['K'].isin(K_array)) & (df['correct'] == False)]
            # only plot if there is data available
            if not (df_sub_correct.empty or df_sub_incorrect.empty):
                means_difference = df_sub_correct[f'morans_I_lambda_{lambda_value}'].mean() - df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].mean()
                standard_error_difference = np.sqrt( ( np.var(df_sub_correct[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_correct[f'morans_I_lambda_{lambda_value}'].count() ) + ( (np.var(df_sub_incorrect[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].count()) ) )
                # Normalizing difference by overall mean Moran's I 
                mean_morans_I_all = pd.concat([df_sub_correct, df_sub_incorrect])[f'morans_I_lambda_{lambda_value}'].mean()
                means_difference_normalized = means_difference / mean_morans_I_all if mean_morans_I_all != 0 else 0         
                standard_error_difference_normalized = standard_error_difference / mean_morans_I_all
                ax.errorbar(
                    graph_size + random.random(), means_difference_normalized, yerr=standard_error_difference_normalized,
                    fmt=models_legend[model_name]["marker"],
                    color=color,
                    alpha=1,
                    markersize=3,
                    label=f"{model_name}" if (i == 0) else "" # only add label to first K value
                )
            else:
                continue
                # raise ValueError("No data available for plotting with current selection criterion. Check for mistakes")

        # Humans:
        else:
            # Load dataframe for N and clique condition
            df_path = os.path.join(os.getcwd(), "human_strategy_data", f"humans_visual_strategy_2025-07_exp1-30subjects_N{graph_size}_CLIQUE.csv")
            df = pd.read_csv(df_path)
            
            # Define K values to include in plot (where fraction correct is within range)         
            K_array = df['K'].unique()
            print("K values: ", K_array)
            K_array_within_range = []
            for K_value in K_array:
                df_sub = df[df['K'] == K_value]
                fraction_correct = df_sub['correct'].mean()
                print(f"Safety check: K={K_value}, fraction_correct={fraction_correct}")
                if fraction_correct_range["min"] <= fraction_correct <= fraction_correct_range["max"]:
                    K_array_within_range.append(K_value)
            K_array = np.array(K_array_within_range)
            print("K values within fraction correct range: ", K_array)
            # Select correct/incorrect trials within range
            df_sub_correct = df[(df['K'].isin(K_array)) & (df['correct'] == True)]
            df_sub_incorrect =  df[(df['K'].isin(K_array)) & (df['correct'] == False)]       
            # only plot if there is data available
            if not (df_sub_correct.empty or df_sub_incorrect.empty):
                means_difference = df_sub_correct[f'morans_I_lambda_{lambda_value}'].mean() - df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].mean()
                standard_error_difference = np.sqrt( ( np.var(df_sub_correct[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_correct[f'morans_I_lambda_{lambda_value}'].count() ) + ( (np.var(df_sub_incorrect[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].count()) ) )
                # Normalizing by overall mean Moran's I 
                mean_morans_I_all = pd.concat([df_sub_correct, df_sub_incorrect])[f'morans_I_lambda_{lambda_value}'].mean()
                means_difference_normalized = means_difference / mean_morans_I_all
                standard_error_difference_normalized = standard_error_difference / mean_morans_I_all
                # NOTE: try to represent with violin plot for consistency?
                ax.errorbar(
                    graph_size, means_difference_normalized, yerr=standard_error_difference_normalized,
                    fmt=models_legend[model_name]["marker"],
                    color=color,
                    alpha=1,
                    markersize=10,
                    label=f"{model_name}" if (i == 0) else "" # only add label to first K value
                )
            else:
                continue
                # raise ValueError("No data available for plotting with current selection criterion. Check for mistakes")
    
    ax.set_xlim(0, 1250)
    ax.set_xlabel('Number of nodes (N)', fontsize=12)
    ax.set_ylabel("Normalized Moran's I difference (correct - incorrect)", fontsize=7)
    ax.tick_params(axis='x', rotation=45)
    
    if i == 0: 
        ax.legend(fontsize=6, loc='best')
    
    print("---------")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(f"Normalized Moran's I difference (lambda = {lambda_value}) between correct and incorrect trials with clique", fontsize=12)    
base_path = os.path.join(os.getcwd(), f'plots',f'NNs_humans-visual-strategy-moransI_finergrid_withCNN_range{fraction_correct_range["min"]}-{fraction_correct_range["max"]}')
# plt.savefig(base_path + '.svg', dpi=300, bbox_inches="tight")
plt.savefig(base_path + '.png', dpi=300, bbox_inches="tight")
# plt.show()
print("|Completed generating visual strategy graphs for correct/incorrect responses.")