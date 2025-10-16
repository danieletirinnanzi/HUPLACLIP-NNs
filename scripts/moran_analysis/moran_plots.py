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

fig, axes = plt.subplots(1, len(config["graph_sizes"]), figsize=(4 * len(config["graph_sizes"]), 4))

for i, graph_size in enumerate(config["graph_sizes"]):
    print("N = ", graph_size)
    
    ax = axes[i]
    for model_specs in config["models"]:
        
        model_name = model_specs["model_name"]
        print("Model name: ", model_name)

        # ANNs were tested at N=1200, Humans at N=1000
        if graph_size == 1000 and model_name != "Humans":
            continue
        elif graph_size == 1200 and model_name == "Humans":
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
        ax.axvline(K0_value, color=color, linestyle=':', linewidth=1.5)
        # Draw a horizontal dashed gray line at y=0
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.1)

        # ANNs:
        if model_name != "Humans":
            # Load dataframe for current model, N and clique condition
            df_path = os.path.join(os.getcwd(), "results_finer_grid", f"N{graph_size}", f"{model_name}_N{graph_size}_moran_results.csv")
            df = pd.read_csv(df_path)

            # Isolate trials around empirical K0 (three above, three below)            
            K_array = df['K'].unique()
            print("K values: ", K_array)
            distances = abs(K_array - K0_value)
            min_idx = np.argmin(distances)
            closest_K = K_array[min_idx]
            print(f"Closest K to empirical K0 for N={graph_size}: {closest_K} (empirical K0={K0_value})")
            # Selecting K values to show on the plot (2 below closest_K, 2 above closest_K)
            K_array_plot = K_array[(min_idx-2):(min_idx+3)]
            print("K values to plot: ", K_array_plot)                 

            for K_value in K_array_plot:
                df_sub_correct = df[(df['K'] == K_value) & (df['correct'] == True)]
                df_sub_incorrect =  df[(df['K'] == K_value) & (df['correct'] == False)]
                # only plot if there is data available
                if not (df_sub_correct.empty or df_sub_incorrect.empty):
                    means_difference = df_sub_correct[f'morans_I_lambda_{lambda_value}'].mean() - df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].mean()
                    standard_error_difference = np.sqrt( ( np.var(df_sub_correct[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_correct[f'morans_I_lambda_{lambda_value}'].count() ) + ( (np.var(df_sub_incorrect[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].count()) ) )
                    ax.errorbar(
                        K_value, means_difference, yerr=standard_error_difference,
                        fmt=models_legend[model_name]["marker"],
                        color=color,
                        alpha=1,
                        markersize=4,
                        label=f"{model_name}" if (K_value == K_array_plot[0]) else "" # only add label to first K value
                    )
                else:
                    raise ValueError("Missing correct or incorrec trials. Check mistakes in data.")
        # Humans:
        else:
            # Load dataframe for N and clique condition
            df_path = os.path.join(os.getcwd(), "human_strategy_data", f"humans_visual_strategy_2025-07_exp1-30subjects_N{graph_size}_CLIQUE.csv")
            df = pd.read_csv(df_path)
            # Isolate trials around empirical K0 (one above, one below)            
            K_array = df['K'].unique()
            print("K values: ", K_array)
            distances = abs(K_array - K0_value)
            min_idx = np.argmin(distances)
            closest_K = K_array[min_idx]
            print(f"Closest K to empirical K0 for N={graph_size}: {closest_K} (empirical K0={K0_value})")

            # Selecting K values to show on the plot (2 below closest_K, 2 above closest_K)
            K_array_plot = K_array[(min_idx-2):(min_idx+3)]
            print("K values to plot: ", K_array_plot)            

            for K_value in K_array_plot:
                df_sub_correct = df[(df['K'] == K_value) & (df['correct'] == True)]
                df_sub_incorrect =  df[(df['K'] == K_value) & (df['correct'] == False)]
                # only plot if there is data available
                if not (df_sub_correct.empty or df_sub_incorrect.empty):
                    means_difference = df_sub_correct[f'morans_I_lambda_{lambda_value}'].mean() - df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].mean()
                    # NOTE: standard error of difference might be inaccurate for Humans (independent stimuli, but for a given K, data also coming from same subject. Alternative can be to compute for single subjects and plot distribution, but only 12 trials per subject for each K value)
                    standard_error_difference = np.sqrt( ( np.var(df_sub_correct[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_correct[f'morans_I_lambda_{lambda_value}'].count() ) + ( (np.var(df_sub_incorrect[f'morans_I_lambda_{lambda_value}'], ddof=1) / df_sub_incorrect[f'morans_I_lambda_{lambda_value}'].count()) ) )
                    ax.errorbar(
                        K_value, means_difference, yerr=standard_error_difference,
                        fmt=models_legend[model_name]["marker"],
                        color=color,
                        alpha=1,
                        markersize=8,
                        label=f"{model_name}"  if (K_value == K_array_plot[0]) else "" # only add label to first K value
                    )
                else:
                    raise ValueError("Missing correct or incorrec trials. Check mistakes in data.")  
                    
    ax.set_xlabel('Clique size (K)', fontsize=12)
    ax.set_ylabel("Moran's I difference (correct - incorrect)", fontsize=12)
    ax.set_title(f"N = {graph_size}", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    if i == 0: 
        ax.legend(fontsize=6, loc='best')
    
    print("---------")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(f"Moran's I difference (lambda = {lambda_value}) between correct and incorrect trials with clique", fontsize=18)    
base_path = os.path.join(os.getcwd(), f'plots','NNs_humans-visual-strategy-moransI_finergrid')
# plt.savefig(base_path + '.svg', dpi=300, bbox_inches="tight")
plt.savefig(base_path + '.png', dpi=300, bbox_inches="tight")
# plt.show()
print("|Completed generating visual strategy graphs for correct/incorrect responses.")