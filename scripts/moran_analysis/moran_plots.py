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
        "marker": "s",
    },
    "ViTscratch": {
        "color": sns.color_palette(cc.glasbey, 6)[1],
        "marker": "D",
    },
    "ViTpretrained": {
        "color": sns.color_palette(cc.glasbey, 6)[2],
        "marker": "^",
    },
}
alpha_correct = {True: 1.0, False: 0.3}
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

        # Plot specs:
        color = models_legend[model_name]["color"]

        # Load dataframe for current model, N and clique condition
        df_path = os.path.join(os.getcwd(), "results", f"N{graph_size}", f"{model_name}_N{graph_size}_moran_results.csv")
        df = pd.read_csv(df_path)

        # Isolate trials where K is closest to empirical K0
        K_array = df['K'].unique()
        K_array_plot = np.linspace(min(K_array), max(K_array), 8, dtype=int)
        print("K values in plot: ", K_array_plot)
        # add empirical K0 as vertical dashed line:
        empirical_K0_entry = next(e for e in config["empirical_K0s"] if e["N"] == graph_size)
        K0_value = empirical_K0_entry["values"][model_name]        
        ax.axvline(K0_value, color=color, linestyle=':', linewidth=1.5)
        # Draw a horizontal dashed gray line at y=0
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.1)

        for K_value in K_array_plot:
            for correct_value in [True, False]:
                # Filter by correctness
                df_sub = df[(df['K'] == K_value) & (df['correct'] == correct_value)]
                if not df_sub.empty:
                    mean = df_sub[f'morans_I_lambda_{lambda_value}'].mean()
                    sem = df_sub[f'morans_I_lambda_{lambda_value}'].sem()
                    if graph_size < 480:
                        x_jitter = K_value + (random.uniform(-0.5, 0.5))
                    else:
                        x_jitter = K_value + (random.uniform(-1.5, 1.5))  
                    ax.errorbar(
                        x_jitter, mean, yerr=sem,
                        fmt=models_legend[model_name]["marker"],
                        color=color,
                        alpha=alpha_correct[correct_value],
                        markersize=4,
                        label=f"{model_name} {'Correct' if correct_value else 'Incorrect'}" if (K_value == K_array_plot[0]) else "" # only add label to first K value
                    )
                else:
                    # Place an "x" at y=0 for missing data
                    if graph_size < 480:
                        x_jitter = K_value + (random.uniform(-0.5, 0.5))
                    else:
                        x_jitter = K_value + (random.uniform(-1.5, 1.5))  
                    ax.plot(
                        x_jitter, 0,
                        marker="x",
                        color=color,
                        alpha=alpha_correct[correct_value],
                        markersize=4,
                        linestyle="None",
                        label=f"{model_name} {'Correct' if correct_value else 'Incorrect'} (missing)" if (K_value == K_array_plot[0]) else ""    # only add label to first K value
                    )                                     
        
    ax.set_xlabel('Clique size (K)', fontsize=12)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title(f"N = {graph_size}", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    if i == 0:
        # Get current handles and labels
        handles, labels = ax.get_legend_handles_labels()
        missing_marker_correct = Line2D(
            [0], [0],
            marker='x',
            color='gray',
            linestyle='None',
            markersize=4,
            alpha=alpha_correct[True],            
            label='No Correct Trials'
        )
        handles.append(missing_marker_correct)
        labels.append('No Correct Trials')        
        missing_marker_incorrect = Line2D(
            [0], [0],
            marker='x',
            color='gray',
            linestyle='None',
            markersize=4,
            alpha=alpha_correct[False],
            label='No Incorrect Trials'
        )        
        # Add the missing_marker handle and label
        handles.append(missing_marker_incorrect)
        labels.append('No Incorrect Trials')
        # Remove duplicates while preserving order
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend([h for h, l in unique], [l for h, l in unique], fontsize=6, loc='best')
    
    print("---------")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(f"Moran's I (lambda = {lambda_value}) of correct/incorrect trials with clique (100 graphs per K value)", fontsize=18)    
base_path = os.path.join(os.getcwd(), f'plots','NNs-visual-strategy-moransI')
# plt.savefig(base_path + '.svg', dpi=300, bbox_inches="tight")
plt.savefig(base_path + '.png', dpi=300, bbox_inches="tight")
# plt.show()
print("|Completed generating visual strategy graphs for correct/incorrect responses.")