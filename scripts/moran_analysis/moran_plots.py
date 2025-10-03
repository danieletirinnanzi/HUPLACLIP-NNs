# TODO: ADAPT
# Plot 3: Moran's I for K value closest to K0, only for clique (correct/incorrect distribution)
K0_values_exp1 = {
    # empirical K0 values from human experiment
    100: 29.87,
    150: 39.75,
    200: 47.62,
    300: 61.38,
    400: 74.84,
    480: 83.22,
    600: 90.19,
    800: 114.93,
    1000: 133.42
}    
lambda_value = 2  # CHANGE THIS TO TEST DIFFERENT LAMBDAS
palette_correct = {True: 'green', False: 'red'}
fig, axes = plt.subplots(1, n_N, figsize=(5 * n_N, 4))

for i, N_value in enumerate(sorted(basic_pipeline_results["N_values"])):
    print("N = ", N_value)
    ax = axes[i]
    all_points = []

    # Load dataframe for current N and clique condition
    df_path = f'./data/humans/{params["experiment_name"]}/humans_visual_strategy_{params["experiment_name"]}_N{N_value}_CLIQUE.csv'
    df = pd.read_csv(df_path)

    # Isolate trials where K is closest to empirical K0
    K_array = df['K'].unique()
    print("K values: ", K_array)
    distances = abs(K_array - K0_values_exp1[N_value])
    min_idx = np.argmin(distances)
    closest_K = K_array[min_idx]
    print(f"Closest K to empirical K0 for N={N_value}: {closest_K} (empirical K0={K0_values_exp1[N_value]})")

    # Selecting K values to show on the plot (4 above closest_K, 4 below closest_K)
    K_plot_array = K_array[(min_idx-4):(min_idx+5)]
    print("K values to plot: ", K_plot_array)

    for K_value in K_plot_array:
        for correct_value in [True, False]:
            # Filter by correctness
            df_sub = df[(df['K'] == K_value) & (df['correct'] == correct_value)]
            # Group by K and calculate mean and sem
            grouped = df_sub.groupby('K')[f'morans_I_lambda_{lambda_value}'].agg(['mean', 'sem', 'count']).reset_index()
            for _, row in grouped.iterrows():
                all_points.append({
                    'K': row['K'],
                    'mean': row['mean'],
                    'sem': row['sem'],
                    'count': row['count'],  # Number of points aggregated
                    'clique': True,
                    'correct': correct_value
                })

    # Convert to DataFrame for easier plotting
    plot_df = pd.DataFrame(all_points)
    display(plot_df)

    for correct_value in [True, False]:
        sub_df = plot_df[plot_df['correct'] == correct_value]
        color = palette_correct[correct_value]
        marker = 'o'
        linestyle = '-'
        # Plot mean values with error bars
        ax.errorbar(
            sub_df['K'], sub_df['mean'], yerr=sub_df['sem'],
            fmt=marker, color=color, linestyle=linestyle, linewidth=1, markersize=8,
            # NOTE: could be useful to add 'count' in the legend (maybe summed over all Correct/Incorrect datapoints? It would be unbalanced anyway)
            label=f"{'Correct' if correct_value else 'Incorrect'} ({int(sub_df['count'].iloc[0])} trials)"
        )
    
    # add empirical K0 as vertical dashed line:
    ax.axvline(K0_values_exp1[N_value], color='black', linestyle='--', linewidth=1.5)
    ax.text(K0_values_exp1[N_value]+0.5, ax.get_ylim()[1]*0.9, 'K₀', color='black', ha='left', va='top', fontsize=15)     

    ax.set_xlabel('Clique size (K)', fontsize=12)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title(f"N = {N_value} (K0 = {K0_values_exp1[N_value]})", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    ax.legend(fontsize=8, loc='best')
    
    print("---------")

plt.suptitle(f"Moran's I (lambda = {lambda_value}) of correct/incorrect trials with clique (4 K values below/above K closest to K0)", fontsize=18)    
plt.tight_layout(rect=[0, 0, 0.85, 1])  
base_path = f'./plots/humans/{params["experiment_name"]}/strategy/humans-visual-strategy-moransI-correct{params["experiment_name"]}'
# plt.savefig(base_path + '.svg', dpi=300, bbox_inches="tight")
# plt.savefig(base_path + '.png', dpi=300, bbox_inches="tight")
plt.show()
print("|Completed generating visual strategy graphs for correct/incorrect responses.")