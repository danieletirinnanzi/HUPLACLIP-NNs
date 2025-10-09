# SCRIPT TO TEST IO-K-NOT-KNOWN AT N=10'000
import sys
import os
import numpy as np
import torch
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import binom

# CUSTOM IMPORTS
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from src.utils import load_config
import src.graphs_generation as graphs_gen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COMMON FUNCTIONS
def p_correction(p_nodes, graph_size, clique_size):
    '''Returns the value of the corrected p-value in the graph with clique ("p_reduce" case) '''
    p_corrected = (
        p_nodes * graph_size * (graph_size - 1)
        - clique_size * (clique_size - 1)
    ) / ((graph_size - clique_size) * (graph_size + clique_size - 1))
    return p_corrected

# P(d|C=0)
def p_noclique(degree_arr, graph_size):
    ''' 
    For an array of degree values (degree_arr: ndarray of shape [N_graphs, 1, graph_size]), returns the probability that (in a graph WITHOUT the clique) a node has exactly that degree.    
    Returns: ndarray of probabilities, same shape as degree_arr
    NOTE: already log-transformed, to be used directly in decision variable computation.
    '''
    return binom.logpmf(degree_arr, 
                     graph_size-1, 
                     0.5    # "p_reduce" correction only acts on graph with clique
                     )

def degree_distribution_noclique(degree_arr, graph_size):
    '''
    Uses p_noclique to obtain the overall probability that (in a graph WITHOUT the clique) a node has exactly that degree.
    '''
    return p_noclique(degree_arr, graph_size)

# P(d|C=1)
def p_ingroup(degree_arr, graph_size, clique_size, p_corrected):
    ''' 
    For an array of degree values (degree_arr: ndarray of shape [N_graphs, 1, graph_size]), returns the probability that (in a graph WITH the clique) a node INSIDE the clique has exactly that degree.    
    Returns: ndarray of probabilities, same shape as degree_arr
    '''
    return binom.pmf(degree_arr - (clique_size-1),  # number of non-clique connections
                     graph_size - clique_size,      # number of possible non-clique nodes
                     p_corrected
                     )

def p_outgroup(degree_arr, graph_size, clique_size, p_corrected):
    ''' 
    For an array of degree values (degree_arr: ndarray of shape [N_graphs, 1, graph_size]), returns the probability that (in a graph WITH the clique) a node OUTSIDE the clique has exactly that degree.    
    Returns: ndarray of probabilities, same shape as degree_arr
    '''
    return binom.pmf(degree_arr, 
                     graph_size-1, 
                     p_corrected
                     )

def degree_distribution_clique(degree_arr, graph_size, clique_size, p_corrected):
    '''Combines p_outgroup and p_ingroup (single mixture) to obtain the overall probability that (in a graph WITH the clique) a node has exactly that degree'''
    epsilon = 1e-300
    prob = clique_size/graph_size * p_ingroup(degree_arr, graph_size, clique_size, p_corrected) + (1-clique_size/graph_size) * p_outgroup(degree_arr, graph_size, clique_size, p_corrected)    
    return prob + epsilon

# IO DEFINITION:
def ideal_observer_k_not_known(graphs_batch, graph_size, clique_size_values, p_corrected_values):
    # NOTE: "clique_size_values" and "p_corrected_values" are arrays of the possible K values and the corresponding p_corrected values
    batch_size = len(graphs_batch)
    n_clique_size_values = len(clique_size_values)
    # for each graph in the batch, compute the degree of each node and store in a 1-D array
    degrees = np.array(torch.sum(graphs_batch, dim=2))  # shape: (batch_size, 1, graph_size)
    # broadcasting degrees so that they can be used in formula for different clique size values
    degrees_broadcast = np.broadcast_to(degrees, (batch_size, n_clique_size_values, graph_size)) # shape: (batch_size, n_clique_size_values, graph_size)
    # reshaping clique size values so that they can be broadcasted
    clique_size_values_broadcast = np.reshape(clique_size_values, (1, n_clique_size_values, 1))  # shape: (1, n_clique_size_values, 1)  
    # reshaping p_corrected so that it can be broadcasted
    p_corrected_broadcast = np.reshape(p_corrected_values, (1, n_clique_size_values, 1))  # shape: (1, n_clique_size_values, 1)  
    
    # # 1st VERSION ("p_clique" = -inf (underflow) at large graph_size values)
    # # Compute degree distribution for all graphs and all clique sizes ("np.prod" is over graph_size dimension; "np.sum" is over n_clique_size_values dimension)
    # p_clique = np.log(
    #             (np.sum(
    #                 np.prod(degree_distribution_clique(degrees_broadcast, graph_size, clique_size_values_broadcast, p_corrected_broadcast),
    #                 axis = 2),
    #             axis = 1))
    #             / n_clique_size_values) # final shape: (batch_size, 1, 1)     
    
    # 2nd VERSION (with log-sum-exp trick to avoid underflow)
    sigma_kprime = np.sum(
        np.log(degree_distribution_clique(
            degrees_broadcast, graph_size, clique_size_values_broadcast, p_corrected_broadcast
        )),
        axis=2  # sum over nodes
    )  # shape: (batch_size, n_clique_size_values) -> all negative values
    
    sigma_star = np.max(sigma_kprime, axis=1, keepdims=True)   # maximum over "clique size" axis (shape: (15, 1) )
    
    # computing p_clique:   
    p_clique = sigma_star + np.log(np.sum(np.exp(sigma_kprime - sigma_star), axis = 1, keepdims=True)) - np.log(n_clique_size_values)  # shape: (batch_size, 1)
    
    # sanity check to make sure p_clique has no +/- inf:
    if p_clique.dtype == np.float64:
        if np.isinf(p_clique).any() or np.isnan(p_clique).any():
            raise ValueError("p_clique contains inf or NaN values, check numerical stability and logsumexp effectiveness.")
    
    # # DEBUGGING
    # print("Clique probabilities for all graphs:")
    # print(p_clique.squeeze())
    # print("No clique probabilities for all graphs:")
    # print(np.sum(degree_distribution_noclique(degrees, graph_size), 2).squeeze())

    # computing decision variable
    decision_variable = p_clique.squeeze() - np.sum(degree_distribution_noclique(degrees, graph_size), 2).squeeze()
    # Convert values > 0 to 1, values <= 0 to 0
    hard_output = (decision_variable > 0).astype(int)

    # # DEBUGGING
    # print("Hard output:")
    # print(hard_output)

    return hard_output.squeeze()


# TEST FUNCTION:
# read configuration file:
config = load_config(
    os.path.join("Ideal-observer_test_config.yml")
)
    
# looping over the different graph sizes in the experiment:
for graph_size in config["graph_size_values"]:

    # Calculate max clique size for testing (proportion of graph size):
    if graph_size in [100, 150, 200]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][0]
    elif graph_size in [300, 400, 480, 600]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][1]
    elif graph_size in [800, 1000, 1200]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][2]                        
    elif graph_size in [1500, 2000]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][3]                            
    elif graph_size in [3000, 5000, 10000]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][4]
    # else:
    #     max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][5]
    max_clique_size = int(
        max_clique_size_proportion_test * graph_size
    )

    # defining priors for clique sizes at current graph size:
    stat_limit = round(2 * np.log2(graph_size))
    clique_size_values = np.arange(stat_limit, (max_clique_size_proportion_test * graph_size) + 1 )
    print("Clique size values for prior: ", clique_size_values)
    # defining corresponding p_corrected values:
    p_corrected_values = p_correction(0.5, graph_size, clique_size_values)
    print("P-corrected values for each clique size value: ", p_corrected_values)
    
    # Create empty dictionaries for storing testing results:
    fraction_correct_results = {}  # Fraction correct for each clique size
    metrics_results_list = []

    # Calculate array of clique sizes for all test curriculum
    # NOTE: if max clique size is smaller than the the number of test levels, use max clique size as the number of test levels
    clique_sizes = np.linspace(
        max_clique_size,
        1,
        num=min(max_clique_size, config["testing_parameters"]["clique_testing_levels"]),
    ).astype(int)

    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # Metrics initialization (for each clique size)
        TP, FP, TN, FN = 0, 0, 0, 0  

        # Initialize fraction correct list, updated at each test iteration
        fraction_correct_list = []

        # Loop for testing iterations:
        for test_iter in range(config["testing_parameters"]["test_iterations"]):

            # Generate clique size value of each graph in the current batch
            clique_size_array_test = graphs_gen.generate_batch_clique_sizes(
                np.array([current_clique_size]),
                config["testing_parameters"]["num_test"],
            )

            # Generate validation graphs
            test = graphs_gen.generate_batch(
                config["testing_parameters"]["num_test"],
                graph_size,
                clique_size_array_test,
                config["p_correction_type"],
                False,
            )
                        
            hard_output = ideal_observer_k_not_known(test[0], graph_size, clique_size_values, p_corrected_values)
            test_labels = np.array(test[1]) # convert list to numpy array
            # print(hard_output.shape, test_labels.shape)   # DEBUGGING

            # transforming hard_output and test_labels to torch tensors:
            hard_output = torch.tensor(hard_output, dtype=torch.float32)
            test_labels = torch.tensor(test[1], dtype=torch.float32)
            
            # Compute metrics
            TP += ((hard_output == 1) & (test_labels == 1)).sum().item()
            FP += ((hard_output == 1) & (test_labels == 0)).sum().item()
            TN += ((hard_output == 0) & (test_labels == 0)).sum().item()
            FN += ((hard_output == 0) & (test_labels == 1)).sum().item()

            # updating fraction correct list with the accuracy of the current test iteration:
            fraction_correct_list.append(
                (hard_output == test_labels).sum().item()
                / (1.0 * config["testing_parameters"]["num_test"])
            )
            
            # delete unused variables
            del test, hard_output, test_labels, clique_size_array_test

        # Updating dictionary after all test iterations for current clique size have been completed:
        fraction_correct_results[current_clique_size] = round(
            sum(fraction_correct_list) / len(fraction_correct_list), 2
        )
        
        # Computing metrics:
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        # AUC - ROC cannot be calculated (no soft outputs)
        # num_params has no meaning
        metrics_results = {
            "N_value": graph_size,
            "clique_size": current_clique_size,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "F1": F1,
        }
        metrics_results_list.append(metrics_results)

        # Printing the size of the clique just tested and the corresponding test accuracy:
        print(
            f"||| Completed testing for clique = {current_clique_size}. "
            f"Average fraction correct = {fraction_correct_results[current_clique_size]}"
        )
        print("|||===========================================================")


    # - notify completion of testing:
    print(f"| Finished testing Ideal observer at N = {graph_size}.")

    # Saving accuracy results in .csv file:
    # - defining file name and path:
    file_path = os.path.join(
        os.getcwd(), "results", "ideal-observer-k-not-known", f"Ideal-observer-k-not-known_N{graph_size}_fraction_correct.csv"
    )
    # - saving the dictionary as a .csv file:
    with open(file_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["clique size", "fraction correct"])  # Add column labels
        for key, value in fraction_correct_results.items():
            writer.writerow([key, value])
    # Saving metrics results in .csv file:
    metrics_df = pd.DataFrame(metrics_results_list)    
    # - defining file name and path:
    file_path = os.path.join(
        os.getcwd(), "results", "ideal-observer-k-not-known", f"Ideal-observer-k-not-known_N{graph_size}_metrics.csv"
    )
    metrics_df.to_csv(file_path, index=False)

    print(f"- Ideal observer Results saved successfully for N = {graph_size}.")