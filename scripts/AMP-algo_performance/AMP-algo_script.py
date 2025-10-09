# SCRIPT TO TEST AMP-ALGO AT N=10'000
import sys
import os
import numpy as np
import torch
import pandas as pd
import csv
import math
from scipy.stats import norm
from scipy import integrate
from typing import List, Tuple

# CUSTOM IMPORTS
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from src.utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GRAPHS GENERATION
def plant_clique(graph, clique_size, graph_size):
    """
    Plants a clique of specified size within a given graph.

    Args:
        graph (torch.Tensor): Adjacency matrix representing the graph.
        size (int): Size of the clique to be planted.
        graph_size (int): Total number of nodes in the graph.

    Returns:
        torch.Tensor: Modified adjacency matrix with the planted clique.
    """
    # adding the planted clique to the adjacency matrix in the top left corner
    graph[:clique_size, :clique_size] = torch.ones(clique_size, clique_size)
    # creating a random permutation of the nodes
    random_permutation = torch.randperm(graph_size)
    # placing the rows and columns of the adjacency matrix according to the random permutation
    graph = graph[random_permutation[:, None], random_permutation]
    return graph


def generate_graph(
    on_off_label: int,
    graph_size: int,
    clique_size: int,
    p_correction_type: str,
    input_magnification: bool,
    p_nodes: float = 0.5,
) -> torch.Tensor:
    """
    Generates a graph with or without a planted clique with the specified correction.

    Args:
        on_off_label (int): Label indicating whether the graph will have a planted clique.
        graph_size (int): Number of nodes in the graph.
        clique_size (int): Size of the planted clique.
        p_correction_type (str): Type of p correction to apply.
        input_magnification (bool): Whether the input needs to be magnified. If True, the graph will be made a 2400x2400 tensor.
        p_nodes (float): Probability of an edge being present between two nodes.

    Raises:
        ValueError: If an invalid p_correction_type is provided.
        ValueError: If the clique size is too large for the graph size and the "p_reduce" correction type is used.

    Returns:
        torch.Tensor: Adjacency matrix representing a graph with the specified features.
    """

    # differentiating between the two types of correction:
    if p_correction_type == "p_increase":
        # (OLD CORRECTION) increasing the p value of the graph without the clique so that the average degree is matched between the two graphs

        # generating lower triangle of the adjacency matrix
        if on_off_label:
            # clique present
            adjacency_matrix = torch.bernoulli(
                p_nodes * torch.ones(graph_size, graph_size)
            )  # regular graph without clique
            # adding clique to adjacency matrix
            adjacency_matrix = plant_clique(adjacency_matrix, clique_size, graph_size)

        else:
            # clique not present
            p_corrected = p_nodes + (1 - p_nodes) * (
                (clique_size * (clique_size - 1)) / (graph_size * (graph_size - 1))
            )
            adjacency_matrix = torch.bernoulli(
                p_corrected * torch.ones(graph_size, graph_size)
            )

        # generating upper triangular matrix
        upper_triangular = torch.triu(adjacency_matrix)
        adjacency_matrix = upper_triangular + torch.transpose(upper_triangular, 0, 1)
        adjacency_matrix.fill_diagonal_(0)

    elif p_correction_type == "p_reduce":
        # (NEW CORRECTION) reducing the p value of the graph where the clique will be added

        # - making sure that the "p_reduce" corrected probability of association will be positive for requested clique size
        if clique_size > (
            (1 + math.sqrt(1 + 4 * p_nodes * graph_size * (graph_size - 1))) / 2
        ):
            clique_limit = int(
                (1 + math.sqrt(1 + 4 * p_nodes * graph_size * (graph_size - 1))) / 2
            )
            raise ValueError(
                f"Clique size {clique_size} in a graph of size {graph_size} leads to a negative corrected probability of association between nodes. Please choose a clique size smaller than {round(clique_limit)}"
            )

        # generating lower triangle of the adjacency matrix
        if on_off_label:
            # clique present (new correction acts on the reduction of p before adding the clique):
            # - computing the new probability of association
            p_corrected = (
                p_nodes * graph_size * (graph_size - 1)
                - clique_size * (clique_size - 1)
            ) / ((graph_size - clique_size) * (graph_size + clique_size - 1))
            # - creating the new random graph with the probability just computed
            adjacency_matrix = torch.bernoulli(
                p_corrected * torch.ones(graph_size, graph_size)
            )  # regular graph without clique, but with reduced p value
            # adding clique to adjacency matrix
            adjacency_matrix = plant_clique(adjacency_matrix, clique_size, graph_size)

        else:
            # clique not present (no need to correct)
            adjacency_matrix = torch.bernoulli(
                p_nodes * torch.ones(graph_size, graph_size)
            )

        # generating upper triangular matrix (MODIFIED FOR AMP ALGO COMPATIBILITY)
        upper_triangular = torch.triu(adjacency_matrix)
        adjacency_matrix = upper_triangular + torch.transpose(upper_triangular, 0, 1)
        adjacency_matrix.fill_diagonal_(0)

    else:
        raise ValueError(
            "Invalid p_correction_type. Must be either 'p_increase' or 'p_reduce'"
        )

    # transform zeros to -1s, keeping diagonal elements = 0
    adjacency_matrix[adjacency_matrix == 0] = -1
    adjacency_matrix.fill_diagonal_(0)

    return adjacency_matrix


def generate_batch_clique_sizes(allowed_clique_sizes, batch_size):
    """
    Generate the clique sizes for each graph in the batch (based on the allowed clique size values).

    Parameters:
    allowed_clique_sizes (np.ndarray): Allowed clique size values.
    batch_size (int): Size of the batch to generate.

    Returns:
    np.ndarray: Array of generated clique sizes for each graph in the batch.
    """
    # TESTING INPUT VALUES:
    if not isinstance(allowed_clique_sizes, np.ndarray):
        raise ValueError("allowed_clique_sizes must be a numpy array")
    if not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    # if more than one clique size value is allowed, checking that last value of array is the smallest one:
    if len(allowed_clique_sizes) > 1:
        if min(allowed_clique_sizes) != allowed_clique_sizes[-1]:
            raise ValueError(
                "the last provided clique size value is not the smallest one: something might be wrong with the curriculum training procedure"
            )
    # END TESTING INPUT VALUES

    # probability of single clique size value is < 0.25:
    if 1 / len(allowed_clique_sizes) < 0.25:
        # - set minimum probability value for lowest clique size
        prob_lowest = 0.25
        # - calculate probability for remaining values (easier versions):
        prob_easier = (1 - 0.25) / (len(allowed_clique_sizes) - 1)
        # - define array of probabilities:
        allowed_clique_sizes_probs = np.full(len(allowed_clique_sizes) - 1, prob_easier)
        allowed_clique_sizes_probs = np.concatenate(
            (allowed_clique_sizes_probs, [prob_lowest])
        )

    # probability of single clique size value is >= 0.25
    else:
        # - simply define single prob value
        prob_each = 1 / len(allowed_clique_sizes)
        allowed_clique_sizes_probs = np.full(len(allowed_clique_sizes), prob_each)

    # Normalize the probabilities to ensure they sum to 1 (in case of rounding errors)
    allowed_clique_sizes_probs /= np.sum(allowed_clique_sizes_probs)

    # Generate the clique size array
    batch_clique_sizes = np.random.choice(
        allowed_clique_sizes, batch_size, p=allowed_clique_sizes_probs
    )

    return batch_clique_sizes


def generate_batch(
    number_of_graphs: int,
    graph_size: int,
    clique_size_array: List[int],
    p_correction_type: str,
    input_magnification: bool,
    p_clique: float = 0.5,
    p_nodes: float = 0.5,
) -> Tuple:
    """
    Generates batch of graphs. The size of each graph in the batch can be different, as well as the clique size.

    Args:
        number_of_graphs (int): Number of graphs in the batch.
        graph_size (int): Number of nodes in the graphs of the batch.
        clique_size_array (List[int]): Size of the planted clique in each graph of the batch.
        p_correction_type (str): Type of p correction to apply.
        input_magnification (bool): Whether the input needs to be magnified. If True, the graph will be made a 2400x2400 tensor.
        p_clique (float): Probability of a graph having a planted clique. Default is 0.5.
        p_nodes (float): Probability of an edge being present between two nodes. Default is 0.5.

    Returns:
        tuple: A tuple containing the batch of graphs and the corresponding on_off flags.
    """

    # Testing validity of input parameters:
    if number_of_graphs == 0:
        raise ValueError("At least one graph must be generated.")
    # - testing that number of graphs and clique size array length match:
    if len(clique_size_array) != number_of_graphs:
        raise ValueError(
            "The number of graphs must be the same of clique size array length"
        )
    # - testing that all values are positive integers:
    if (
        not all(
            isinstance(size, (int, np.int32, np.int64)) for size in clique_size_array
        )
        or any(size <= 0 for size in clique_size_array)
        or graph_size <= 0
    ):
        raise ValueError(
            "All clique size values and graph size value must be positive integers"
        )
    # - testing that the probability values are in the [0-1] range:
    elif (p_nodes <= 0 or p_nodes > 1) or (p_clique <= 0 or p_clique > 1):
        raise ValueError(
            "Probability of association between nodes and probability of the presence of a clique must be included in the range [0-1]"
        )

    # Generating the labels (with/without clique)
    on_off = torch.bernoulli(p_clique * torch.ones(number_of_graphs))
    # Generating the graph_list that will contain the adjacency matrices (now filled with zeros, will be filled with the actual adjacency matrices later on)
    # - magnified input (for all models except MLP):
    if input_magnification:
        graphs = torch.zeros(number_of_graphs, 1, 2400, 2400)
    # - standard input (for MLP):
    else:
        graphs = torch.zeros(number_of_graphs, 1, graph_size, graph_size)

    for i in range(number_of_graphs):
        graphs[i, 0] = generate_graph(
            on_off[i],
            graph_size,
            clique_size_array[i],
            p_correction_type,
            input_magnification,
            p_nodes,
        )

    # returning the generated graphs and the on_off flag
    return graphs, on_off.tolist()

# AMP-ALGO DEFINITIONS
# ---------- params ----------
save_mu = {}
save_L = {}
d = 5  # maximum exponent degree

# ---------- precompute mu and L up to level T ----------
def compute_mu_L(T):
    # returns lists indexed by level (0..T). We will keep index 0 unused for mu (mu[1]=1)
    fact = [math.factorial(k) for k in range(d+1)]
    mu_vals = [None] * (T + 1)
    L_vals = [None] * (T + 1)
    mu_vals[1] = 1.0
    # L(1) depends on mu(1)
    def L_integrand_factory(mu_l):
        def integrand(x):
            poly = 0.0
            xval = x
            for k in range(d+1):
                poly += (mu_l**k) * (xval**k) / fact[k]
            return norm.pdf(x) * (poly**2)
        return integrand
    L_vals[1] = math.sqrt(integrate.quad(L_integrand_factory(mu_vals[1]), -15, 15)[0])
    # now iteratively compute mu_l and L_l
    for l in range(2, T+1):
        # integrand for mu_l uses p(., l-1) with mu_{l-1}, L_{l-1}
        mu_prev = mu_vals[l-1]
        L_prev = L_vals[l-1]
        def mu_integrand(x):
            z = mu_prev + x
            poly = 0.0
            zpow = 1.0
            for k in range(d+1):
                poly += (mu_prev**k) * (zpow) / fact[k]
                zpow *= z
            return norm.pdf(x) * (poly / L_prev)
        mu_l = integrate.quad(mu_integrand, -15, 15)[0]
        mu_vals[l] = mu_l
        # now L_l
        L_vals[l] = math.sqrt(integrate.quad(L_integrand_factory(mu_l), -15, 15)[0])
    # precompute coefficient tensors coeffs[l] = [mu_l**k / k!] for k=0..d
    coeffs = []
    for l in range(T+1):
        if l == 0:
            coeffs.append(None)
        else:
            coeffs.append(torch.tensor([ (mu_vals[l]**k) / math.factorial(k) for k in range(d+1) ], dtype=torch.get_default_dtype()))
    return mu_vals, L_vals, coeffs

# ---------- vectorized polynomial for torch tensors ----------
def p_torch(z, l, coeffs, L_vals):
    # z: torch tensor (any shape)
    if l == 0:
        return torch.ones_like(z)
    coeff = coeffs[l]                 # tensor shape (d+1,)
    res = torch.zeros_like(z)
    z_pow = torch.ones_like(z)        # z**0
    for k in range(coeff.shape[0]):
        res = res + coeff[k] * z_pow
        z_pow = z_pow * z
    return res / float(L_vals[l])

# ---------- message passing ----------
def CliqueMarginals(W, t, coeffs, L_vals):
    # W: torch [n,n], symmetric
    n = W.shape[0]
    A = W / math.sqrt(n)
    A = A.clone()
    A.fill_diagonal_(0.0)

    theta_ij = torch.ones((n, n), dtype=W.dtype, device=W.device)
    theta_ij.fill_diagonal_(0.0)

    for it in range(t):
        p_vals = p_torch(theta_ij, it, coeffs, L_vals)    # shape (n,n)
        theta_i = torch.sum(A * p_vals, dim=1)            # shape (n,)
        theta_ij = theta_i.unsqueeze(1) - A * p_vals.T    # use p_vals.T instead of recomputing
        theta_ij.fill_diagonal_(0.0)
    return theta_i

# ---------- decision procedure ----------
def CliqueDecision(W, n, k, t=2, eps=0.0, method='eig_threshold', mu_t=None, device='cpu'):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.get_default_dtype(), device=device)
    else:
        W = W.to(device)

    mu_vals, L_vals, coeffs = compute_mu_L(t)
    x = CliqueMarginals(W, t, coeffs, L_vals)  # torch tensor

    mu_t_val = float(mu_vals[t]) if mu_t is None else float(mu_t)
    candidates_mask = (x > mu_t_val)
    index = torch.nonzero(candidates_mask, as_tuple=False).view(-1)

    if index.numel() == 0:
        return 0

    m = int(index.numel())
    W_sub = W.index_select(0, index).index_select(1, index) / math.sqrt(max(1, m))
    eigvals, eigvecs = torch.linalg.eigh(W_sub)
    lambda_max = float(eigvals[-1].cpu().item())

    if method == 'eig_threshold':
        return int(lambda_max > 2.0 + eps)

    else:
        raise ValueError("Unknown method")

# TEST FUNCTION
# TEST FUNCTION
# read configuration file:
config = load_config(
    os.path.join("AMP-algo_test_config.yml")
)  # CHANGE THIS TO PERFORM DIFFERENT EXPERIMENTS

# looping over the different graph sizes in the experiment:
for graph_size in config["graph_size_values"]:

    # Create empty dictionaries for storing testing results:
    fraction_correct_results = {}  # Fraction correct for each clique size
    metrics_results = {}  # Metrics dictionary

    # Calculate max clique size for testing (proportion of graph size):
    if graph_size in [100, 150, 200]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][0]
    elif graph_size in [300, 400, 480, 600]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][1]
    elif graph_size in [800, 1000, 1200]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][2]                        
    elif graph_size in [1500, 2000]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][3]                            
    elif graph_size in [3000, 5000]:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][4]
    else:
        max_clique_size_proportion_test = config["testing_parameters"]["max_clique_size_proportion_test"][5]
    max_clique_size = int(
        max_clique_size_proportion_test * graph_size
    )

    # Calculate array of clique sizes for all test curriculum
    # NOTE: if max clique size is smaller than the the number of test levels, use max clique size as the number of test levels
    clique_sizes = np.linspace(
        max_clique_size,
        1,
        num=min(max_clique_size, config["testing_parameters"]["clique_testing_levels"]),
    ).astype(int)
    
    # Metrics initialization (local to each GPU)
    TP, FP, TN, FN = 0, 0, 0, 0  

    # Loop for decreasing clique sizes
    for current_clique_size in clique_sizes:

        # Initialize fraction correct list, updated at each test iteration
        fraction_correct_list = []

        # Loop for testing iterations:
        for test_iter in range(config["testing_parameters"]["test_iterations"]):

            # Generate clique size value of each graph in the current batch
            clique_size_array_test = generate_batch_clique_sizes(
                np.array([current_clique_size]),
                config["testing_parameters"]["num_test"],
            )

            # Generate validation graphs
            test = generate_batch(
                config["testing_parameters"]["num_test"],
                graph_size,
                clique_size_array_test,
                config["p_correction_type"],
                False,
            )
            
            hard_output = torch.zeros([config["testing_parameters"]["num_test"]])
            
            for graph_index, graph in enumerate(test[0]):
                AMP_output = CliqueDecision(
                    graph.squeeze(), graph_size, current_clique_size
                )
                hard_output[graph_index] = AMP_output
            
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

        # Printing the size of the clique just tested and the corresponding test accuracy:
        print(
            f"||| Completed testing for clique = {current_clique_size}. "
            f"Average fraction correct = {fraction_correct_results[current_clique_size]}"
        )
        print("|||===========================================================")

    # - notify completion of testing:
    print(f"| Finished testing AMP algo at N = {graph_size}.")

    # Computing metrics:
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    # AUC - ROC cannot be calculated (no soft outputs)
    # num_params has no meaning
    metrics_results = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "AUC_ROC": np.nan,
        "total_params": np.nan,
    }

    # Saving accuracy results in .csv file:
    # - defining file name and path:
    file_path = os.path.join(
        os.getcwd(), "results", f"AMP-algo_N{graph_size}_fraction_correct.csv"
    )
    # - saving the dictionary as a .csv file:
    with open(file_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["clique size", "fraction correct"])  # Add column labels
        for key, value in fraction_correct_results.items():
            writer.writerow([key, value])
    # Saving metrics results in .csv file:
    # - defining file name and path:
    file_path = os.path.join(
        os.getcwd(), "results", f"AMP-algo_N{graph_size}_metrics.csv"
    )
    # - saving the dictionary as a .csv file:
    pd.DataFrame([metrics_results]).to_csv(file_path, index=False)

    print(f"- AMP Results saved successfully for N = {graph_size}.")