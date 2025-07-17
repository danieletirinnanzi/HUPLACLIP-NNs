import math
import torch
import numpy as np
from typing import List, Tuple

# import input transform
from src.input_transforms import magnify_input


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
        # if input to CNN models, magnify the matrix
        if input_magnification:
            adjacency_matrix = magnify_input(adjacency_matrix)

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

        # generating upper triangular matrix
        upper_triangular = torch.triu(adjacency_matrix)
        adjacency_matrix = upper_triangular + torch.transpose(upper_triangular, 0, 1)
        adjacency_matrix.fill_diagonal_(0)
        # if input to CNN models, magnify the matrix
        if input_magnification:
            adjacency_matrix = magnify_input(adjacency_matrix)

    else:
        raise ValueError(
            "Invalid p_correction_type. Must be either 'p_increase' or 'p_reduce'"
        )

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
