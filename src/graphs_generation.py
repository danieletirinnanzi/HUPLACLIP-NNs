import math
import torch
import numpy as np

# import vgg transform
from src.image_transforms import VGG16transform as transform


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
    graph[:clique_size, :clique_size] = torch.ones(clique_size, clique_size)
    random_permutation = torch.randperm(graph_size)
    graph = graph[random_permutation[:, None], random_permutation]
    return graph


def generate_graphs(
    number_of_graphs,
    graph_size,
    clique_size,
    p_correction_type,
    vgg_input,
    p_nodes=0.5,
    p_clique=0.5,
):
    """
    Generates multiple graphs with specified correction.

    Args:
        number_of_graphs (int): Number of graphs to generate.
        graph_size (int): Number of nodes in each graph.
        clique_size (int): Size of the planted clique.
        p_correction_type (str): Type of p correction to apply.
        vgg_input (bool): Whether the graph will be used as input for VGG16.If True, the graph will be 3-dimensional.
        p_nodes (float): Probability of an edge being present between two nodes.
        p_clique (float): Probability of a graph having a planted clique.


    Returns:
        tuple: A tuple containing the generated graphs and the on_off flag for each graph.
    """

    # Testing validity of input parameters (these bounds are needed for the "p_reduce" correction)
    if (
        not isinstance(graph_size, (int, np.int32))  # also int32 values are accepted
        or not isinstance(
            clique_size, (int, np.int32)
        )  # also int32 values are accepted
        or graph_size <= 0
        or clique_size <= 0
    ):

        raise ValueError("Graph size and clique size must be positive integers")
    elif (p_nodes <= 0 or p_nodes > 1) or (p_clique <= 0 or p_clique > 1):
        raise ValueError(
            "Probability of association between nodes and probability of the presence of a clique must be included in the range [0-1]"
        )
    elif clique_size > (
        (1 + math.sqrt(1 + 4 * p_nodes * graph_size * (graph_size - 1))) / 2
    ):
        # making sure that the corrected probability of association will be positive:
        raise ValueError(
            "Decrease the clique size to avoid having a negative corrected probability of association between nodes"
        )

    # Generating the graphs
    on_off = torch.bernoulli(p_clique * torch.ones(number_of_graphs))
    # if graphs are input to VGG16, they must be 3-dimensional:
    if vgg_input:
        data = torch.zeros(number_of_graphs, 3, graph_size, graph_size)
    else:
        data = torch.zeros(number_of_graphs, 1, graph_size, graph_size)

    # differentiating between the two types of correction:
    if p_correction_type == "p_increase":
        # (OLD CORRECTION) increasing the p value of the graph without the clique so that the average degree is matched between the two graphs
        for i in range(number_of_graphs):
            # generating lower triangle of the adjacency matrix
            if not on_off[i]:
                # clique not present
                p_corrected = p_nodes + (1 - p_nodes) * (
                    (clique_size * (clique_size - 1)) / (graph_size * (graph_size - 1))
                )
                adjacency_matrix = torch.bernoulli(
                    p_corrected * torch.ones(graph_size, graph_size)
                )
            else:
                # clique present
                adjacency_matrix = torch.bernoulli(
                    p_nodes * torch.ones(graph_size, graph_size)
                )  # regular graph without clique
                # adding clique to adjacency matrix
                adjacency_matrix = plant_clique(
                    adjacency_matrix, clique_size, graph_size
                )
            # generating upper triangular matrix
            upper_triangular = torch.triu(adjacency_matrix)
            adjacency_matrix = upper_triangular + torch.transpose(
                upper_triangular, 0, 1
            )
            adjacency_matrix.fill_diagonal_(1)
            # if the graph is used as input for VGG16, the adjacency matrix must first be resized, and then made 3-dimensional:
            if vgg_input:
                # NOTE: IF INPUT SIZE < 224, OTHERWISE DO NOTHING
                adjacency_matrix = transform(adjacency_matrix)
                adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(3, 1, 1)
                data[i] = adjacency_matrix
            else:
                data[i, 0] = adjacency_matrix

    elif p_correction_type == "p_reduce":
        # (NEW CORRECTION) reducing the p value of the graph where the clique will be added
        for i in range(number_of_graphs):
            # generating lower triangle of the adjacency matrix
            if not on_off[i]:
                # clique not present (no need to correct)
                adjacency_matrix = torch.bernoulli(
                    p_nodes * torch.ones(graph_size, graph_size)
                )
            else:
                # clique present (new correction acts on the reduction of p before adding the clique):
                # - computing the new probability of association
                p_corrected = (
                    p_nodes * graph_size * (graph_size - 1)
                    - clique_size * (clique_size - 1)
                ) / ((graph_size - clique_size) * (graph_size + clique_size - 1))
                # - creating the new random graph with the probability just computed
                adjacency_matrix = torch.bernoulli(
                    p_corrected * torch.ones(graph_size, graph_size)
                )  # regular graph without clique
                # adding clique to adjacency matrix
                adjacency_matrix = plant_clique(
                    adjacency_matrix, clique_size, graph_size
                )
            # generating upper triangular matrix
            upper_triangular = torch.triu(adjacency_matrix)
            adjacency_matrix = upper_triangular + torch.transpose(
                upper_triangular, 0, 1
            )
            adjacency_matrix.fill_diagonal_(1)
            # if the graph is used as input for VGG16, the adjacency matrix must first be resized, and then made 3-dimensional:
            if vgg_input:

                print("Pre-transform")
                print(adjacency_matrix.shape)

                adjacency_matrix = transform(adjacency_matrix)

                print("Post-transform")
                print(adjacency_matrix.shape)

                adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(3, 1, 1)
                data[i] = adjacency_matrix
            else:
                data[i, 0] = adjacency_matrix

    else:
        raise ValueError(
            "Invalid p_correction_type. Must be either 'p_increase' or 'p_reduce'"
        )

    # returning the generated graphs and the on_off flag
    return data, on_off.tolist()
