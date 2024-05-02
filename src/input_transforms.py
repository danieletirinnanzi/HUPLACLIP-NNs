import math


def imageNet_transform(adjacency_matrix, output_size=(224, 224)):
    """
    Transform for models trained on ImageNet dataset. The transformation is applied to the adjacency matrix of the graph:
    - adjacency matrix should have 3 channels, not only one
    - adjacency matrix is magnified (so that it is bigger than 224x224) only if graph_size is smaller than 224)

    Args:
        adjacency_matrix (torch.Tensor): The adjacency matrix of the graph.
        output_size (tuple): The desired output size of the transformed adjacency matrix.

    Returns:
        torch.Tensor: The transformed adjacency matrix.
    """
    if adjacency_matrix.shape[0] >= output_size[0]:
        # adjacency matrix is already bigger than the expected size, only 3 channels are added:
        adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(3, 1, 1)
        return adjacency_matrix

    # calculating magnification factor:
    factor = math.ceil(
        output_size[0] / adjacency_matrix.shape[0]
    )  # rounding up to the nearest integer, so that input is never smaller than 224

    # repeating each element in each line "factor" times
    adjacency_matrix = adjacency_matrix.repeat_interleave(factor, dim=1)

    # repeating each line "factor" times
    adjacency_matrix = adjacency_matrix.repeat_interleave(factor, dim=0)

    # adding 3 channels
    adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(3, 1, 1)

    return adjacency_matrix


# - Function needed to define the patch size based on the graph size:
def find_patch_size(graph_size):
    """
    Find the patch size for the ViT model based on the size of the input image. The patch size is the size of the square
    that is used as an input to the model.

    Args:
        graph_size (int): The size of the graph.

    Returns:
        int: The patch size.

    Raises:
        ValueError: If a patch size different than 1 cannot be found.
    """

    from src.graphs_generation import generate_graphs as gen_graphs

    # storing size of the input image:
    single_graph = gen_graphs(1, graph_size, int(graph_size / 2), "p_increase", True)[
        0
    ]  # generating a single graph (already includes the magnification factor for ImageNet models)
    image_size = single_graph.shape[-1]

    # defining patch size based on the image size:
    patch_size = (
        image_size // 20  # floor division
    )  # initial value of patch_size is 1/20 of the image size, is then increased until it is a divisor of the graph size
    # increase patch size until an integer divisor of the image size is found:
    while patch_size <= image_size:
        if image_size % patch_size == 0:
            break
        patch_size += 1

    if patch_size == image_size:  # if no divisor is found, patch size is set to 1
        patch_size = 1

    # notifying user if patch size different than 1 cannot be found:
    if patch_size == 1:
        raise ValueError(
            "A patch size different than 1 cannot be found. The chosen graph size might be incompatible with the ViT model."
        )

    return patch_size
