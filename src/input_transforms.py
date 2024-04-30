import math
import src.graphs_generation as gen_graphs


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

    # storing rescaling factor for the adjacency matrix:
    factor = imageNet_transform(
        gen_graphs.generate_graphs(1, graph_size, graph_size, "p_increase", True)[0]
    )[1]

    print(factor)

    # calculating the size of the image:
    image_size = graph_size * factor

    patch_size = (
        image_size // 20  # floor division
    )  # initial value of patch_size is 1/20 of the graph size, is then increased until it is a divisor of the graph size
    while patch_size <= image_size:
        if image_size % patch_size == 0:
            break
        patch_size += 1

    patch_size = patch_size if patch_size <= image_size else 1

    if patch_size == 1:
        raise ValueError("A patch size different than 1 cannot be found.")

    return patch_size
