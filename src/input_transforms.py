import math


# Function needed to resize images to 2400x2400 tensors (for ConvNets and all other models except MLP)
def magnify_input(adjacency_matrix, output_size=(2400, 2400)):
    """
    Transform for creating inputs for Convolutional Neural Networks and CNN models trained on ImageNet. The transformation is applied to the adjacency matrix of the graph:
    - adjacency matrix is magnified (so that it is exactly 2400x2400)
    - adjacency matrix keeps having 1 channel

    Args:
        adjacency_matrix (torch.Tensor): The adjacency matrix of the graph.
        output_size (tuple): The desired output size of the transformed adjacency matrix.

    Returns:
        torch.Tensor: The transformed adjacency matrix.
    """

    # checking that the shape of the adjacency matrix is evenly divisible by the output size:
    if (
        output_size[0] % adjacency_matrix.shape[0] != 0
        or output_size[1] % adjacency_matrix.shape[1] != 0
    ):
        raise ValueError(
            "The shape of the adjacency matrix is not evenly divisible by the output size."
        )

    # calculating magnification factor:
    factor = int(output_size[0] / adjacency_matrix.shape[0])

    # repeating each element in each line "factor" times
    adjacency_matrix = adjacency_matrix.repeat_interleave(factor, dim=1)

    # repeating each expanded line "factor" times
    adjacency_matrix = adjacency_matrix.repeat_interleave(factor, dim=0)

    return adjacency_matrix


# - Function needed to define the patch size of the Visual Transformer architecture based on the graph size:
def find_patch_size(graph_size):
    """
    Find the patch size for the ViT model based on the size of the input image. The patch size is the size of the square
    that is used as an input to the model.

    Args:
        graph_size (int): The size of the graph.

    Returns:
        int: The patch size.
        int: The size of the input image.

    Raises:
        ValueError: If a patch size different than 1 cannot be found.
    """

    from src.graphs_generation import generate_graphs as gen_graphs

    # storing size of the input image:
    single_graph = gen_graphs(
        1, graph_size, int(graph_size / 2), "p_increase", input_magnification=False
    )[
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

    print(f"Patch size: {patch_size}, Image size: {image_size}")

    return patch_size, image_size
