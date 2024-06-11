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
