import math


def VGG16transform(adjacency_matrix, output_size=(224, 224)):
    """
    Transform for VGG16:
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
