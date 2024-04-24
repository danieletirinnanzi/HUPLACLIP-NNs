import torch
import unittest
from src.graphs_transforms import VGG16transform
from src.graphs_generation import generate_graphs as graphs_gen


class TestVGG16Transform(unittest.TestCase):
    def test_transform(self):
        # Test case 1: adjacency matrix is 4x4 and is transformed to 8x8
        adjacency_matrix = torch.tensor(
            [[1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1]]
        )
        expected_output = torch.tensor(
            [
                [1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1],
            ]
        )
        transformed_matrix = VGG16transform(adjacency_matrix, output_size=(8, 8))
        self.assertEqual(transformed_matrix.shape, (3, 8, 8))
        self.assertTrue(torch.all(torch.eq(transformed_matrix, expected_output)))

        # Test case 2: adjacency matrix is 20x20 and is transformed to 224x224
        adjacency_matrix = torch.bernoulli(torch.rand((20, 20)))
        transformed_matrix = VGG16transform(adjacency_matrix, output_size=(224, 224))
        # checking that number of edges in transformed matrix is a multiple of the number of edges in the original matrix:
        self.assertTrue(
            transformed_matrix.sum().item() % adjacency_matrix.sum().item() == 0
        )
        # checking that size size of the transformed matrix is at least 224x224:
        self.assertTrue(
            transformed_matrix.shape[1] >= 224 and transformed_matrix.shape[2] >= 224
        )

        # Test case 3: adjacency matrix is bigger than 224x224, nothing should be done
        adjacency_matrix = torch.bernoulli(torch.rand((300, 300)))
        transformed_matrix = VGG16transform(adjacency_matrix, output_size=(224, 224))
        self.assertTrue(torch.all(torch.eq(transformed_matrix, adjacency_matrix)))
