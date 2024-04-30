import torch
import unittest
from src.input_transforms import imageNet_transform, find_patch_size


class TestInputTransform(unittest.TestCase):

    def test_imageNet_transform(self):
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
        transformed_matrix = imageNet_transform(adjacency_matrix, output_size=(8, 8))
        self.assertEqual(transformed_matrix.shape, (3, 8, 8))
        self.assertTrue(torch.all(torch.eq(transformed_matrix, expected_output)))

        # Test case 2: adjacency matrix is 20x20 and is transformed to 224x224
        adjacency_matrix = torch.bernoulli(torch.rand((20, 20)))
        transformed_matrix = imageNet_transform(
            adjacency_matrix, output_size=(224, 224)
        )
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
        transformed_matrix = imageNet_transform(
            adjacency_matrix, output_size=(224, 224)
        )
        self.assertTrue(torch.all(torch.eq(transformed_matrix, adjacency_matrix)))

    def test_find_patch_size(self):

        # Test case where graph is 224 -> the function should find a patch size of 14 (NOTE: this is slightly inaccurate, in the original paper the patch size is 16, but this is a good approximation)
        self.assertEqual(find_patch_size(224), 14)

        # Test case where graph size is 400 -> the function should find a patch size of 20
        self.assertEqual(find_patch_size(40), 2)

        # Test case where graph size is 50 -> the function should find a patch size of 2
        self.assertEqual(find_patch_size(50), 2)

        # Test case where graph size is 37 -> the function should raise an error
        with self.assertRaises(ValueError):
            find_patch_size(37)

        # Test case where graph size is 1 -> the function should raise an error
        with self.assertRaises(ValueError):
            find_patch_size(1)

        # Test case where graph size is 0 -> the function should raise an error
        with self.assertRaises(ValueError):
            find_patch_size(0)

        # Test case where graph size is negative -> the function should raise an error
        with self.assertRaises(ValueError):
            find_patch_size(-10)
