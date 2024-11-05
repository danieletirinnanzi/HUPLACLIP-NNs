import torch
import unittest
from src.input_transforms import magnify_input, find_patch_size


class TestInputTransform(unittest.TestCase):

    def test_magnify_input(self):
        # Test case 1: 100x100 adjacency matrix is resized to 2400x2400
        adjacency_matrix = torch.bernoulli(torch.rand(100, 100))
        # Call the magnify_input function
        transformed_matrix = magnify_input(adjacency_matrix)
        # Check if the transformed matrix has the correct shape
        self.assertEqual(transformed_matrix.shape, (2400, 2400))
        # Check if the transformed matrix is a torch.Tensor
        self.assertIsInstance(transformed_matrix, torch.Tensor)

        # Test case 2: if the adjacency matrix size is not evenly divisible by the output size, a ValueError is raised
        with self.assertRaises(ValueError):
            magnify_input(torch.randn(130, 130))

        # Test case 3: adjacency matrix is 4x4 and is expanded to 8x8
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
        # Call the magnify_input function
        transformed_matrix = magnify_input(adjacency_matrix, output_size=(8, 8))
        # Check if the transformed matrix has the correct shape
        self.assertEqual(transformed_matrix.shape, (8, 8))
        # Check if the transformed matrix is equal to the expected output
        self.assertTrue(torch.all(torch.eq(transformed_matrix, expected_output)))

        # Test case 4: adjacency matrix is 20x20 and is transformed to 2400x2400
        adjacency_matrix = torch.bernoulli(torch.rand((20, 20)))
        transformed_matrix = magnify_input(adjacency_matrix)
        # checking that number of edges in transformed matrix is a multiple of the number of edges in the original matrix:
        self.assertTrue(
            transformed_matrix.sum().item() % adjacency_matrix.sum().item() == 0
        )
        # checking that size size of the transformed matrix is exactly 2400x2400:
        self.assertTrue(
            transformed_matrix.shape[0] == 2400 and transformed_matrix.shape[1] == 2400
        )

    def test_find_patch_size(self):

        # Test case where graph is 100 -> the function should find a patch size of 5 (100/20 = 5)
        self.assertEqual(find_patch_size(100), 5)

        # Test case where graph is 200 -> the function should find a patch size of 10 (200/20 = 10)
        self.assertEqual(find_patch_size(200), 10)

        # Test case where graph is 224 -> the function should find a patch size of 14 (NOTE: this is slightly inaccurate, in the original paper the patch size is 16, but this is a good approximation)
        self.assertEqual(find_patch_size(224), 14)

        # Test case where graph size is 300 -> the function should find a patch size of 15
        self.assertEqual(find_patch_size(300), 15)

        # Test case where graph size is 400 -> the function should find a patch size of 20
        self.assertEqual(find_patch_size(400), 20)

        # Test case where graph size is 500 -> the function should find a patch size of 25
        self.assertEqual(find_patch_size(500), 25)

        # Test case where graph size is 600 -> the function should find a patch size of 30
        self.assertEqual(find_patch_size(600), 30)

        # Test case where graph size is 700 -> the function should find a patch size of 35
        self.assertEqual(find_patch_size(700), 35)

        # Test case where graph size is 800 -> the function should find a patch size of 40
        self.assertEqual(find_patch_size(800), 40)

        # Test case where graph size is 900 -> the function should find a patch size of 45
        self.assertEqual(find_patch_size(900), 45)

        # Test case where graph size is 1000 -> the function should find a patch size of 50
        self.assertEqual(find_patch_size(1000), 50)

        # Test case where graph size is 331 -> it is a prime number, so the function should return 1 as patch_size, and notify the user with a ValueError
        with self.assertRaises(ValueError):
            find_patch_size(331)
