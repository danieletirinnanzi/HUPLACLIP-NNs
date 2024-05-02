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

        # Test case where graph is 100 -> the function should find a patch size of 20 (100 becomes 300 after resizing, then 300/20 = 15)
        self.assertEqual(find_patch_size(100), 15)

        # Test case where graph is 200 ->the function should find a patch size of (200 becomes 400 after resizing, then 400/20 = 20)
        self.assertEqual(find_patch_size(200), 20)

        # Test case where graph is 224 -> the function should find a patch size of 14 (224 is not resized, and the function starts searching from graph_size/20=11, then increases the patch size. NOTE: this is slightly inaccurate, in the original paper the patch size is 16, but this is a good approximation)
        self.assertEqual(find_patch_size(224), 14)

        # Test case where graph size is 300 -> the function should find a patch size of 15
        self.assertEqual(find_patch_size(300), 15)

        # Test case where graph size is 400 -> the function should find a patch size of 20
        self.assertEqual(find_patch_size(400), 20)

        # Test case where graph size is 500 -> the function should find a patch size of 25
        self.assertEqual(find_patch_size(50), 25)

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
