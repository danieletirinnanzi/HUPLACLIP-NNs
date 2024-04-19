import unittest
import torch
import src.graphs_generation as gen_graphs


class TestGraphsGeneration(unittest.TestCase):

    def test_generate_graphs(self):
        # Test case 1: Generating 5 graphs with graph size 10, clique size 3, p_correction_type 'p_increase', and vgg_input False
        data, on_off = gen_graphs.generate_graphs(5, 10, 3, "p_increase", False)
        self.assertEqual(len(data), 5)
        self.assertEqual(len(on_off), 5)
        self.assertEqual(data.shape, (5, 1, 10, 10))
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 2: Generating 3 graphs with graph size 8, clique size 2, p_correction_type 'p_reduce', and vgg_input True
        data, on_off = gen_graphs.generate_graphs(3, 8, 2, "p_reduce", True)
        self.assertEqual(len(data), 3)
        self.assertEqual(len(on_off), 3)
        self.assertEqual(data.shape, (3, 3, 8, 8))
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 3: Generating 1 graph with graph size 50, clique size 5, p_correction_type 'p_increase', and vgg_input False
        data, on_off = gen_graphs.generate_graphs(1, 50, 5, "p_increase", False)
        self.assertEqual(len(data), 1)
        self.assertEqual(len(on_off), 1)
        self.assertEqual(data.shape, (1, 1, 50, 50))
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 4: Generating 0 graphs with graph size 10, clique size 3, p_correction_type 'p_reduce', and vgg_input True
        data, on_off = gen_graphs.generate_graphs(0, 10, 3, "p_reduce", True)
        self.assertEqual(len(data), 0)
        self.assertEqual(len(on_off), 0)
        self.assertEqual(data.shape, (0, 3, 10, 10))
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 5: Generating 2 graphs with graph size -5, clique size 2, p_correction_type 'p_increase', and vgg_input False
        with self.assertRaises(ValueError) as cm:
            data, on_off = gen_graphs.generate_graphs(2, -5, 2, "p_increase", False)
        self.assertEqual(
            str(cm.exception), "Graph size and clique size must be positive integers"
        )

        # Test case 6: Generating 4 graphs with graph size 10, clique size 12, p_correction_type 'p_reduce', and vgg_input True
        with self.assertRaises(ValueError) as cm:
            data, on_off = gen_graphs.generate_graphs(4, 10, 12, "p_reduce", True)
        self.assertEqual(
            str(cm.exception),
            "Decrease the clique size to avoid having a negative corrected probability of association between nodes",
        )

        # Test case 7: Generating 3 graphs with graph size 10, clique size 3, p_correction_type 'invalid', and vgg_input False
        with self.assertRaises(ValueError) as cm:
            data, on_off = gen_graphs.generate_graphs(3, 10, 3, "invalid", False)
        self.assertEqual(
            str(cm.exception),
            "Invalid p_correction_type. Must be either 'p_increase' or 'p_reduce'",
        )

        # Test case 8: Generating 1 graph with graph size 10, clique size 3, p_correction_type 'p_increase', vgg_input True, p_nodes 0.8, and p_clique 0.7
        data, on_off = gen_graphs.generate_graphs(
            1, 10, 3, "p_increase", True, p_nodes=0.8, p_clique=0.7
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(len(on_off), 1)
        self.assertEqual(data.shape, (1, 3, 10, 10))
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(on_off, list)


if __name__ == "__main__":
    unittest.main()
