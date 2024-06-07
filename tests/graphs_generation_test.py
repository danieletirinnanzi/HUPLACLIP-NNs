import unittest
import torch
import src.graphs_generation as gen_graphs


class TestGraphsGeneration(unittest.TestCase):

    def test_generate_graphs(self):
        # Test case 1: Generating 5 graphs with graph size 10 and clique size 3, p_correction_type 'p_increase', inputs for MLP (magnify_input False)
        graphs, on_off = gen_graphs.generate_graphs(5, 10, 3, "p_increase", False)
        self.assertEqual(len(graphs), 5)
        self.assertEqual(len(on_off), 5)
        self.assertEqual(graphs.shape, (5, 1, 10, 10))
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 2: Generating 3 graphs with graph size 8 and clique size 2, p_correction_type 'p_reduce', inputs for VGG/ResNet (magnify_input True)
        graphs, on_off = gen_graphs.generate_graphs(3, 8, 2, "p_reduce", True)
        self.assertEqual(len(graphs), 3)
        self.assertEqual(len(on_off), 3)
        print(graphs.shape[1])
        self.assertTrue(
            (
                graphs.shape[0] == 3
                and graphs.shape[1] == 1
                and graphs.shape[2] == 2400
                and graphs.shape[3] == 2400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 3: Generating 1 graph with graph size 50 and clique size 5, p_correction_type 'p_increase', input to MLP (magnify_input False)
        graphs, on_off = gen_graphs.generate_graphs(1, 50, 5, "p_increase", False)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertEqual(graphs.shape, (1, 1, 50, 50))
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 4: Generating 0 graphs with graph size 10 and clique size 3, p_correction_type 'p_reduce', input to VGG/ResNet (magnify_input True)
        graphs, on_off = gen_graphs.generate_graphs(0, 10, 3, "p_reduce", True)
        self.assertEqual(len(graphs), 0)
        self.assertEqual(len(on_off), 0)
        self.assertTrue(
            (
                graphs.shape[0] == 0
                and graphs.shape[1] == 1
                and graphs.shape[2] == 2400
                and graphs.shape[3] == 2400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 5: Generating 2 graphs with graph size -5 and clique size 2, p_correction_type 'p_increase', inputs to MLP (magnify_input False). Should raise a ValueError.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_graphs(2, -5, 2, "p_increase", False)
        self.assertEqual(
            str(cm.exception), "Graph size and clique size must be positive integers"
        )

        # Test case 6: Generating 4 graphs with graph size 10, clique size 12, p_correction_type 'p_reduce', inputs to VGG/ResNet (magnify_input True). Should raise a ValueError since the clique size is too large.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_graphs(4, 10, 12, "p_reduce", True)
        self.assertEqual(
            str(cm.exception),
            "Decrease the clique size to avoid having a negative corrected probability of association between nodes",
        )

        # Test case 7: Generating 3 graphs with graph size 10, clique size 3, p_correction_type 'invalid', inputs to MLP (magnify_input False). Should raise a ValueError since the p_correction_type is invalid.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_graphs(3, 10, 3, "invalid", False)
        self.assertEqual(
            str(cm.exception),
            "Invalid p_correction_type. Must be either 'p_increase' or 'p_reduce'",
        )

        # Test case 8: Generating 1 graph with graph size 10, clique size 3, p_correction_type 'p_increase', input to VGG/Resnet (magnify_input True), p_nodes 0.8, and p_clique 0.7
        graphs, on_off = gen_graphs.generate_graphs(
            1, 10, 3, "p_increase", True, p_nodes=0.8, p_clique=0.7
        )
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertTrue(
            (
                graphs.shape[0] == 1
                and graphs.shape[1] == 1
                and graphs.shape[2] == 2400
                and graphs.shape[3] == 2400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 9: Generating 1 graph with graph size 400, clique size 150, p_correction_type 'p_increase', input to VGG/ResNet (magnify_input True)
        graphs, on_off = gen_graphs.generate_graphs(1, 400, 150, "p_increase", True)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertTrue(
            (
                graphs.shape[0] == 1
                and graphs.shape[1] == 1
                and graphs.shape[2] == 2400
                and graphs.shape[3] == 2400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 10: Generating 1 graph with graph size 400, clique size 150, p_correction_type 'p_increase', input to CNN (magnify_input True)
        graphs, on_off = gen_graphs.generate_graphs(1, 400, 150, "p_increase", True)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertTrue(
            (
                graphs.shape[0] == 1
                and graphs.shape[1] == 1
                and graphs.shape[2] == 2400
                and graphs.shape[3] == 2400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )

        # Test case 11: Generating 4 graph with graph size 310, clique size 100, p_correction_type 'p_increase', input to CNN (magnify_input True). Should raise a ValueError saying "The shape of the adjacency matrix is not evenly divisible by the output size."
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_graphs(4, 310, 100, "p_increase", True)
        self.assertEqual(
            str(cm.exception),
            "The shape of the adjacency matrix is not evenly divisible by the output size.",
        )
