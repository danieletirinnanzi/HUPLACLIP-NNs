import unittest
import torch
import numpy as np

# CUSTOM IMPORTS
import src.graphs_generation as gen_graphs
from src.graphs_generation import generate_batch_clique_sizes


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

    def test_generate_graphs(self):

        # TESTING ERROR MESSAGES FIRST:
        # Test case 1: Generating 0 graphs with graph size 10 and clique size 3, p_correction_type 'p_reduce'. Should raise a ValueError
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(0, 10, [3], "p_increase", False)
        self.assertEqual(str(cm.exception), "At least one graph must be generated.")

        # Test case 2: generating 2 graphs but indicating only 1 clique size. Should raise a ValueError.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(2, 10, [3], "p_increase", False)
        self.assertEqual(
            str(cm.exception),
            "The number of graphs must be the same of clique size array length",
        )

        # Test case 2: Generating 2 graphs with graph size -5 and clique size 2, p_correction_type 'p_increase'. Should raise a ValueError.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(
                2, -5, [2, 2], "p_increase", False
            )
        self.assertEqual(
            str(cm.exception),
            "All clique size values and graph size value must be positive integers",
        )

        # Test case 3: Generating 4 graphs with graph size 10, clique size 12, p_correction_type 'p_reduce'. Should raise a ValueError since the clique size is too large.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(
                4,
                10,
                [12, 12, 12, 12],
                "p_reduce",
                False,
            )
        self.assertEqual(
            str(cm.exception),
            "Clique size 12 in a graph of size 10 leads to a negative corrected probability of association between nodes. Please choose a clique size smaller than 7",
        )

        # Test case 4: Generating 3 graphs with graph size 10, clique size 3, p_correction_type 'invalid'. Should raise a ValueError since the p_correction_type is invalid.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(
                3,
                10,
                [3, 3, 3],
                "invalid",
                True,
            )
        self.assertEqual(
            str(cm.exception),
            "Invalid p_correction_type. Must be either 'p_increase' or 'p_reduce'",
        )

        # Test case 5: Generating 1 graph with graph size 10, clique size 3, p_correction_type 'p_reduce', p_nodes 0.8, and p_clique 1.4. Should raise a ValueError since p_clique is greater than 1.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(
                1,
                10,
                [3],
                "p_reduce",
                True,
                p_clique=1.4,
                p_nodes=0.8,
            )
        self.assertEqual(
            str(cm.exception),
            "Probability of association between nodes and probability of the presence of a clique must be included in the range [0-1]",
        )

        # Test case 6: Generating 1 graph with graph size 34, clique size 3, p_correction_type 'p_reduce', input to CNN (input_magnification True). Should raise a ValueError since the graph size is not evenly divisible by the output size.
        with self.assertRaises(ValueError) as cm:
            graphs, on_off = gen_graphs.generate_batch(1, 34, [3], "p_reduce", True)
        self.assertEqual(
            str(cm.exception),
            "The shape of the graph is not evenly divisible by the output size. Graph size is 34, output size is 2400.",
        )

        # TESTING CORRECT OUTPUTS:
        # Test case 7: Generating 5 graphs with graph size 10 and clique size 3, p_correction_type 'p_increase', inputs to MLP (input_magnification False)
        clique_size_array = [3, 3, 3, 3, 3]
        graphs, on_off = gen_graphs.generate_batch(
            5,
            10,
            clique_size_array,
            "p_increase",
            False,
        )
        self.assertEqual(len(graphs), 5)
        self.assertEqual(len(on_off), 5)
        self.assertEqual(graphs.shape, (5, 1, 10, 10))
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 8: Generating 3 graphs with graph size 8 and clique size 2, p_correction_type 'p_reduce', inputs to MLP (input_magnification False)
        clique_size_array = [2, 2, 2]
        graphs, on_off = gen_graphs.generate_batch(
            3, 8, clique_size_array, "p_reduce", False
        )
        self.assertEqual(len(graphs), 3)
        self.assertEqual(len(on_off), 3)
        self.assertTrue(
            (
                graphs.shape[0] == 3
                and graphs.shape[1] == 1
                and graphs.shape[2] == 8
                and graphs.shape[3] == 8
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 9: Generating 1 graph with graph size 50 and clique size 5, p_correction_type 'p_increase', input to MLP (input_magnification False)
        graphs, on_off = gen_graphs.generate_batch(1, 50, [5], "p_increase", False)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertEqual(graphs.shape, (1, 1, 50, 50))
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 10: Generating 1 graph with graph size 10, clique size 3, p_correction_type 'p_increase', p_nodes 0.8, and p_clique 0.7, input to MLP (input_magnification False)
        graphs, on_off = gen_graphs.generate_batch(
            1,
            10,
            [3],
            "p_increase",
            p_nodes=0.8,
            p_clique=0.7,
            input_magnification=False,
        )
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertTrue(
            (
                graphs.shape[0] == 1
                and graphs.shape[1] == 1
                and graphs.shape[2] == 10
                and graphs.shape[3] == 10
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 11: Generating 1 graph with graph size 400, clique size 150, p_correction_type 'p_increase', input to MLP (input_magnification False)
        graphs, on_off = gen_graphs.generate_batch(1, 400, [150], "p_increase", False)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertTrue(
            (
                graphs.shape[0] == 1
                and graphs.shape[1] == 1
                and graphs.shape[2] == 400
                and graphs.shape[3] == 400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )
        self.assertIsInstance(graphs, torch.Tensor)
        self.assertIsInstance(on_off, list)

        # Test case 12: Generating 1 graph with graph size 400, clique size 150, p_correction_type 'p_increase', input to MLP (input_magnification False)
        graphs, on_off = gen_graphs.generate_batch(1, 400, [150], "p_increase", False)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(on_off), 1)
        self.assertTrue(
            (
                graphs.shape[0] == 1
                and graphs.shape[1] == 1
                and graphs.shape[2] == 400
                and graphs.shape[3] == 400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )

        # Test case 13: Generating 1 graph with graph size 400, clique size 150, p_correction_type 'p_increase', input to CNN (input_magnification True)
        graphs, on_off = gen_graphs.generate_batch(1, 400, [150], "p_increase", True)
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

        # Test case 14: Generating 4 graph with graph size 480, clique size 100, p_correction_type 'p_increase', input to CNN (input_magnification True)
        graphs, on_off = gen_graphs.generate_batch(
            4, 480, [100, 100, 100, 100], "p_increase", True
        )
        self.assertEqual(len(graphs), 4)
        self.assertEqual(len(on_off), 4)
        self.assertTrue(
            (
                graphs.shape[0] == 4
                and graphs.shape[1] == 1
                and graphs.shape[2] == 2400
                and graphs.shape[3] == 2400
            ),
            "The shape of the graphs tensor does not meet the requirements",
        )

    def test_generate_batch_clique_sizes(self):
        # TESTING ERROR MESSAGES FIRST:
        with self.assertRaises(ValueError):
            generate_batch_clique_sizes(
                [3, 2, 1], 100
            )  # allowed_clique_sizes is not a numpy array
        with self.assertRaises(ValueError):
            generate_batch_clique_sizes(
                np.array([3, 2, 1]), "100"
            )  # batch_size is not an integer
        with self.assertRaises(ValueError):
            generate_batch_clique_sizes(
                np.array([1, 2, 3]), 100
            )  # the last provided clique size value is not the smallest one

        # TESTING CORRECT OUTPUTS:
        allowed_clique_sizes = np.array([4, 3, 2, 1])
        batch_size = 100
        result = generate_batch_clique_sizes(allowed_clique_sizes, batch_size)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), batch_size)

        # TESTING THAT FUNCTION WORKS CORRECTLY FOR A SINGLE CLIQUE SIZE VALUE:
        allowed_clique_sizes = np.array([3])
        batch_size = 100
        result = generate_batch_clique_sizes(allowed_clique_sizes, batch_size)
        self.assertIsInstance(result, np.ndarray)

        # TESTING THAT LAST CLIQUE SIZE IS THE MOST FREQUENT WHEN THERE ARE MORE THAN 4 CLIQUE SIZE VALUES:
        allowed_clique_sizes = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
        batch_size = 1000
        result = generate_batch_clique_sizes(allowed_clique_sizes, batch_size)
        unique, counts = np.unique(result, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        print(counts_dict)
        self.assertGreater(
            counts_dict[1],
            max(
                counts_dict[2],
                counts_dict[3],
                counts_dict[4],
                counts_dict[5],
                counts_dict[6],
                counts_dict[7],
                counts_dict[8],
                counts_dict[9],
            ),
        )
