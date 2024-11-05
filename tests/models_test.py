import unittest
import torch
import os
from src.utils import load_model
from src.utils import load_config
import src.graphs_generation as gen_graphs

# loading experiment configuration file of single experiment and grid experiment:
grid_doc_path = os.path.join(
    os.path.dirname(__file__), "..", "docs", "grid_exp_config.yml"
)
grid_config = load_config(grid_doc_path)

# defining a graph size between the ones in the grid experiment:
graph_size = grid_config["graph_size_values"][2]

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ModelPredictionTest(unittest.TestCase):

    # MLP:
    def test_MLP_predictions(self):

        # loading model
        model = load_model(grid_config["models"][0], graph_size, device)

        # defining clique size (taking maximum clique size on which model will be trained):
        clique_size = int(
            graph_size
            * (grid_config["training_parameters"]["max_clique_size_proportion"])
        )

        # generating two graphs and predicting
        prediction = model(
            gen_graphs.generate_batch(
                2,
                graph_size,
                [clique_size, clique_size],
                grid_config["p_correction_type"],
                False,
            )[0].to(device)
        )

        # checking that the output is one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

    # CNN:
    def test_CNN_predictions(self):

        # loading models
        small_model_1 = load_model(grid_config["models"][1], graph_size, device)
        small_model_2 = load_model(grid_config["models"][2], graph_size, device)
        large_model_1 = load_model(grid_config["models"][3], graph_size, device)
        large_model_2 = load_model(grid_config["models"][4], graph_size, device)
        medium_model_1 = load_model(grid_config["models"][5], graph_size, device)
        medium_model_2 = load_model(grid_config["models"][6], graph_size, device)

        # defining clique size (taking maximum clique size on which model will be trained):
        clique_size = int(
            graph_size
            * (grid_config["training_parameters"]["max_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_batch(
            2,
            graph_size,
            [clique_size, clique_size],
            grid_config["p_correction_type"],
            True,
        )[0]

        # generating two graphs and predicting with each model:
        # - small model
        prediction_small_1 = small_model_1(graphs.to(device))
        prediction_small_2 = small_model_2(graphs.to(device))
        # - large model
        prediction_large_1 = large_model_1(graphs.to(device))
        prediction_large_2 = large_model_2(graphs.to(device))
        # - medium model
        prediction_medium_1 = medium_model_1(graphs.to(device))
        prediction_medium_2 = medium_model_2(graphs.to(device))

        # checking that the all outputs are one-dimensional (and have two elements) after squeezing:
        self.assertEqual(prediction_small_1.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_small_2.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_large_1.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_large_2.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_medium_1.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_medium_2.squeeze().size(), torch.Size([2]))

        # checking that all predictions are between 0 and 1:
        # - small model 1
        self.assertTrue(torch.all(prediction_small_1 >= 0))
        self.assertTrue(torch.all(prediction_small_1 <= 1))
        # - small model 2
        self.assertTrue(torch.all(prediction_small_2 >= 0))
        self.assertTrue(torch.all(prediction_small_2 <= 1))
        # - medium model 1
        self.assertTrue(torch.all(prediction_medium_1 >= 0))
        self.assertTrue(torch.all(prediction_medium_1 <= 1))
        # - medium model 2
        self.assertTrue(torch.all(prediction_medium_2 >= 0))
        self.assertTrue(torch.all(prediction_medium_2 <= 1))
        # - large model 1
        self.assertTrue(torch.all(prediction_large_1 >= 0))
        self.assertTrue(torch.all(prediction_large_1 <= 1))
        # - large model 2
        self.assertTrue(torch.all(prediction_large_2 >= 0))
        self.assertTrue(torch.all(prediction_large_2 <= 1))
 
    # VIT:
    def test_VIT_predictions(self):

        # defining clique size (taking maximum clique size on which model will be trained):
        clique_size = int(
            graph_size
            * (grid_config["training_parameters"]["max_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_batch(
            2,
            graph_size,
            [clique_size, clique_size],
            grid_config["p_correction_type"],
            False,
        )[0]

        # SCRATCH MODEL:
        print("testing VIT_scratch")
        model = load_model(grid_config["models"][7], graph_size, device)
        model.eval()

        print(model)

        # generating two graphs and predicting
        prediction = model(graphs.to(device))
        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print(prediction)

        print("ok")
        print("-------------------")

        # PRETRAINED MODEL:
        print("testing VIT_pretrained")
        model = load_model(grid_config["models"][8], graph_size, device)
        model.eval()
        # checking that requires_grad is True in pretrained model only in first and last layer
        for name, param in model.named_parameters():
            if "model.head" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)
        # generating two graphs and predicting
        prediction = model(graphs.to(device))

        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print("ok")


class ModelMemoryTest(unittest.TestCase):

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping CUDA memory tests")
    def check_trainability(self, model, batch_size = grid_config["training_parameters"]["num_train"]):
        """Helper function to test forward and backward pass on a given model for memory issues."""
        # Define maximum clique size for the graph
        clique_size = int(
            graph_size
            * (grid_config["training_parameters"]["max_clique_size_proportion"])
        )
        # Generate a batch of graphs
        graphs = gen_graphs.generate_batch(
            batch_size, graph_size, [clique_size] * batch_size,
            grid_config["p_correction_type"], True
        )[0].to(device)

        model.train()  # Set model to training mode
        optimizer = torch.optim.Adam(model.parameters(), lr=grid_config["training_parameters"]["learning_rate"])
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # - OPTIMIZER:        
        if grid_config["training_parameters"]["optimizer"] == "Adam":
            optim = torch.optim.Adam(
                model.parameters(), lr=grid_config["training_parameters"]["learning_rate"]
            )
        elif grid_config["training_parameters"]["optimizer"] == "AdamW":
            optim = torch.optim.AdamW(
                model.parameters(), lr=grid_config["training_parameters"]["learning_rate"]
            )
        elif grid_config["training_parameters"]["optimizer"] == "SGD":
            optim = torch.optim.SGD(
                model.parameters(),
                lr=grid_config["training_parameters"]["learning_rate"],
                momentum=0.9,  # default value is zero
            )
        else:
            raise ValueError("Optimizer not found")

        # - LOSS FUNCTION:
        if grid_config["training_parameters"]["loss_function"] == "BCELoss":
            criterion = torch.nn.BCELoss()
        elif grid_config["training_parameters"]["loss_function"] == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss()
        elif grid_config["training_parameters"]["loss_function"] == "MSELoss":
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Loss function not found")    
            

        try:
            # Perform a forward pass
            predictions = model(graphs)
            loss = criterion(predictions.squeeze(), torch.ones(batch_size, device=device))

            # Perform a backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except RuntimeError as e:
            # Capture CUDA out of memory errors
            if 'out of memory' in str(e):
                self.fail(f"CUDA out of memory encountered for model {model}")
            else:
                raise e

    def test_MLP_trainability(self):
        model = load_model(grid_config["models"][0], graph_size, device)
        self.check_trainability(model)

    def test_CNN_small_1_trainability(self):
        model = load_model(grid_config["models"][1], graph_size, device)
        self.check_trainability(model)
        
    def test_CNN_small_2_trainability(self):
        model = load_model(grid_config["models"][2], graph_size, device)
        self.check_trainability(model)

    def test_CNN_large_1_trainability(self):
        model = load_model(grid_config["models"][3], graph_size, device)
        self.check_trainability(model)
        
    def test_CNN_large_2_trainability(self):
        model = load_model(grid_config["models"][4], graph_size, device)
        self.check_trainability(model)       
        
    def test_CNN_medium_1_trainability(self):
        model = load_model(grid_config["models"][5], graph_size, device)
        self.check_trainability(model)
        
    def test_CNN_medium_2_trainability(self):
        model = load_model(grid_config["models"][6], graph_size, device)
        self.check_trainability(model)                

    def test_VIT_scratch_trainability(self):
        model = load_model(grid_config["models"][7], graph_size, device)
        self.check_trainability(model)

    def test_VIT_pretrained_trainability(self):
        model = load_model(grid_config["models"][8], graph_size, device)
        self.check_trainability(model)