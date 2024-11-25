import unittest
import torch
import os
import numpy as np
import scipy.special as special
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import load_model
from src.utils import load_config
import src.graphs_generation as gen_graphs
from src.variance_test import Variance_algo
from src.distributed_data_parallel import setup_ddp, cleanup_ddp

# TO RUN TESTS FROM HOME FOLDER (DDP initialization?)
# python -m unittest discover -s test -p "test_*.py"

# Load experiment configuration file
configfile_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "docs",
    "grid_exp_config.yml",  # CHANGE THIS TO TEST DIFFERENT CONFIGURATIONS
)
configfile = load_config(configfile_path)

# Define a graph size for the test (choosing the last value, which is the largest graph size)
graph_size = configfile["graph_size_values"][-9]
print("Performing tests for graph size = ", graph_size)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ModelPredictionTest(unittest.TestCase):

    def generate_and_predict(self, model, model_name):
        """Helper function to generate graphs, make predictions, and run assertions on output."""
        clique_size = int(
            graph_size
            * (configfile["training_parameters"]["max_clique_size_proportion"])
        )
        # Defining if magnification is needed for current model
        input_magnification = True if "CNN" in model_name else False
        # Generate graphs:
        graphs = gen_graphs.generate_batch(
            2,
            graph_size,
            [clique_size, clique_size],
            configfile["p_correction_type"],
            input_magnification,
        )[0].to(device)
        # Make predictions
        prediction = model(graphs)
        # Assertions for output size and value range
        self.assertEqual(
            prediction.squeeze().size(),
            torch.Size([2]),
            f"Output size mismatch for model {model_name}",
        )
        self.assertTrue(
            torch.all(prediction >= 0) and torch.all(prediction <= 1),
            f"Prediction values out of range for model {model_name}",
        )
        # Clear memory
        del graphs

    def test_models_prediction(self):
        for i, model_specs in enumerate(configfile["models"]):
            model_name = model_specs["model_name"]
            print(f"Testing prediction for model: {model_name}")
            # Load model based on its index in configfile["models"]
            model = load_model(model_specs, graph_size, device)
            model.eval()
            # Check model type and run corresponding assertions
            if "MLP" in model_name:
                self.generate_and_predict(model, model_name)
            elif "CNN" in model_name:
                self.generate_and_predict(model, model_name)
            elif "ViTscratch" in model_name:
                self.generate_and_predict(model, model_name)
                # Check that all layers are trainable
                for name, param in model.named_parameters():
                    self.assertTrue(param.requires_grad)
            elif "ViTpretrained" in model_name:
                # Additional check for the pretrained model layers
                for name, param in model.named_parameters():
                    if any(key in name for key in ["cls_token", "embed", "head"]):
                        self.assertTrue(param.requires_grad)
                    else:
                        self.assertFalse(param.requires_grad)
                self.generate_and_predict(model, model_name)
            else:
                print(f"Warning: Model type {model_name} not recognized, skipping test")

            # Clear memory
            del model
            torch.cuda.empty_cache()


class ModelMemoryTest(unittest.TestCase):

    @unittest.skipIf(
        not torch.cuda.is_available(), "CUDA not available, skipping CUDA memory tests"
    )
    def check_trainability(
        self,
        model,
        model_name,
        rank,
        world_size,
        batch_size=configfile["training_parameters"]["num_train"],
    ):
        """Test forward and backward pass on a given model for memory issues with DDP."""
        print(f"Testing trainability for model: {model_name} on rank: {rank}")
        clique_size = int(
            graph_size
            * (configfile["training_parameters"]["max_clique_size_proportion"])
        )
        # Defining if magnification is needed for current model
        input_magnification = True if "CNN" in model_name else False
        
        # Adjust batch size for DDP
        local_batch_size = batch_size // world_size
        
        # Optimizer and criterion setup
        optimizer = self.get_optimizer(model)
        criterion = self.get_loss_function()

        # Generate graphs for each local GPU
        train = gen_graphs.generate_batch(
            local_batch_size,
            graph_size,
            [clique_size] * local_batch_size,
            configfile["p_correction_type"],
            input_magnification,
        )
        
        # Run forward and backward pass
        try:
            model.train()
            train_pred = model(train[0].to(rank))
            train_pred = train_pred.squeeze()  # remove extra dimension
            train_loss = criterion(
                train_pred.type(torch.float).to(rank),
                torch.Tensor(train[1])
                .type(torch.float)
                .to(rank),  # labels should be float for BCELoss
            )
            # Backward pass
            train_loss.backward()
            # Update weights
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.fail(f"CUDA out of memory encountered for model {model_name} on rank: {rank}")
            else:
                raise e

        # Clear memory
        del train
        torch.cuda.empty_cache()

        # If DDP is being used, clean up the DDP process group
        if world_size > 1:
            torch.distributed.destroy_process_group()

    def get_optimizer(self, model):
        """Returns the optimizer based on the configuration"""
        optimizer_type = configfile["training_parameters"]["optimizer"]
        lr = configfile["training_parameters"]["learning_rate"]
        if optimizer_type == "Adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Optimizer not found")

    def get_loss_function(self):
        """Returns the loss function based on the configuration"""
        loss_function_type = configfile["training_parameters"]["loss_function"]
        if loss_function_type == "BCELoss":
            return torch.nn.BCELoss()
        elif loss_function_type == "CrossEntropyLoss":
            return torch.nn.CrossEntropyLoss()
        elif loss_function_type == "MSELoss":
            return torch.nn.MSELoss()
        else:
            raise ValueError("Loss function not found")

# DDP setup: get environment variables
world_size = torch.cuda.device_count()  # Total number of GPUs available

# Dynamically add tests for each model in `configfile["models"]`
for idx, model_specs in enumerate(configfile["models"]):
    def generate_test(model_index, model_specifications):
        def test_func(self):
            # Get rank and world size
            rank = int(os.environ["LOCAL_RANK"])
            world_size = torch.cuda.device_count()

            # DDP setup
            if world_size > 1:
                rank = int(os.environ['RANK'])  # Global rank
                local_rank = int(os.environ['LOCAL_RANK'])  # Local rank
                device = torch.device(f"cuda:{local_rank}")
                setup_ddp(rank, world_size)
            else:
                device = torch.device("cuda")

            # Load the model
            model = load_model(model_specifications, graph_size, device, rank)

            # Run trainability test
            self.check_trainability(model, model_specifications["model_name"], rank, world_size)

            # Clear memory and DDP cleanup
            del model
            torch.cuda.empty_cache()

            if world_size > 1:
                torch.distributed.destroy_process_group()

        return test_func

    # Create a unique test name
    test_name = f"test_{model_specs['model_name']}_trainability"
    # Attach the dynamically created test to `ModelMemoryTest`
    setattr(ModelMemoryTest, test_name, generate_test(idx, model_specs))



class VarianceAlgoTest(unittest.TestCase):

    def setUp(self):
        # Set up a valid configuration file and graph size for testing
        self.config_file = {"p_correction_type": "p_reduce"}
        self.graph_size = graph_size
        self.p0 = 0.5
        self.variance_algo = Variance_algo(self.config_file, self.graph_size)

    def test_initialization(self):
        # Test valid initialization
        self.assertEqual(self.variance_algo.graph_size, self.graph_size)
        self.assertEqual(self.variance_algo.p0, 0.5)

        # Test invalid initialization
        with self.assertRaises(ValueError):
            invalid_config = {"p_correction_type": "p_increase"}
            Variance_algo(invalid_config, self.graph_size)

    def test_calculate_fraction_correct(self):
        # Test the calculate_fraction_correct method with a known value
        clique_size = 10
        q_val = clique_size / self.graph_size
        z_val = (q_val**2 / (1 - q_val**2)) * ((1 - self.p0) / self.p0)
        fraction_correct = self.variance_algo.calculate_fraction_correct(clique_size)
        expected_fraction_correct = 0.5 + 0.5 * (
            special.erf(np.sqrt(np.log(1 / (1 - z_val)) / z_val))
            - special.erf(np.sqrt((1 - z_val) / z_val * np.log((1 / (1 - z_val)))))
        )
        self.assertAlmostEqual(fraction_correct, expected_fraction_correct, places=5)

    def test_find_k0(self):
        # Test the find_k0 method
        k0 = self.variance_algo.find_k0()
        self.assertIsInstance(k0, int)
        self.assertGreater(k0, 0)

    def test_save_k0(self):
        # Test the save_k0 method
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        self.variance_algo.save_k0(results_dir)
        file_path = os.path.join(
            results_dir, f"Variance_test_N{self.graph_size}_K0.csv"
        )
        self.assertTrue(os.path.exists(file_path))

        # Clean up
        os.remove(file_path)
        os.rmdir(results_dir)
