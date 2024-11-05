import unittest
import torch
import os
from src.utils import load_model
from src.utils import load_config
import src.graphs_generation as gen_graphs


# Load experiment configuration file
grid_doc_path = os.path.join(
    os.path.dirname(__file__), "..", "docs", "grid_exp_config.yml"
)
grid_config = load_config(grid_doc_path)

# Define a graph size for the test (choosing the last value, which is the largest graph size)
graph_size = grid_config["graph_size_values"][-1]

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ModelPredictionTest(unittest.TestCase):

    def generate_and_predict(self, model, model_name):
        """Helper function to generate graphs, make predictions, and run assertions on output."""
        clique_size = int(
            graph_size
            * (grid_config["training_parameters"]["max_clique_size_proportion"])
        )
        # Defining if magnification is needed for current model
        input_magnification = True if "CNN" in model_name else False
        # Generate graphs:
        graphs = gen_graphs.generate_batch(
            2,
            graph_size,
            [clique_size, clique_size],
            grid_config["p_correction_type"],
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

    def test_models(self):
        for i, model_specs in enumerate(grid_config["models"]):
            model_name = model_specs["model_name"]
            print(f"Testing model: {model_name}")
            # Load model based on its index in grid_config["models"]
            model = load_model(model_specs, graph_size, device)
            model.eval()
            # Check model type and run corresponding assertions
            if "MLP" in model_name:
                self.generate_and_predict(model, model_name)
            elif "CNN" in model_name:
                self.generate_and_predict(model, model_name)
            elif "ViTscratch" in model_name:
                self.generate_and_predict(model, model_name)
            elif "ViTpretrained" in model_name:
                # Additional check for the pretrained model layers
                for name, param in model.named_parameters():
                    if "model.head" in name:
                        self.assertTrue(param.requires_grad)
                    else:
                        self.assertFalse(param.requires_grad)
                self.generate_and_predict(model, model_name)
            else:
                print(f"Warning: Model type {model_name} not recognized, skipping test")

            # Clear memory
            del model


class ModelMemoryTest(unittest.TestCase):

    @unittest.skipIf(
        not torch.cuda.is_available(), "CUDA not available, skipping CUDA memory tests"
    )
    def check_trainability(
        self, model, batch_size=grid_config["training_parameters"]["num_train"]
    ):
        """Helper function to test forward and backward pass on a given model for memory issues."""
        clique_size = int(
            graph_size
            * (grid_config["training_parameters"]["max_clique_size_proportion"])
        )
        # Defining if magnification is needed for current model
        input_magnification = True if "CNN" in model_name else False
        # Generate graphs:
        graphs = gen_graphs.generate_batch(
            batch_size,
            graph_size,
            [clique_size] * batch_size,
            grid_config["p_correction_type"],
            input_magnification,
        )[0].to(device)
        # Run forward and backward pass
        model.train()
        # Reading optimizer and learning rate
        if grid_config["training_parameters"]["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=grid_config["training_parameters"]["learning_rate"],
            )
        elif grid_config["training_parameters"]["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=grid_config["training_parameters"]["learning_rate"],
            )
        elif grid_config["training_parameters"]["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=grid_config["training_parameters"]["learning_rate"],
                momentum=0.9,  # default value is zero
            )
        else:
            raise ValueError("Optimizer not found")
        # Reading loss function
        if grid_config["training_parameters"]["loss_function"] == "BCELoss":
            criterion = torch.nn.BCELoss()
        elif grid_config["training_parameters"]["loss_function"] == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss()
        elif grid_config["training_parameters"]["loss_function"] == "MSELoss":
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Loss function not found")
        # Run forward and backward pass
        try:
            predictions = model(graphs)
            loss = criterion(
                predictions.squeeze(), torch.ones(batch_size, device=device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.fail(f"CUDA out of memory encountered for model {model}")
            else:
                raise e

        # Clear memory
        del graphs


# Dynamically add tests for each model in `grid_config["models"]`
for idx, model_specs in enumerate(grid_config["models"]):

    model_name = model_specs["model_name"]

    def generate_test(model_index):
        def test_func(self):
            model = load_model(grid_config["models"][model_index], graph_size, device)
            self.check_trainability(model)
            # Clear memory
            del model

        return test_func

    # Create a unique test name
    test_name = f"test_{model_name}_trainability"
    # Attach the dynamically created test to `ModelMemoryTest`
    setattr(ModelMemoryTest, test_name, generate_test(idx))
