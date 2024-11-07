import unittest
import torch
import os
from src.utils import load_model
from src.utils import load_config
import src.graphs_generation as gen_graphs


# Load experiment configuration file
configfile_path = os.path.join(
    os.path.dirname(__file__), "..", "docs", "cnn_exp_config.yml"   #CHANGE THIS TO TEST DIFFERENT CONFIGURATIONS
)
configfile = load_config(configfile_path)

# Define a graph size for the test (choosing the last value, which is the largest graph size)
graph_size = configfile["graph_size_values"][-1]
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
            torch.cuda.empty_cache()


class ModelMemoryTest(unittest.TestCase):

    @unittest.skipIf(
        not torch.cuda.is_available(), "CUDA not available, skipping CUDA memory tests"
    )
    def check_trainability(
        self, model, model_name, batch_size=configfile["training_parameters"]["num_train"]
    ):
        """Helper function to test forward and backward pass on a given model for memory issues."""
        print(f"Testing trainability for model: {model_name}")        
        clique_size = int(
            graph_size
            * (configfile["training_parameters"]["max_clique_size_proportion"])
        )
        # Defining if magnification is needed for current model
        input_magnification = True if "CNN" in model_name else False
        # Run forward and backward pass
        model.train()
        # Reading optimizer and learning rate
        if configfile["training_parameters"]["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=configfile["training_parameters"]["learning_rate"],
            )
        elif configfile["training_parameters"]["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=configfile["training_parameters"]["learning_rate"],
            )
        elif configfile["training_parameters"]["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=configfile["training_parameters"]["learning_rate"],
                momentum=0.9,  # default value is zero
            )
        else:
            raise ValueError("Optimizer not found")
        # Reading loss function
        if configfile["training_parameters"]["loss_function"] == "BCELoss":
            criterion = torch.nn.BCELoss()
        elif configfile["training_parameters"]["loss_function"] == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss()
        elif configfile["training_parameters"]["loss_function"] == "MSELoss":
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Loss function not found")
        # Generate graphs:
        train = gen_graphs.generate_batch(
            batch_size,
            graph_size,
            [clique_size] * batch_size,
            configfile["p_correction_type"],
            input_magnification,
        )                
        # Run forward and backward pass
        try:
            train_pred = model(train[0].to(device))
            train_pred = train_pred.squeeze()  # remove extra dimension            
            loss = criterion(
                train_pred.squeeze(), torch.ones(batch_size, device=device)
            )
            train_loss = criterion(
                train_pred.type(torch.float).to(device),
                torch.Tensor(train[1])
                .type(torch.float)
                .to(device),  # labels should be float for BCELoss
            )
            # Backward pass
            train_loss.backward()
            # Update weights
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)            

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.fail(f"CUDA out of memory encountered for model {model_name}")
            else:
                raise e

        # Clear memory
        del train
        torch.cuda.empty_cache()


# Dynamically add tests for each model in `configfile["models"]`
for idx, model_specs in enumerate(configfile["models"]):

    def generate_test(model_index, model_specifications):
        def test_func(self):
            model = load_model(model_specifications, graph_size, device)
            self.check_trainability(model, model_specifications['model_name'])
            # Clear memory
            del model
            torch.cuda.empty_cache()

        return test_func

    # Create a unique test name
    test_name = f"test_{model_specs['model_name']}_trainability"
    # Attach the dynamically created test to `ModelMemoryTest`
    setattr(ModelMemoryTest, test_name, generate_test(idx, model_specs))
