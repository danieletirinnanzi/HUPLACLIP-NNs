import unittest
import torch
from src.utils import load_model
from src.utils import load_config
from src.models import Models
import src.graphs_generation as gen_graphs

# loading experiment configuration file of global experiment:
global_config = load_config("docs\MLP_CNN_VGG_experiment_configuration.yml")

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTest(unittest.TestCase):

    # for each model:
    # - first testing correspondence between single model experiment file and global experiment;
    # - then testing that model predictions are either 0 or 1

    # MLP:
    def test_MLP_predictions(self):

        # loading experiment configuration file of MLP experiment:
        MLP_config = load_config("docs\MLP_experiment_configuration.yml")
        # checking correspondence of graph size:
        self.assertEqual(global_config["graph_size"], MLP_config["graph_size"])
        # checking correspondence of p correction type:
        self.assertEqual(
            global_config["p_correction_type"], MLP_config["p_correction_type"]
        )

        # checking that the MLP section in the global experiment configuration file is the same as the one in the MLP experiment configuration file:
        self.assertEqual(
            global_config["models"][0]["model_name"],
            MLP_config["models"][0]["model_name"],
        )
        # checking that hyperparameters correspond:
        self.assertEqual(
            global_config["models"][0]["hyperparameters"],
            MLP_config["models"][0]["hyperparameters"],
        )

        # loading model
        model = load_model(
            MLP_config["models"][0]["model_name"],
            MLP_config["graph_size"],
            MLP_config["models"][0]["hyperparameters"],
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            MLP_config["graph_size"]
            * (MLP_config["models"][0]["hyperparameters"]["min_clique_size_proportion"])
        )

        # generating two graphs and predicting
        prediction = model(
            gen_graphs.generate_graphs(
                2,
                MLP_config["graph_size"],
                clique_size,
                MLP_config["p_correction_type"],
                False,
            )[0].to(device)
        )

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

    # CNN:
    def test_CNN_predictions(self):

        # loading experiment configuration file of CNN experiment:
        CNN_config = load_config("docs\CNN_experiment_configuration.yml")
        # checking correspondence of graph size:
        self.assertEqual(global_config["graph_size"], CNN_config["graph_size"])
        # checking correspondence of p correction type:
        self.assertEqual(
            global_config["p_correction_type"], CNN_config["p_correction_type"]
        )

        # checking that the CNN section in the global experiment configuration file is the same as the one in the CNN experiment configuration file:
        self.assertEqual(
            global_config["models"][1]["model_name"],
            CNN_config["models"][0]["model_name"],
        )
        # checking that hyperparameters correspond:
        self.assertEqual(
            global_config["models"][1]["hyperparameters"],
            CNN_config["models"][0]["hyperparameters"],
        )

        # loading model
        model = load_model(
            CNN_config["models"][0]["model_name"],
            CNN_config["graph_size"],
            CNN_config["models"][0]["hyperparameters"],
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            CNN_config["graph_size"]
            * (CNN_config["models"][0]["hyperparameters"]["min_clique_size_proportion"])
        )

        # generating two graphs and predicting
        prediction = model(
            gen_graphs.generate_graphs(
                2,
                CNN_config["graph_size"],
                clique_size,
                CNN_config["p_correction_type"],
                False,
            )[0].to(device)
        )

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

    # VGG:
    def test_VGG_predictions(self):

        # loading experiment configuration file of VGG experiment:
        VGG_config = load_config("docs\VGG_experiment_configuration.yml")
        # checking correspondence of graph size:
        self.assertEqual(global_config["graph_size"], VGG_config["graph_size"])
        # checking correspondence of p correction type:
        self.assertEqual(
            global_config["p_correction_type"], VGG_config["p_correction_type"]
        )

        # checking that the VGG section in the global experiment configuration file is the same as the one in the VGG experiment configuration file:
        self.assertEqual(
            global_config["models"][2]["model_name"],
            VGG_config["models"][0]["model_name"],
        )
        # checking that hyperparameters correspond:
        self.assertEqual(
            global_config["models"][2]["hyperparameters"],
            VGG_config["models"][0]["hyperparameters"],
        )

        # loading model
        model = load_model(
            VGG_config["models"][0]["model_name"],
            VGG_config["graph_size"],
            VGG_config["models"][0]["hyperparameters"],
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            VGG_config["graph_size"]
            * (VGG_config["models"][0]["hyperparameters"]["min_clique_size_proportion"])
        )

        # generating two graphs and predicting
        prediction = model(
            gen_graphs.generate_graphs(
                2,
                VGG_config["graph_size"],
                clique_size,
                VGG_config["p_correction_type"],
                True,
            )[0].to(device)
        )

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))


if __name__ == "__main__":
    unittest.main()
