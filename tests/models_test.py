import unittest
import torch
from src.utils import load_model
from src.utils import load_config
from src.models import (
    MLP,
    CNN,
    VGG16_scratch,
    VGG16_pretrained,
    ResNet50_scratch,
    ResNet50_pretrained,
    GoogLeNet_scratch,
    GoogLeNet_pretrained,
    ViT_scratch,
    ViT_pretrained,
)
import src.graphs_generation as gen_graphs

# loading experiment configuration file of single experiment and grid experiment:
grid_config = load_config("docs\grid_exp_config.yml")

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTest(unittest.TestCase):

    # for each model:
    # - first testing correspondence between single model experiment file and corresponding section of "single" experiment;
    # - then testing that model predictions are either 0 or 1

    # MLP:
    def test_MLP_predictions(self):

        # loading experiment configuration file of MLP experiment:
        MLP_config = load_config("docs\MLP_exp_config.yml")
        # checking correspondence of p correction type:
        self.assertEqual(
            grid_config["p_correction_type"], MLP_config["p_correction_type"]
        )
        # checking correspondence of training parameters:
        self.assertEqual(
            grid_config["training_parameters"], MLP_config["training_parameters"]
        )
        # checking correspondence of testing parameters:
        self.assertEqual(
            grid_config["testing_parameters"], MLP_config["testing_parameters"]
        )

        # checking that the MLP architecture in the grid experiment configuration file is the same as the one in the MLP experiment configuration file:
        self.assertEqual(
            grid_config["models"][0]["architecture"],
            MLP_config["models"][0]["architecture"],
        )

        # loading model
        model = load_model(MLP_config["models"][0], MLP_config["graph_size"])

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            MLP_config["graph_size"]
            * (MLP_config["training_parameters"]["min_clique_size_proportion"])
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

        # checking that the output is one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

    # CNN:
    def test_CNN_predictions(self):

        # loading experiment configuration file of CNN experiment:
        CNN_config = load_config("docs\CNN_exp_config.yml")
        # checking correspondence of p correction type:
        self.assertEqual(
            grid_config["p_correction_type"], CNN_config["p_correction_type"]
        )

        # checking correspondence of training parameters:
        self.assertEqual(
            grid_config["training_parameters"], CNN_config["training_parameters"]
        )
        # checking correspondence of testing parameters:
        self.assertEqual(
            grid_config["testing_parameters"], CNN_config["testing_parameters"]
        )

        # loading models
        small_model = load_model(CNN_config["models"][0], CNN_config["graph_size"])
        large_model = load_model(CNN_config["models"][1], CNN_config["graph_size"])
        medium_model = load_model(CNN_config["models"][2], CNN_config["graph_size"])

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            CNN_config["graph_size"]
            * (CNN_config["training_parameters"]["min_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_graphs(
            2,
            CNN_config["graph_size"],
            clique_size,
            CNN_config["p_correction_type"],
            True,
        )[0]

        # generating two graphs and predicting with each model:
        # - small model
        prediction_small = small_model(graphs.to(device))
        # - large model
        prediction_large = large_model(graphs.to(device))
        # - medium model
        prediction_medium = medium_model(graphs.to(device))

        # checking that the all outputs are one-dimensional (and have two elements) after squeezing:
        self.assertEqual(prediction_small.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_large.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_medium.squeeze().size(), torch.Size([2]))

        # checking that all predictions are between 0 and 1:
        # - small model
        self.assertTrue(torch.all(prediction_small >= 0))
        self.assertTrue(torch.all(prediction_small <= 1))
        # - large model
        self.assertTrue(torch.all(prediction_large >= 0))
        self.assertTrue(torch.all(prediction_large <= 1))
        # - medium model
        self.assertTrue(torch.all(prediction_medium >= 0))
        self.assertTrue(torch.all(prediction_medium <= 1))

    # VGG:
    def test_VGG_predictions(self):

        # loading experiment configuration file of VGG experiment:
        VGG_config = load_config("docs\VGG_exp_config.yml")
        # checking correspondence of p correction type:
        self.assertEqual(
            grid_config["p_correction_type"], VGG_config["p_correction_type"]
        )

        # checking correspondence of training parameters:
        self.assertEqual(
            grid_config["training_parameters"], VGG_config["training_parameters"]
        )
        # checking correspondence of testing parameters:
        self.assertEqual(
            grid_config["testing_parameters"], VGG_config["testing_parameters"]
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            VGG_config["graph_size"]
            * (VGG_config["training_parameters"]["min_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_graphs(
            2,
            VGG_config["graph_size"],
            clique_size,
            VGG_config["p_correction_type"],
            False,  # SHOULD BE TRUE (when running test on laptop, set to False to avoid memory error)
        )[0]

        # SCRATCH MODEL:
        print("testing VGG16_scratch")
        model = load_model(VGG_config["models"][0], VGG_config["graph_size"])
        model.eval()
        # generating two graphs and predicting
        prediction = model(graphs.to(device))

        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print("ok")
        print("-------------------")

        # PRETRAINED MODEL:
        print("testing VGG16_pretrained")
        model = load_model(VGG_config["models"][1], VGG_config["graph_size"])
        model.eval()
        # making sure that requires_grad is True in pretrained model only in first and last layer
        for name, param in model.named_parameters():
            if "model.classifier" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)

        prediction = model(graphs.to(device))

        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print("ok")

    # RESNET:
    def test_RESNET_predictions(self):

        # loading experiment configuration file of RESNET experiment:
        RESNET_config = load_config("docs\RESNET_exp_config.yml")
        # checking correspondence of p correction type:
        self.assertEqual(
            grid_config["p_correction_type"], RESNET_config["p_correction_type"]
        )

        # checking correspondence of training parameters:
        self.assertEqual(
            grid_config["training_parameters"], RESNET_config["training_parameters"]
        )
        # checking correspondence of testing parameters:
        self.assertEqual(
            grid_config["testing_parameters"], RESNET_config["testing_parameters"]
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            RESNET_config["graph_size"]
            * (RESNET_config["training_parameters"]["min_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_graphs(
            2,
            RESNET_config["graph_size"],
            clique_size,
            RESNET_config["p_correction_type"],
            False,  # SHOULD BE TRUE (when running test on laptop, set to False to avoid memory error)
        )[0]

        # SCRATCH MODEL:
        print("testing RESNET50_scratch")
        model = load_model(RESNET_config["models"][0], RESNET_config["graph_size"])
        model.eval()
        # generating two graphs and predicting
        prediction = model(graphs.to(device))
        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print("ok")
        print("-------------------")

        # PRETRAINED MODEL:
        print("testing RESNET50_pretrained")
        model = load_model(RESNET_config["models"][1], RESNET_config["graph_size"])
        model.eval()
        # checking that requires_grad is True in pretrained model only in first and last layer
        for name, param in model.named_parameters():
            if "model.fc" in name:
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

    # GOOGLENET (not working, automatically performing input resizing during inference):
    def test_GOOGLENET_predictions(self):

        # loading experiment configuration file of GOOGLENET experiment:
        GOOGLENET_config = load_config("docs\GOOGLENET_exp_config.yml")
        # checking correspondence of p correction type:
        self.assertEqual(
            grid_config["p_correction_type"], GOOGLENET_config["p_correction_type"]
        )

        # checking correspondence of training parameters:
        self.assertEqual(
            grid_config["training_parameters"], GOOGLENET_config["training_parameters"]
        )
        # checking correspondence of testing parameters:
        self.assertEqual(
            grid_config["testing_parameters"], GOOGLENET_config["testing_parameters"]
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            GOOGLENET_config["graph_size"]
            * (GOOGLENET_config["training_parameters"]["min_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_graphs(
            2,
            GOOGLENET_config["graph_size"],
            clique_size,
            GOOGLENET_config["p_correction_type"],
            True,  # SHOULD BE TRUE (when running test on laptop, set to False to avoid memory error)
        )[0]

        # SCRATCH MODEL:
        print("testing GoogLeNet_scratch")
        model = load_model(
            GOOGLENET_config["models"][0], GOOGLENET_config["graph_size"]
        )
        model.eval()

        prediction = model(graphs.to(device))

        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print("ok")
        print("-------------------")

        # PRETRAINED MODEL:
        print("testing GoogLeNet_pretrained")
        model = load_model(
            GOOGLENET_config["models"][1], GOOGLENET_config["graph_size"]
        )
        model.eval()

        # checking that requires_grad is True in pretrained model only in first and last layer
        for name, param in model.named_parameters():
            if "model.fc" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)
        # predicting
        prediction = model(graphs.to(device))
        # checking that the outputs are one-dimensional (and has two elements) after squeezing:
        self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
        # checking that both predictions are between 0 and 1:
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))

        print("ok")

    # VIT:
    def test_VIT_predictions(self):
        # loading experiment configuration file of VIT experiment:
        VIT_config = load_config("docs\VIT_exp_config.yml")
        # checking correspondence of p correction type:
        self.assertEqual(
            grid_config["p_correction_type"], VIT_config["p_correction_type"]
        )

        # checking correspondence of training parameters:
        self.assertEqual(
            grid_config["training_parameters"], VIT_config["training_parameters"]
        )
        # checking correspondence of testing parameters:
        self.assertEqual(
            grid_config["testing_parameters"], VIT_config["testing_parameters"]
        )

        # defining clique size (taking minimum clique size on which model will be trained):
        clique_size = int(
            2400 * (VIT_config["training_parameters"]["min_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_graphs(
            2,
            2400,
            clique_size,
            VIT_config["p_correction_type"],
            True,  # SHOULD BE TRUE (when running test on laptop, set to False to avoid memory error)
        )[0]

        # SCRATCH MODEL:
        print("testing VIT_scratch")
        model = load_model(VIT_config["models"][0], 2400)
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
        model = load_model(VIT_config["models"][1], 2400)
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
