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

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTest(unittest.TestCase):

    # for each model:
    # - first testing correspondence between single model experiment file and corresponding section of "single" experiment;
    # - then testing that model predictions are either 0 or 1

    # MLP:
    def test_MLP_predictions(self):

        # loading experiment configuration file of MLP experiment:
        MLP_config = load_config(
            os.path.join(os.path.dirname(__file__), "..", "docs", "MLP_exp_config.yml")
        )
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
        model = load_model(MLP_config["models"][0], MLP_config["graph_size"], device)

        # defining clique size (taking maximum clique size on which model will be trained):
        clique_size = int(
            MLP_config["graph_size"]
            * (MLP_config["training_parameters"]["max_clique_size_proportion"])
        )

        # generating two graphs and predicting
        prediction = model(
            gen_graphs.generate_batch(
                2,
                MLP_config["graph_size"],
                [clique_size, clique_size],
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
        CNN_config = load_config(
            os.path.join(os.path.dirname(__file__), "..", "docs", "CNN_exp_config.yml")
        )
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
        small_model_1 = load_model(
            CNN_config["models"][0], CNN_config["graph_size"], device
        )
        small_model_2 = load_model(
            CNN_config["models"][1], CNN_config["graph_size"], device
        )
        large_model_1 = load_model(
            CNN_config["models"][2], CNN_config["graph_size"], device
        )
        large_model_2 = load_model(
            CNN_config["models"][3], CNN_config["graph_size"], device
        )
        medium_model_1 = load_model(
            CNN_config["models"][4], CNN_config["graph_size"], device
        )
        medium_model_2 = load_model(
            CNN_config["models"][5], CNN_config["graph_size"], device
        )
        # rudy_model = load_model(
        #     CNN_config["models"][6], CNN_config["graph_size"], device
        # )

        # print(rudy_model)

        # defining clique size (taking maximum clique size on which model will be trained):
        clique_size = int(
            CNN_config["graph_size"]
            * (CNN_config["training_parameters"]["max_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_batch(
            2,
            CNN_config["graph_size"],
            [clique_size, clique_size],
            CNN_config["p_correction_type"],
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
        # - rudy model
        # prediction_rudy = rudy_model(graphs.to(device))

        # checking that the all outputs are one-dimensional (and have two elements) after squeezing:
        self.assertEqual(prediction_small_1.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_small_2.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_large_1.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_large_2.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_medium_1.squeeze().size(), torch.Size([2]))
        self.assertEqual(prediction_medium_2.squeeze().size(), torch.Size([2]))
        # self.assertEqual(prediction_rudy.squeeze().size(), torch.Size([2]))

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
        # - rudy model
        # self.assertTrue(torch.all(prediction_rudy >= 0))
        # self.assertTrue(torch.all(prediction_rudy <= 1))

    # # VGG:
    # def test_VGG_predictions(self):

    #     # loading experiment configuration file of VGG experiment:
    #     VGG_config = load_config(
    #         os.path.join(os.path.dirname(__file__), "..", "docs", "VGG_exp_config.yml")
    #     )
    #     # checking correspondence of p correction type:
    #     self.assertEqual(
    #         grid_config["p_correction_type"], VGG_config["p_correction_type"]
    #     )

    #     # checking correspondence of training parameters:
    #     self.assertEqual(
    #         grid_config["training_parameters"], VGG_config["training_parameters"]
    #     )
    #     # checking correspondence of testing parameters:
    #     self.assertEqual(
    #         grid_config["testing_parameters"], VGG_config["testing_parameters"]
    #     )

    #     # defining clique size (taking maximum clique size on which model will be trained):
    #     clique_size = int(
    #         VGG_config["graph_size"]
    #         * (VGG_config["training_parameters"]["max_clique_size_proportion"])
    #     )

    #     # generating 2 graphs:
    #     graphs = gen_graphs.generate_batch(
    #         2,
    #         VGG_config["graph_size"],
    #         [clique_size, clique_size],
    #         VGG_config["p_correction_type"],
    #     )[0]

    #     # SCRATCH MODEL:
    #     print("testing VGG16_scratch")
    #     model = load_model(VGG_config["models"][0], VGG_config["graph_size"], device)
    #     model.eval()
    #     # generating two graphs and predicting
    #     prediction = model(graphs.to(device))

    #     # checking that the outputs are one-dimensional (and has two elements) after squeezing:
    #     self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

    #     # checking that both predictions are between 0 and 1:
    #     self.assertTrue(torch.all(prediction >= 0))
    #     self.assertTrue(torch.all(prediction <= 1))

    #     print("ok")
    #     print("-------------------")

    #     # PRETRAINED MODEL:
    #     print("testing VGG16_pretrained")
    #     model = load_model(VGG_config["models"][1], VGG_config["graph_size"], device)
    #     model.eval()
    #     # making sure that requires_grad is True in pretrained model only in first and last layer
    #     for name, param in model.named_parameters():
    #         if "model.classifier" in name:
    #             self.assertTrue(param.requires_grad)
    #         else:
    #             self.assertFalse(param.requires_grad)

    #     prediction = model(graphs.to(device))

    #     # checking that the outputs are one-dimensional (and has two elements) after squeezing:
    #     self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

    #     # checking that both predictions are between 0 and 1:
    #     self.assertTrue(torch.all(prediction >= 0))
    #     self.assertTrue(torch.all(prediction <= 1))

    #     print("ok")

    # # RESNET:
    # def test_RESNET_predictions(self):

    #     # loading experiment configuration file of RESNET experiment:
    #     RESNET_config = load_config(
    #         os.path.join(
    #             os.path.dirname(__file__), "..", "docs", "RESNET_exp_config.yml"
    #         )
    #     )
    #     # checking correspondence of p correction type:
    #     self.assertEqual(
    #         grid_config["p_correction_type"], RESNET_config["p_correction_type"]
    #     )

    #     # checking correspondence of training parameters:
    #     self.assertEqual(
    #         grid_config["training_parameters"], RESNET_config["training_parameters"]
    #     )
    #     # checking correspondence of testing parameters:
    #     self.assertEqual(
    #         grid_config["testing_parameters"], RESNET_config["testing_parameters"]
    #     )

    #     # defining clique size (taking maximum clique size on which model will be trained):
    #     clique_size = int(
    #         RESNET_config["graph_size"]
    #         * (RESNET_config["training_parameters"]["max_clique_size_proportion"])
    #     )

    #     # generating 2 graphs:
    #     graphs = gen_graphs.generate_batch(
    #         2,
    #         RESNET_config["graph_size"],
    #         [clique_size, clique_size],
    #         RESNET_config["p_correction_type"],
    #     )[0]

    #     # SCRATCH MODEL:
    #     print("testing RESNET50_scratch")
    #     model = load_model(
    #         RESNET_config["models"][0], RESNET_config["graph_size"], device
    #     )
    #     model.eval()
    #     # generating two graphs and predicting
    #     prediction = model(graphs.to(device))
    #     # checking that the outputs are one-dimensional (and has two elements) after squeezing:
    #     self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
    #     # checking that both predictions are between 0 and 1:
    #     self.assertTrue(torch.all(prediction >= 0))
    #     self.assertTrue(torch.all(prediction <= 1))

    #     print("ok")
    #     print("-------------------")

    #     # PRETRAINED MODEL:
    #     print("testing RESNET50_pretrained")
    #     model = load_model(
    #         RESNET_config["models"][1], RESNET_config["graph_size"], device
    #     )
    #     model.eval()
    #     # checking that requires_grad is True in pretrained model only in first and last layer
    #     for name, param in model.named_parameters():
    #         if "model.fc" in name:
    #             self.assertTrue(param.requires_grad)
    #         else:
    #             self.assertFalse(param.requires_grad)
    #     # generating two graphs and predicting
    #     prediction = model(graphs.to(device))

    #     # checking that the outputs are one-dimensional (and has two elements) after squeezing:
    #     self.assertEqual(prediction.squeeze().size(), torch.Size([2]))

    #     # checking that both predictions are between 0 and 1:
    #     self.assertTrue(torch.all(prediction >= 0))
    #     self.assertTrue(torch.all(prediction <= 1))

    #     print("ok")

    # # GOOGLENET (not working, automatically performing input resizing during inference):
    # def test_GOOGLENET_predictions(self):

    #     # loading experiment configuration file of GOOGLENET experiment:
    #     GOOGLENET_config = load_config(
    #         os.path.join(
    #             os.path.dirname(__file__), "..", "docs", "GOOGLENET_exp_config.yml"
    #         )
    #     )
    #     # checking correspondence of p correction type:
    #     self.assertEqual(
    #         grid_config["p_correction_type"], GOOGLENET_config["p_correction_type"]
    #     )

    #     # checking correspondence of training parameters:
    #     self.assertEqual(
    #         grid_config["training_parameters"], GOOGLENET_config["training_parameters"]
    #     )
    #     # checking correspondence of testing parameters:
    #     self.assertEqual(
    #         grid_config["testing_parameters"], GOOGLENET_config["testing_parameters"]
    #     )

    #     # defining clique size (taking maximum clique size on which model will be trained):
    #     clique_size = int(
    #         GOOGLENET_config["graph_size"]
    #         * (GOOGLENET_config["training_parameters"]["max_clique_size_proportion"])
    #     )

    #     # generating 2 graphs:
    #     graphs = gen_graphs.generate_batch(
    #         2,
    #         GOOGLENET_config["graph_size"],
    #         [clique_size, clique_size],
    #         GOOGLENET_config["p_correction_type"],
    #     )[0]

    #     # SCRATCH MODEL:
    #     print("testing GoogLeNet_scratch")
    #     model = load_model(
    #         GOOGLENET_config["models"][0], GOOGLENET_config["graph_size"], device
    #     )

    #     # NOTE: only in scratch model, prediction.logits is needed when in train mode
    #     # TRAIN:
    #     model.train()
    #     prediction = model(graphs.to(device)).logits
    #     # # TEST
    #     # model.eval()
    #     # prediction = model(graphs.to(device))

    #     # checking that the outputs are one-dimensional (and has two elements) after squeezing:
    #     self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
    #     # checking that both predictions are between 0 and 1:
    #     self.assertTrue(torch.all(prediction >= 0))
    #     self.assertTrue(torch.all(prediction <= 1))

    #     print("ok")
    #     print("-------------------")

    #     # PRETRAINED MODEL:
    #     print("testing GoogLeNet_pretrained")
    #     model = load_model(
    #         GOOGLENET_config["models"][1], GOOGLENET_config["graph_size"], device
    #     )
    #     # model.eval()
    #     model.train()

    #     # checking that requires_grad is True in pretrained model only in first and last layer
    #     for name, param in model.named_parameters():
    #         if "model.fc" in name:
    #             self.assertTrue(param.requires_grad)
    #         else:
    #             self.assertFalse(param.requires_grad)
    #     # predicting
    #     prediction = model(graphs.to(device))
    #     # checking that the outputs are one-dimensional (and has two elements) after squeezing:
    #     self.assertEqual(prediction.squeeze().size(), torch.Size([2]))
    #     # checking that both predictions are between 0 and 1:
    #     self.assertTrue(torch.all(prediction >= 0))
    #     self.assertTrue(torch.all(prediction <= 1))

    #     print("ok")

    # VIT:
    def test_VIT_predictions(self):
        # loading experiment configuration file of VIT experiment:
        VIT_config = load_config(
            os.path.join(os.path.dirname(__file__), "..", "docs", "VIT_exp_config.yml")
        )
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

        # defining clique size (taking maximum clique size on which model will be trained):
        clique_size = int(
            VIT_config["graph_size"]
            * (VIT_config["training_parameters"]["max_clique_size_proportion"])
        )

        # generating 2 graphs:
        graphs = gen_graphs.generate_batch(
            2,
            VIT_config["graph_size"],
            [clique_size, clique_size],
            VIT_config["p_correction_type"],
        )[0]

        # SCRATCH MODEL:
        print("testing VIT_scratch")
        model = load_model(VIT_config["models"][0], VIT_config["graph_size"], device)
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
        model = load_model(VIT_config["models"][1], VIT_config["graph_size"], device)
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
