import torch
import torch.nn as nn
import torchvision.models as models
from src.input_transforms import find_patch_size


class Models:

    @staticmethod
    def mlp(graph_size, hyperparameters):

        # MLP
        model = nn.Sequential(
            # flatten operation
            nn.Flatten(),
            # 1st block
            nn.Linear(graph_size * graph_size, hyperparameters["l1"]),
            nn.BatchNorm1d(hyperparameters["l1"]),
            nn.ReLU(),
            nn.Dropout(hyperparameters["dropout_prob"]),
            # 2nd block
            nn.Linear(hyperparameters["l1"], hyperparameters["l2"]),
            nn.BatchNorm1d(hyperparameters["l2"]),
            nn.ReLU(),
            nn.Dropout(hyperparameters["dropout_prob"]),
            # 3rd block
            nn.Linear(hyperparameters["l2"], hyperparameters["l3"]),
            nn.BatchNorm1d(hyperparameters["l3"]),
            nn.ReLU(),
            # output layer
            nn.Linear(hyperparameters["l3"], 1),
            nn.Sigmoid(),
        )

        return model

    @staticmethod
    def cnn(graph_size, hyperparameters):

        def create_conv_block(
            in_channels, out_channels, kernel_size, stride, padding, dropout_prob=0.5
        ):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size),
                nn.Dropout(dropout_prob),
            )

        # Definition of the CNN architecture
        model = nn.Sequential(
            # 1st block
            create_conv_block(
                1,
                hyperparameters["c1"],
                hyperparameters["kernel_size"],
                hyperparameters["stride"],
                hyperparameters["padding"],
                hyperparameters["dropout_prob"],
            ),
            # 2nd block
            create_conv_block(
                hyperparameters["c1"],
                hyperparameters["c2"],
                hyperparameters["kernel_size"],
                hyperparameters["stride"],
                hyperparameters["padding"],
                hyperparameters["dropout_prob"],
            ),
            # 3rd block
            create_conv_block(
                hyperparameters["c2"],
                hyperparameters["c3"],
                hyperparameters["kernel_size"],
                hyperparameters["stride"],
                hyperparameters["padding"],
                hyperparameters["dropout_prob"],
            ),
            # 4th block
            create_conv_block(
                hyperparameters["c3"],
                hyperparameters["c4"],
                hyperparameters["kernel_size"],
                hyperparameters["stride"],
                hyperparameters["padding"],
                hyperparameters["dropout_prob"],
            ),
            # 5th block
            create_conv_block(
                hyperparameters["c4"],
                hyperparameters["c5"],
                hyperparameters["kernel_size"],
                hyperparameters["stride"],
                hyperparameters["padding"],
                hyperparameters["dropout_prob"],
            ),
            # ADDITIONAL BLOCKS (stabilize learning, but now cause features to vanish for N=300). Alternative solutions:
            # - Adaptive pooling;
            # - Global average pooling;
            # - Dilated convolutions;
            # - Skip connections;
            # - Use exact same Rudy's architecture.
            # # 6th block
            # create_conv_block(
            #     hyperparameters["c5"],
            #     hyperparameters["c6"],
            #     hyperparameters["kernel_size"],
            #     hyperparameters["stride"],
            #     hyperparameters["padding"],
            #     hyperparameters["dropout_prob"],
            # ),
            # # 7th block
            # create_conv_block(
            #     hyperparameters["c6"],
            #     hyperparameters["c7"],
            #     hyperparameters["kernel_size"],
            #     hyperparameters["stride"],
            #     hyperparameters["padding"],
            #     hyperparameters["dropout_prob"],
            # ),
        )

        # performing forward pass on random input to get the size of the output tensor (gradients are unused here)
        # NOTE: the proper way of doing this is to use a function that calculates the output size of the CNN, given its structure
        model_output = model(torch.randn(1, 1, graph_size, graph_size))
        model_output_size = model_output.view(-1).size(
            0
        )  # flattening the output tensor

        # adding the output layer
        model.add_module("Flatten", nn.Flatten())
        model.add_module("Linear1", nn.Linear(model_output_size, hyperparameters["l1"]))
        model.add_module("ReLU", nn.ReLU())
        model.add_module("Linear2", nn.Linear(hyperparameters["l1"], 1))
        model.add_module("Sigmoid", nn.Sigmoid())

        return model

    # VGG16
    @staticmethod
    def vgg16():
        model = models.vgg16(weights="DEFAULT")

        # Freeze the architecture
        for param in model.parameters():
            param.requires_grad = False

        # TODO: ADD INTERMEDIATE LAYER?

        model.classifier = nn.Sequential(nn.Linear(25088, 1), nn.Sigmoid())

        return model

    # RESNET50
    @staticmethod
    def resnet50():
        model = models.resnet50(weights="DEFAULT")

        # Freeze the architecture
        for param in model.parameters():
            param.requires_grad = False

        # TODO: ADD INTERMEDIATE LAYER?

        # Modify the classifier for binary classification (excluding FC layers)
        model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

        return model

    # VISUAL TRANSFORMERS:
    # TO FIX: now predictions not working, probably a resizing of the input is required https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    # - ViT from scratch
    @staticmethod
    def vit_scratch(graph_size):

        patch_size, image_size = find_patch_size(graph_size)

        model = models.vit_b_16()  # NOTE: num_classes can be set also from here

        # manually setting patch size and image size:
        model.patch_size = patch_size
        model.image_size = image_size

        # Modify the classifier for binary classification
        model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        return model

    # - ViT pretrained
    @staticmethod
    def vit_pretrained(graph_size):

        patch_size, image_size = find_patch_size(graph_size)

        model = models.vit_b_16(weights="DEFAULT")

        # Freeze the architecture
        for param in model.parameters():
            param.requires_grad = False

        # manually setting patch size and image size:
        model.patch_size = patch_size
        model.image_size = image_size

        # Modify the classifier for binary classification
        model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        return model
