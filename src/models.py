import torch.nn as nn
import torchvision.models as models
from src.input_transforms import find_patch_size


class Models:

    @staticmethod
    def mlp(graph_size, hyperparameters):

        # Definition of the MLP architecture
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
            # AN ADDITIONAL LAYER MIGHT BE NEEDED?
            # output layer
            nn.Linear(hyperparameters["l3"], 1),
            nn.Sigmoid(),
        )

        return model

    @staticmethod
    def cnn(graph_size, hyperparameters):

        # TODO? adding a fourth block only if graph_size is bigger than 500
        # MIGHT BE UNNECESSARY

        # Definition of the CNN architecture
        model = nn.Sequential(
            # dropout
            nn.Dropout(hyperparameters["dropout_prob"]),
            # 1st block
            nn.Conv2d(1, hyperparameters["c1"], hyperparameters["kernel_size"], 1, 1),
            nn.BatchNorm2d(hyperparameters["c1"]),
            nn.ReLU(),
            nn.Dropout(hyperparameters["dropout_prob"]),
            nn.MaxPool2d(hyperparameters["kernel_size"]),
            # 2nd block
            nn.Conv2d(
                hyperparameters["c1"],
                hyperparameters["c2"],
                hyperparameters["kernel_size"],
                1,
                1,
            ),
            nn.BatchNorm2d(hyperparameters["c2"]),
            nn.ReLU(),
            nn.Dropout(hyperparameters["dropout_prob"]),
            nn.MaxPool2d(hyperparameters["kernel_size"]),
            # 3rd block
            nn.Conv2d(
                hyperparameters["c2"],
                hyperparameters["c3"],
                hyperparameters["kernel_size"],
                1,
                1,
            ),
            nn.BatchNorm2d(hyperparameters["c3"]),
            nn.ReLU(),
            nn.Dropout(hyperparameters["dropout_prob"]),
            nn.MaxPool2d(hyperparameters["kernel_size"]),
            # AN ADDITIONAL LAYER MIGHT BE NEEDED?
            # output layer
            nn.Flatten(),
            nn.Linear(hyperparameters["c3"] * 3 * 3, hyperparameters["l3"]),
            nn.ReLU(),
            nn.Linear(hyperparameters["l3"], 1),
            nn.Sigmoid(),
        )

        return model

    @staticmethod
    def vgg16():
        model = models.vgg16(weights="DEFAULT")

        # Freeze the architecture
        for param in model.parameters():
            param.requires_grad = False

        # TODO: ADD INTERMEDIATE LAYER?

        model.classifier = nn.Sequential(nn.Linear(25088, 1), nn.Sigmoid())

        return model

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

    @staticmethod
    def vit_scratch(graph_size):

        patch_size = find_patch_size(graph_size)

        model = models.vit_b_16()  # NOTE: num_classes can be set also from here

        # manually setting patch size and image size:
        model.patch_size = patch_size
        model.image_size = graph_size

        # Modify the classifier for binary classification
        model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        return model

    @staticmethod
    def vit_pretrained():

        # TODO: PATCH SIZE DEFINITION

        model = models.vit_b_16(weights="DEFAULT")

        # Freeze the architecture
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier for binary classification
        model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        return model
