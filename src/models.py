import torch.nn as nn
import torchvision.models as models


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
        model = models.vgg16(pretrained=True)

        # Freeze the architecture
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier for binary classification (excluding FC layers)
        model.classifier = nn.Sequential(nn.Linear(25088, 1), nn.Sigmoid())

        return model

    # def transformer():
    #     # importing the transformer model
