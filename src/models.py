import torch
import torch.nn as nn
import torchvision.models as models
from src.input_transforms import find_patch_size


class MLP(nn.Module):
    def __init__(self, graph_size, architecture_specs):
        super().__init__()
        self.graph_size = graph_size
        self.architecture_specs = architecture_specs

        self.model = nn.Sequential(
            # Flatten layer
            nn.Flatten(),
            # First linear layer
            nn.Linear(graph_size * graph_size, architecture_specs["l1"]),
            nn.BatchNorm1d(architecture_specs["l1"]),
            nn.ReLU(),
            nn.Dropout(architecture_specs["dropout_prob"]),
            # Second linear layer
            nn.Linear(architecture_specs["l1"], architecture_specs["l2"]),
            nn.BatchNorm1d(architecture_specs["l2"]),
            nn.ReLU(),
            nn.Dropout(architecture_specs["dropout_prob"]),
            # Third linear layer
            nn.Linear(architecture_specs["l2"], architecture_specs["l3"]),
            nn.BatchNorm1d(architecture_specs["l3"]),
            nn.ReLU(),
            # Fourth linear layer
            nn.Linear(architecture_specs["l3"], architecture_specs["l4"]),
            nn.BatchNorm1d(architecture_specs["l4"]),
            nn.ReLU(),
            nn.Dropout(architecture_specs["dropout_prob"]),
            # Output layer
            nn.Linear(architecture_specs["l4"], 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, graph_size, architecture_specs):
        super().__init__()
        self.graph_size = graph_size
        self.architecture_specs = architecture_specs

        # dynamically create convolutional layers based on architecture_specs
        conv_layers = []
        for i in range(architecture_specs["num_conv_layers"]):
            in_channels = architecture_specs[f"c{i}"]
            out_channels = architecture_specs[f"c{i+1}"]
            kernel_size_conv = architecture_specs["kernel_size_conv"]
            kernel_size_pool = architecture_specs["kernel_size_pool"]
            stride = architecture_specs["stride"]
            padding = architecture_specs["padding"]
            conv_layers.append(
                self.create_block(
                    in_channels,
                    out_channels,
                    kernel_size_conv,
                    kernel_size_pool,
                    stride,
                    padding,
                )
            )

        self.model = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.calculate_output_size(), architecture_specs["l1"])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(architecture_specs["l1"], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

    def create_block(
        self,
        in_channels,
        out_channels,
        kernel_size_conv,
        kernel_size_pool,
        stride,
        padding,
    ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size_conv, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size_pool),
            nn.Dropout(self.architecture_specs["dropout_prob"]),
        )

    def calculate_output_size(self):

        # performing forward pass on random input to get the size of the feature maps (gradients are unused here)
        model_output = self.model(torch.bernoulli(torch.rand(1, 1, 2400, 2400)))
        # getting the flattened size of the output tensor
        model_output_size = model_output.view(-1).size(0)

        # # UNCOMMENT TO VISUALIZE MODEL OUTPUT SIZE:
        # print("model_output: ", model_output.shape)

        return model_output_size


# 3 CHANNEL INPUT FROM WITHIN THE MODEL https://discuss.pytorch.org/t/transfer-learning-usage-with-different-input-size/20744/4


class VGG16_scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16()
        # Change the classifier
        self.model.classifier = nn.Sequential(nn.Linear(25088, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class VGG16_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(weights="DEFAULT")
        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier
        self.model.classifier = nn.Sequential(nn.Linear(25088, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class ResNet50_scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50()
        # Change the classifier
        self.model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class ResNet50_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier
        self.model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class GoogLeNet_scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.googlenet()
        # Change the classifier
        self.model.fc = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class GoogLeNet_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.googlenet(weights="DEFAULT")
        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier
        self.model.fc = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class ViT_scratch(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size
        patch_size, image_size = find_patch_size(graph_size)
        self.model = models.vit_b_16()
        # manually setting patch size and image size:
        self.model.patch_size = patch_size
        self.model.image_size = image_size
        # Change the head
        self.model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class ViT_pretrained(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size
        patch_size, image_size = find_patch_size(graph_size)
        self.model = models.vit_b_16(weights="DEFAULT")
        # manually setting patch size and image size:
        self.model.patch_size = patch_size
        self.model.image_size = image_size
        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the head
        self.model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
