import torch
import torch.nn as nn
import torchvision.models as models

# custom import:
from src.input_transforms import find_patch_size

# ViT model from timm library (for flexibility in input size)
from timm.models import create_model


class MLP(nn.Module):
    def __init__(self, graph_size, architecture_specs):
        super().__init__()
        self.graph_size = graph_size
        self.architecture_specs = architecture_specs

        # Define the model architecture
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

        # TO REMOVE:
        # Printing outside dimension input and output shapes for debugging
        print(
            "\tIn Model: input size",
            x.size(),
            "output size",
            self.model(x).size(),
        )
        return self.model(x)


class CNN(nn.Module):

    def __init__(self, graph_size, architecture_specs):
        super().__init__()
        self.graph_size = graph_size
        self.architecture_specs = architecture_specs

        # Dynamically create convolutional layers based on architecture_specs
        conv_layers = []
        for i in range(architecture_specs["num_conv_layers"]):
            in_channels = architecture_specs[f"c{i}"]
            out_channels = architecture_specs[f"c{i+1}"]
            kernel_size_conv = architecture_specs["kernel_size_conv"][i]
            stride_conv = architecture_specs["stride_conv"][i]
            padding_conv = architecture_specs["padding_conv"][i]
            kernel_size_pool = architecture_specs["kernel_size_pool"]
            stride_pool = architecture_specs["stride_pool"]
            dropout_prob = architecture_specs["dropout_prob"]

            # Append each layer block
            conv_layers.append(
                self.create_block(
                    in_channels,
                    out_channels,
                    kernel_size_conv,
                    stride_conv,
                    padding_conv,
                    kernel_size_pool,
                    stride_pool,
                    dropout_prob,
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
        stride_conv,
        padding_conv,
        kernel_size_pool,
        stride_pool,
        dropout_prob,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size_conv, stride_conv, padding_conv
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size_pool, stride_pool),
            nn.Dropout(dropout_prob),
        )

    def calculate_output_size(self):
        model_output = self.model(torch.bernoulli(torch.rand(1, 1, 2400, 2400)))
        model_output_size = model_output.view(-1).size(0)
        # # UNCOMMENT TO VISUALIZE MODEL OUTPUT SIZE:
        # print("CNN model final feature map: ", model_output.shape)
        return model_output_size


# class ViT_scratch(nn.Module):
#     def __init__(self, graph_size):
#         super().__init__()
#         self.graph_size = graph_size
#         self.model = models.vit_b_16()
#         # Change the head
#         self.model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

#     def forward(self, x):
#         x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
#         return self.model(x)


# class ViT_pretrained(nn.Module):
#     def __init__(self, graph_size):
#         super().__init__()
#         self.graph_size = graph_size
#         self.model = models.vit_b_16(weights="DEFAULT")
#         # Freeze the architecture
#         for param in self.model.parameters():
#             param.requires_grad = False
#         # Change the head
#         self.model.heads.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

#     def forward(self, x):
#         x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
#         return self.model(x)


class FlexiViT_scratch(nn.Module):

    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size

        # Define patch size dynamically using find_patch_size function
        patch_size = find_patch_size(graph_size)
        print(f"Using patch size: {patch_size}")

        # Train from scratch without pretrained weights
        self.model = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=graph_size,
            patch_size=patch_size,
        )

        # Modify the head for binary classification
        self.model.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
        return self.model(x)


class FlexiViT_pretrained(nn.Module):

    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size

        # Define patch size dynamically using find_patch_size function
        patch_size = find_patch_size(graph_size)
        print(f"Using patch size: {patch_size}")

        # Use pretrained weights
        self.model = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            img_size=graph_size,
            patch_size=patch_size,
        )

        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the head for binary classification
        self.model.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
        return self.model(x)
