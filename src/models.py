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

        # Defining a scaling factor based on graph size
        scale_factor = (
            graph_size / 224
        )  # Original MLP model was tuned for 224x224 images

        # Dynamically scale the number of neurons in each layer based on graph size
        l1_scaled = int(architecture_specs["l1"] * scale_factor)
        l2_scaled = int(architecture_specs["l2"] * scale_factor)
        l3_scaled = int(architecture_specs["l3"] * scale_factor)
        l4_scaled = int(architecture_specs["l4"] * scale_factor)

        self.model = nn.Sequential(
            # Flatten layer
            nn.Flatten(),
            # First linear layer
            nn.Linear(graph_size * graph_size, l1_scaled),
            nn.BatchNorm1d(l1_scaled),
            nn.ReLU(),
            nn.Dropout(architecture_specs["dropout_prob"]),
            # Second linear layer
            nn.Linear(l1_scaled, l2_scaled),
            nn.BatchNorm1d(l2_scaled),
            nn.ReLU(),
            nn.Dropout(architecture_specs["dropout_prob"]),
            # Third linear layer
            nn.Linear(l2_scaled, l3_scaled),
            nn.BatchNorm1d(l3_scaled),
            nn.ReLU(),
            # Fourth linear layer
            nn.Linear(l3_scaled, l4_scaled),
            nn.BatchNorm1d(l4_scaled),
            nn.ReLU(),
            nn.Dropout(architecture_specs["dropout_prob"]),
            # Output layer
            nn.Linear(l4_scaled, 1),
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
        # print("CNN model_output: ", model_output.shape)

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
