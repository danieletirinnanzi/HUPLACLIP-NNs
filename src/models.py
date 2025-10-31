import torch
import torch.nn as nn
from torch.nn import SyncBatchNorm

# custom import:
from src.input_transforms import find_patch_size

# ViT model from timm library (for flexibility in input size)
from timm.models import create_model


class MLP(nn.Module):
    def __init__(self, graph_size, architecture_specs):
        super().__init__()
        self.graph_size = graph_size
        self.architecture_specs = architecture_specs
        # Define the model architecture dynamically from architecture specs
        layers = [nn.Flatten()]
        layer_sizes = architecture_specs["layers"]
        dropout_prob = architecture_specs["dropout_prob"]
        # Defining input size
        input_size = graph_size * graph_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(SyncBatchNorm(layer_size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
             # Input size for following layer
            input_size = layer_size
        
        # Output layer (assume single output for binary classification)
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        
        # Defining model
        self.model = nn.Sequential(*layers)        

    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
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


# ViT models - documentation: https://github.com/huggingface/pytorch-image-models/blob/e35ea733ab1ee9cc35b29b88bf10fc841421eedf/timm/models/vision_transformer.py
class ViT_scratch(nn.Module):

    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size

        # Define patch size dynamically using find_patch_size function
        patch_size = find_patch_size(graph_size)
        print(f"Using patch size = {patch_size}")

        # Define model architecture with required img_size and patch_size
        self.model = create_model(
            "vit_base_patch16_224",  # ViT base model
            img_size=graph_size,
            patch_size=patch_size,
            in_chans=1,
        )

        # Modify the head for binary classification (single output neuron followed by sigmoid function to have output between 0 and 1)
        self.model.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x, **kwargs):
        return self.model(x)


class ViT_pretrained(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size

        # Consistent patch size, or a function that finds the right balance
        patch_size = find_patch_size(graph_size)
        print(f"Using patch size: {patch_size}")

        # Initialize the model with the required img_size and patch_size
        self.model = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            img_size=graph_size,
            patch_size=patch_size,
            in_chans=1,
        )

        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the head for binary classification (single output neuron followed by sigmoid function to have output between 0 and 1)
        self.model.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        # Set the positional encoding layers and the head to require grad
        # NOTE: Positional encoding layers are made trainable because:
        # (1) the spatial relationships relevant for ImageNet are different from the ones of the current task;
        # (2) timm interpolates the positional embeddings to adapt them to the provided patch/image proportions. By training these parameters we're starting from pre-trained positional embedding and adapting them to the current setting
        for name, param in self.model.named_parameters():
            if any(
                key in name for key in ["cls_token", "embed", "head"]
            ):
                param.requires_grad = True

    def forward(self, x, **kwargs):
        return self.model(x)

# ADDITIONAL MODELS from timm library:
# https://github.com/huggingface/pytorch-image-models/tree/e35ea733ab1ee9cc35b29b88bf10fc841421eedf/timm/models
