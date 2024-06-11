import torch
import torch.nn as nn
import torchvision.models as models

# for VIT PRETRAINED (https://github.com/bwconrad/flexivit):
from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from flexivit_pytorch import pi_resize_patch_embed


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


class VGG16_scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16()
        # Change the classifier
        self.model.classifier = nn.Sequential(nn.Linear(25088, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
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
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
        return self.model(x)


class ResNet50_scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50()
        # Change the classifier
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
        return self.model(x)


class ResNet50_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
        return self.model(x)


class GoogLeNet_scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.googlenet(init_weights=True)
        # Change the classifier
        self.model.fc = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
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
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
        return self.model(x)


class ViT_scratch(nn.Module):
    def __init__(self, graph_size, architecture_specs):
        super().__init__()
        self.model = models.VisionTransformer(
            image_size=2400,
            patch_size=architecture_specs["patch_size"],
            num_classes=architecture_specs["num_classes"],
            num_layers=architecture_specs["num_layers"],
            num_heads=architecture_specs["num_heads"],
            mlp_dim=architecture_specs["mlp_dim"],
            hidden_dim=architecture_specs["hidden_dim"],
        )
        # Change the classifier
        self.model.heads = nn.Sequential(
            nn.Linear(architecture_specs["hidden_dim"], 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
        return self.model(x)


# PRETRAINED VIT (https://github.com/bwconrad/flexivit)
class ViT_pretrained(nn.Module):
    def __init__(self, graph_size, architecture_specs):
        super().__init__()

        # Load the pretrained model's state_dict
        state_dict = create_model("vit_base_patch16_224", pretrained=True).state_dict()

        # Resize the patch embedding
        new_patch_size = (
            architecture_specs["patch_size"],
            architecture_specs["patch_size"],
        )
        state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
            patch_embed=state_dict["patch_embed.proj.weight"],
            new_patch_size=new_patch_size,
        )

        # Interpolate the position embedding size
        image_size = 2400
        grid_size = image_size // new_patch_size[0]
        state_dict["pos_embed"] = resample_abs_pos_embed(
            posemb=state_dict["pos_embed"], new_size=[grid_size, grid_size]
        )

        # Load the new weights into a model with the target image and patch sizes
        self.model = create_model(
            "vit_base_patch16_224", img_size=image_size, patch_size=new_patch_size
        )
        self.model.load_state_dict(state_dict, strict=True)

        # Freeze the architecture
        for param in self.model.parameters():
            param.requires_grad = False

        # Change the classifier
        self.model.head = nn.Sequential(
            nn.Linear(architecture_specs["hidden_dim"], 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel 3 times
        return self.model(x)
