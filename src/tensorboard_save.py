import torch
import torchvision
import numpy as np
import torch.nn as nn
import src.graphs_generation as gen_graphs


### SAVING IMAGES


def tensorboard_save_images(
    writer, graph_size, p_correction_type, num_images=10, vgg_input=False
):

    # Set clique size to 60% of the graph size:
    clique_size = int(0.6 * graph_size)
    # Generate random graphs
    graph_pool = gen_graphs.generate_graphs(
        num_images,
        graph_size,
        clique_size,
        p_correction_type,
        vgg_input,
    )

    img = torchvision.utils.make_grid(graph_pool[0], nrow=5)

    # adding to tensorboard
    writer.add_image("Example graphs", img, dataformats="CHW")

    # flushing writer
    writer.flush()


### SAVING MODELS


# SAVING SINGLE GRAPH (working):
def tensorboard_save_models(writer, model, graph_size):

    # Add last element of configuration file to tensorboard:
    # - checking if model is VGG16 from name:
    if "VGG" in str(model):
        # - defining input for VGG16
        dummy_input = torch.randn(1, 3, graph_size, graph_size)
        writer.add_graph(model, dummy_input)
    else:
        # - defining dummy input for CNN and MLP
        dummy_input = torch.randn(1, 1, graph_size, graph_size)
        writer.add_graph(model, dummy_input)
    # flushing writer
    writer.flush()


# # SAVING MULTIPLE GRAPHS (not working, reference link: https://github.com/lanpa/tensorboardX/issues/319):
# # wrapper class to save all models to TensorBoard
# class ModelsWrapper(nn.Module):
#     def __init__(self, models_dict):
#         super().__init__()
#         # Use the provided models and save them to the class
#         for name, model in models_dict.items():
#             setattr(self, name, model)

#     # this method is needed, but it does nothing (the models are trained in train_test.py)
#     def forward(self, x):
#         return x


# # function to save models to tensorboard:
# def tensorboard_save_models(writer, wrapper, graph_size):

#     # Add combined graph to tensorboard:
#     writer.add_graph(
#         wrapper,
#         torch.randn(1, 1, graph_size, graph_size),  # dummy input
#     )
