import torch
import torch.nn as nn
import torch.optim as optim


class Models:
    @staticmethod
    def mlp(graph_size, l1=1000, l2=500, l3=100, dropout_prob=0.1):

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NOTE:
        # - layer sizes should be defined dynamically based on graph_size
        # - with bigger graph_size, Rudy was using an additional block of 3 layers -> define dynamically

        # Define MLP architecture here
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(graph_size * graph_size, l1),
            nn.BatchNorm1d(l1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(l2, l3),
            nn.BatchNorm1d(l3),
            nn.ReLU(),
            nn.Linear(l3, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )

        # TO BE DEFINED IN CONFIGURATION FILES?
        optim = torch.optim.Adam(model.parameters())  # optimization with Adam
        criterion = nn.CrossEntropyLoss()  # criterion = Cross Entropy
        model.to(device)

        return model

    # @staticmethod
    # def cnn():
    #     # Define CNN architecture here
    #     model = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size),
    #         nn.Flatten(),
    #         nn.Linear(hidden_size, out_features),
    #     )
    #     return model
