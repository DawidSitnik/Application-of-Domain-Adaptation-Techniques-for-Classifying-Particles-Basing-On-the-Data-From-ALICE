from torch import nn
from utils.config import Config

num_of_hiden_nodes = 200
num_of_hiden_nodes_dom = 50


class Net_Michal(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(6, num_of_hiden_nodes),
            nn.BatchNorm1d(num_of_hiden_nodes), nn.Dropout(p=0.3), nn.LeakyReLU(0.2, True),
            nn.Linear(num_of_hiden_nodes, num_of_hiden_nodes),
            nn.BatchNorm1d(num_of_hiden_nodes), nn.Dropout(p=0.3), nn.LeakyReLU(0.2, True),
            nn.Linear(num_of_hiden_nodes, num_of_hiden_nodes),
            nn.BatchNorm1d(num_of_hiden_nodes), nn.Dropout(p=0.3), nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_of_hiden_nodes, num_of_hiden_nodes_dom),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(num_of_hiden_nodes_dom, 2),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

class Net(nn.Module):

    def __init__(self, D_in: int):
        super().__init__()
        self.out_features = Config.net_out_features
        self.hidden_dim = Config.net_hidden_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(D_in, self.hidden_dim),
            nn.BatchNorm1d(200), nn.Dropout(p=0.3), nn.LeakyReLU(0.02, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(200), nn.Dropout(p=0.3), nn.LeakyReLU(0.02, True),
            nn.Linear(self.hidden_dim, self.out_features),
            nn.BatchNorm1d(200), nn.Dropout(p=0.3), nn.LeakyReLU(0.02, True),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(50, Config.n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.discriminator(features)
        return logits, features

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.feature_extractor.parameters(), "lr_mult": 0.1},
            {"params": self.discriminator.parameters(), "lr_mult": 1.}
        ]
        return params


wdgrl_critic = nn.Sequential(
    nn.Linear(Config.net_hidden_dim, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
