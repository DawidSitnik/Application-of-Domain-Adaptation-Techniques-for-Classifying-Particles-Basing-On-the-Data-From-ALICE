import torch.nn as nn


def feature_extractor():
    feature_extractor = nn.Sequential(
        nn.Linear(6, 200),
        nn.BatchNorm1d(200), nn.Dropout(p=0.3), nn.LeakyReLU(0.02, True),
        nn.Linear(200, 200),
        nn.BatchNorm1d(200), nn.Dropout(p=0.3), nn.LeakyReLU(0.02, True),
        nn.Linear(200, 200),
        nn.BatchNorm1d(200), nn.Dropout(p=0.3), nn.LeakyReLU(0.02, True)
    )

    return feature_extractor


def discriminator():
    discriminator = nn.Sequential(
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(50, 1),
    )

    return discriminator


def critic():
    critic = nn.Sequential(
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    return critic


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = feature_extractor()
        self.discriminator = discriminator()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.discriminator(features)
        return logits