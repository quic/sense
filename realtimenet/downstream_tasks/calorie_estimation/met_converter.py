import torch.nn as nn


class METValueMLPConverter(nn.Module):

    def __init__(self, global_average_pooling=True):
        super().__init__()

        self.met_regressor = nn.Sequential(
            nn.Linear(1280, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.ReLU()
        )
        self.global_average_pooling = global_average_pooling

    def forward(self, feature):
        if self.global_average_pooling:
            feature = feature.mean(dim=-1).mean(dim=-1)
        return self.met_regressor(feature)
