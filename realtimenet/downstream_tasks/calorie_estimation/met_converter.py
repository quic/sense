import torch.nn as nn


class METValueMLPConverter(nn.Module):

    def __init__(self):
        super().__init__()

        self.met_regressor = nn.Sequential(
            nn.Linear(1280, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.ReLU()
        )

    def forward(self, feature):
        return self.met_regressor(feature)
