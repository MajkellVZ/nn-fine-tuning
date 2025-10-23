import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        return self.net(x)

    def freeze_feature_extractor(self):
        for param in self.parameters():
            param.requires_grad = False
        last_linear = self.net[-1]
        for p in last_linear.parameters():
            p.requires_grad = True
        print('Feature extractor frozen')

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print('Parameters unfrozen')
