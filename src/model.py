import torch
import torch.nn as nn

class BaggedMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for cardiac risk prediction
    trained with Bayesian Bootstrap Aggregation (Bagging).
    """
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64):
        super(BaggedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
