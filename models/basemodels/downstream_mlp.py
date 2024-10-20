import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_layer_size,
        dropout_prob,
    ) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, num_classes)
        self.aggregator = nn.Parameter(torch.randn((1, 13, 1, 1)))

    def forward(self, x):
        bs, n_layers, time_step, dim = x.shape
        weights = F.softmax(self.aggregator, dim=1)
        x = (x * weights).sum(dim=1)
        output = self.output(x)
        return output
