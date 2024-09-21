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

        self.hidden = nn.Linear(input_dim, hidden_layer_size)
        self.output = nn.Linear(hidden_layer_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        x = F.relu(x)

        output = self.output(x)
        return output
