import torch
import torch.nn as nn

class SpikingTransformerLayer(nn.Module):
    def __init__(self, hidden_size, spike_threshold=0.5):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.leak_factor = nn.Parameter(torch.tensor(0.9))
        self.spike_threshold = spike_threshold

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.leak_factor * residual + (1 - self.leak_factor) * x
        return self._spike_fn(x)

    def _spike_fn(self, x):
        return (x > self.spike_threshold).float()
