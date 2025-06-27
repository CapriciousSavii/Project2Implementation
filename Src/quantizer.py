import torch
import torch.nn as nn

class TemporalAdaptiveQuantizer(nn.Module):
    def __init__(self, base_bits=8, min_bits=2, max_bits=8):
        super().__init__()
        self.base_bits = base_bits
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.activity_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Average activity over last dimension
        spike_activity = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        bits = self.compute_adaptive_bits(spike_activity)
        return self.quantize_tensor(x, bits)

    def compute_adaptive_bits(self, activity):
        normalized = activity / self.activity_threshold
        bits = self.base_bits * torch.sigmoid(normalized)
        return torch.clamp(bits, self.min_bits, self.max_bits).round()

    def quantize_tensor(self, x, bits):
        # Quantize tensor with per-batch bit width
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + 1e-5
        x_scaled = x / scale
        bit_levels = (2 ** bits - 1)
        x_quant = torch.round(x_scaled * bit_levels) / bit_levels
        return x_quant * scale
