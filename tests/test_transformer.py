import torch
from src.transformer import SpikingTransformerLayer

def test_spiking_transformer_runs():
    model = SpikingTransformerLayer(hidden_size=128)
    x = torch.randn(2, 128)
    out = model(x)
    assert out.shape == x.shape
