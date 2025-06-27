import torch
from src.quantizer import TemporalAdaptiveQuantizer

def test_quantizer_output_shape():
    quantizer = TemporalAdaptiveQuantizer()
    x = torch.randn(4, 128)
    out = quantizer(x)
    assert out.shape == x.shape

def test_quantizer_stability():
    quantizer = TemporalAdaptiveQuantizer()
    x = torch.ones(2, 128)
    out = quantizer(x)
    assert torch.all(out <= x)
