import torch
from src.encoder import SecurityEncoder

def test_encoder_adds_noise():
    encoder = SecurityEncoder(security_level=2)
    x = torch.randn(4, 128)
    encoded = encoder.encode(x)
    assert not torch.equal(x, encoded)
