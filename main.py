from src.quantizer import TemporalAdaptiveQuantizer
from src.encoder import SecurityEncoder
from src.transformer import SpikingTransformerLayer
from src.detector import AdversarialDetector
import torch

model = SpikingTransformerLayer(hidden_size=128)
quantizer = TemporalAdaptiveQuantizer()
encoder = SecurityEncoder(security_level=2)
detector = AdversarialDetector()

x = torch.randn(1, 128)
secure_x = encoder.encode(x)
quant_x = quantizer(secure_x)
out = model(quant_x)

if detector.detect(out):
    print("Adversarial activity detected")
else:
    print("Output is clean")
