import torch

class AdversarialDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def detect(self, spike_tensor):
        mean = torch.mean(spike_tensor).item()
        std = torch.std(spike_tensor).item()
        if std < 0.01 or mean > self.threshold:
            return True  # likely adversarial
        return False
