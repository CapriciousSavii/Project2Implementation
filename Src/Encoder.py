import torch
import os
import hashlib

class SecurityEncoder:
    def __init__(self, security_level=1, key=None):
        self.security_level = security_level
        self.key = key or os.urandom(32)

    def encode(self, x):
        if self.security_level == 0:
            return x
        noise = self._generate_mask(x.shape, x.device)
        return x + noise

    def _generate_mask(self, shape, device):
        # Use SHA256 to generate pseudorandom values from key
        flat_len = torch.prod(torch.tensor(shape)).item()
        raw_bytes = b""
        for i in range(0, flat_len, 32):
            seed = hashlib.sha256(self.key + i.to_bytes(4, 'big')).digest()
            raw_bytes += seed

        noise = torch.tensor(list(raw_bytes[:4 * flat_len]), dtype=torch.uint8, device=device)
        noise = noise.float().view(shape) / 255.0
        return noise * 0.01 * self.security_level
