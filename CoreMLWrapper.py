import torch
import torch.nn as nn
from model import Generator

class CoreMLWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()

    def forward(self, input):
        # First run the underlying generator
        out = self.generator(input)

        # Now do the clamping and scaling (moved from the Generator's forward)
        out = torch.clamp(out, -1, 1) + 1
        out = out * (255.0 / 2)

        return out