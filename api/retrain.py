import torch
import torch.nn as nn
import os

class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.fc = nn.Linear(8 * 254 * 254, 2)  # example output

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

os.makedirs("model", exist_ok=True)

dummy_model = DummyCNN()

torch.save(dummy_model, "model/model.pt")
print("model/model.pt created!")
