import torch.nn as nn

class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2),
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2),
            nn.Conv2d(32, 64, (3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1920, 256)
        )
    
    def forward(self, x):
        output = self.layers(x)
        return output 