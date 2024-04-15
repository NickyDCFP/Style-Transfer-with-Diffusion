import torch

class ImageEmbedding(torch.nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, (3,3), stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2), stride=2),
            torch.nn.BatchNorm2d(10),
            torch.nn.Conv2d(10, 4, (5,5), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (7,7), stride=1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256)
        )
    
    def forward(self, x):
        output = self.layers(x)
        return output 
    