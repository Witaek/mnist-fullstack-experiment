import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )

        conv2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )

        conv3 = nn.Sequential(
            nn.Conv2d(16,120,5),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, num_classes),
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)

        return output
