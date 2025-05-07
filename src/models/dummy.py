import torch
import torch.nn.functional as F

class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(DummyModel, self).__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=2, stride = 4 )  
        self.fc = torch.nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))      
        print(x.shape)
        x = x.view(x.size(0), -1)     
        x = self.fc(x)                
        return x
    