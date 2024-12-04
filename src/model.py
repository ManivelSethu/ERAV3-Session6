import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
       
        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 3, padding=0),  # Image size: 28x28 -> 26x26
            nn.ReLU(),
            nn.BatchNorm2d(20, track_running_stats=True),
            nn.Dropout(0.1),
            nn.Conv2d(20, 20, 3, padding=0),  # 24x24
            nn.ReLU(),
            nn.BatchNorm2d(20, track_running_stats=True),
            nn.Dropout(0.1),
            nn.Conv2d(20, 10, 1, padding=0),  # 24x24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 12x12 after max pooling
        )

        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0),  # 10x10
            nn.ReLU(),
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, padding=0),  # 8x8
            nn.ReLU(),
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, padding=0),  # 6x6
            nn.ReLU(),
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, padding=0),  # 4x4
            nn.ReLU(),
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.Dropout(0.1),
        )

        # Conv Block 3 with Global Average Pooling
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 10, 4, padding=0),  # Output size: 1x1
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling instead of BatchNorm
        )

        # Fully Connected Layer (final layer)
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        # Apply all the layers sequentially
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, 10)
        
        # Final fully connected layer for 10-class output
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def has_batch_norm(self):
        return any(isinstance(m, nn.BatchNorm2d) for m in self.modules())

    def has_dropout(self):
        return any(isinstance(m, nn.Dropout) for m in self.modules())

    def has_fc_layer(self):
        return any(isinstance(m, nn.Linear) for m in self.modules()) 