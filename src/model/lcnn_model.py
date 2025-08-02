import torch
import torch.nn as nn


class MaxFeatureMap(nn.Module):
    """
    Max-Feature-Map (MFM) activation function.
    """
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.max(x1, x2)


class LCNNModel(nn.Module):
    """
    Conventional Light CNN (LCNN) for voice anti-spoofing.
    """
    def __init__(self, n_class=2, input_channels=1, dropout_rate=0.75):
        super().__init__()
        
        # Initial block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            MaxFeatureMap(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap()
        )
        
        # Block 3
        self.conv_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48)
        )
        
        # Block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap()

        )
        
        # Block 5
        self.conv_block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 6
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap()
        )
        
        # Block 7
        self.conv_block7 = nn.Sequential(
            nn.BatchNorm2d(32)
        )
        
        # Block 8
        self.conv_block8 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap()
        )
        
        # Block 9
        self.conv_block9 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final
        self.fc29 = nn.Linear(32, 160)
        self.mfm30 = MaxFeatureMap()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn31 = nn.BatchNorm1d(80)
        self.fc32 = nn.Linear(80, n_class)

    
    def forward(self, data_object, **batch):
        x = data_object
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc29(x)
        x = self.mfm30(x)

        x = self.dropout(x)
        x = self.bn31(x)
        
        logits = self.fc32(x)
        
        return {"logits": logits}
    
    def __str__(self):
        total_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + f"\nTotal params: {total_params:,}"