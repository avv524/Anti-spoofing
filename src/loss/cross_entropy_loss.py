import torch
import torch.nn as nn


class StandardCrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss for comparison
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels, **batch):
        return {"loss": self.criterion(logits, labels)} 