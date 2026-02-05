import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.6, reduction='none'):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, logits, target):
        
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -torch.sum(true_dist * F.log_softmax(logits, dim=-1), dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  
            return loss