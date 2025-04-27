# utils/losses.py
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = torch.nn.functional.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        smooth = 1.0
        dice = 0.0
        for c in range(self.num_classes):
            input_flat = inputs[:, c].contiguous().view(-1)
            target_flat = targets[:, c].contiguous().view(-1)
            intersection = (input_flat * target_flat).sum()
            dice += (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice / self.num_classes