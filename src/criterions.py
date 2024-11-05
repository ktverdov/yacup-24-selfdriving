import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting.metrics.quantile import QuantileLoss


class CosMseLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CosMseLoss, self).__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, input1, input2):
        cosine_loss = self.cosine_loss(input1, input2, torch.ones(input1.size(0)).to(input1.device))

        mse_loss = self.mse_loss(input1, input2)
        
        combined_loss = self.alpha * cosine_loss + (1 - self.alpha) * mse_loss
        return combined_loss


def get_loss(loss_config):
    if loss_config.type == "MSELoss":
        criterion = nn.MSELoss()
    elif loss_config.type == "CosMseLoss":
        criterion = CosMseLoss()
    elif loss_config.type == "CosLoss":
        criterion = CosMseLoss(alpha=1)
    elif loss_config.type == "QuantileLoss":
        # pass quantiles list, default n = 7
        criterion = QuantileLoss()
    else:
        raise ValueError("Error in defining loss")

    return criterion
