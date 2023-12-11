import torch.nn.functional as F
import torch


class LossEight_L1Smooth(torch.nn.Module):
    def __init__(self):
        super(LossEight_L1Smooth, self).__init__()

    def forward(self, preds, labels):
        attn_a = labels['react_attn_a']
        targets_a = labels['react_a']
        attn_d = labels['react_attn_d']
        targets_d = labels['react_d']

        preds_a = preds[:, :, 0]
        preds_d = preds[:, :, 1]

        loss_a = F.smooth_l1_loss(preds_a, targets_a, beta=0.1, reduction='none')
        loss_a = loss_a[attn_a].mean()
        loss_d = F.smooth_l1_loss(preds_d, targets_d, beta=0.1, reduction='none')
        loss_d = loss_d[attn_d].mean()

        loss = (loss_a + loss_d) / 2
        return loss


class LossNine_MSE(torch.nn.Module):
    def __init__(self):
        super(LossNine_MSE, self).__init__()

    def forward(self, preds, labels):
        attn_a = labels['react_attn_a']
        targets_a = labels['react_a']
        attn_d = labels['react_attn_d']
        targets_d = labels['react_d']

        preds_a = preds[:, :, 0]
        preds_d = preds[:, :, 1]

        loss_a = F.mse_loss(preds_a, targets_a, reduction='none')
        loss_a = loss_a[attn_a].mean()
        loss_d = F.mse_loss(preds_d, targets_d, reduction='none')
        loss_d = loss_d[attn_d].mean()

        loss = (loss_a + loss_d)  # won't divide by 2 to make the number a bit bigger
        return loss





