import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        seq_len = loss.shape[1]
        future_discounts = self.future_discount ** torch.arange(seq_len, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * future_discounts

        return loss[mask].mean()


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )

        loss = loss.view(b, s, h, w)

        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)


class ProbabilisticLoss(nn.Module):
    def forward(self, output):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss
