# from collections import defaultdict
import torch
import math
import torch.nn.functional as F
# import matplotlib.pyplot as plt


def masked_l1_loss(input, target, mask):
    return (F.l1_loss(input, target, reduction='none') * mask.float()).sum()


def huber_loss(input, target, mask=None):
    loss = F.smooth_l1_loss(input, target, reduction='none')
    if mask is None:
        return loss.sum()
    return (loss * mask.float()).sum()


def hard_neg_mining_loss(scores, labels, neg_ratio=5):

    # Flatten tensors along the spatial dimensions
    scores = scores.flatten(2, 3)
    labels = labels.flatten(2, 3)
    count = labels.size(-1)

    # Rank negative locations by the predicted confidence
    _, inds = (scores.sigmoid() * (~labels).float()).sort(-1, descending=True)
    ordinals = torch.arange(count, out=inds.new_empty(count)).expand_as(inds)
    rank = torch.empty_like(inds)
    rank.scatter_(-1, inds, ordinals)

    # Include only positive locations + N most confident negative locations
    num_pos = labels.long().sum(dim=-1, keepdim=True)
    num_neg = (num_pos + 1) * neg_ratio
    mask = (labels | (rank < num_neg)).float()

    # Apply cross entropy loss
    return F.binary_cross_entropy_with_logits(
        scores, labels.float(), mask, reduction='sum')


def balanced_cross_entropy_loss(scores, labels):
    labels = labels.float()

    # Weight the loss by the relative number of positive and negative examples
    num_pos = int(labels.long().sum()) + 1
    num_neg = labels.numel() - num_pos
    weights = (num_neg - num_pos) * labels + num_pos

    # Compute cross entropy loss
    return F.binary_cross_entropy_with_logits(scores, labels, weights)


# def heatmap_loss(scores, labels, thresh=0.05, pos_weight=100):
#     labels = labels.float()
#     mask = (labels > thresh).float()
#     loss = F.l1_loss(scores, labels, reduction='none')
#     weighted = loss * (1. + (pos_weight - 1.) * mask)

#     return weighted.sum()


def heatmap_loss(heatmap, gt_heatmap, weights=[100], thresh=0.05):

    positives = (gt_heatmap > thresh).float()
    weights = heatmap.new(weights).view(1, -1, 1, 1)

    loss = F.l1_loss(heatmap, gt_heatmap, reduce=False)

    loss *= positives * weights + (1 - positives)
    return loss.sum()


# def uncertainty_loss(logvar, sqr_dists):
#     sqr_dists = sqr_dists.clamp(min=1.+1e-6)
#     c = (1 + torch.log(sqr_dists)) / sqr_dists
#     loss = torch.log1p(logvar.exp()) / sqr_dists + torch.sigmoid(-logvar) - c
#     print('dists', float(sqr_dists.min()), float(sqr_dists.max()))
#     print('logvar', float(logvar.min()), float(logvar.max()))
#     print('loss', float(loss.min()), float(loss.max()))

#     def hook(grad):
#         print('grad', float(grad.min()), float(grad.max()), float(grad.sum()))
#     logvar.register_hook(hook)

#     return loss.mean()


def uncertainty_loss(logvar, sqr_dists):
    dists = sqr_dists + 1.
    loss = torch.exp(-logvar) + (logvar - dists.log() - 1) / dists
    print('dists', float(sqr_dists.min()), float(sqr_dists.max()))
    print('logvar', float(logvar.min()), float(logvar.max()))
    print('loss', float(loss.min()), float(loss.max()))

    def hook(grad):
        print('grad', float(grad.min()), float(grad.max()), float(grad.sum()))
    logvar.register_hook(hook)

    if (logvar > 10).any():
        raise RuntimeError()

    return loss.mean()


def compute_uncertainty(logvar, sqr_dists, min_dist):
    var = torch.exp(logvar)
    return min_dist / torch.sqrt(var) * torch.exp(
        -0.5 * (sqr_dists / logvar - 1.))


CONST = 1.1283791670955126


def log_ap_loss(logvar, sqr_dists, num_thresh=10):

    print('dists', float(sqr_dists.min()), float(sqr_dists.max()))
    print('logvar', float(logvar.min()), float(logvar.max()))

    def hook(grad):
        print('grad', float(grad.min()), float(grad.max()), float(grad.sum()))
    logvar.register_hook(hook)

    variance = torch.exp(logvar).view(-1, 1)
    stdev = torch.sqrt(variance)
    print('stdev', float(stdev.min()), float(stdev.max()))

    max_dist = math.sqrt(float(sqr_dists.max()))
    minvar, maxvar = float(stdev.min()), float(stdev.max())
    thresholds = torch.logspace(
        math.log10(1 / maxvar), math.log10(max_dist / minvar), num_thresh).type_as(stdev)

    print('maxdist: {:.2e} minvar: {:.2e} maxvar: {:.2e}'.format(max_dist, minvar, maxvar))
    print('thresholds {:.2e} - {:.2e}'.format(thresholds.min(), thresholds.max()))

    k_sigma = stdev * thresholds
    k_sigma_sqr = variance * thresholds ** 2
    mask = (sqr_dists.view(-1, 1) < k_sigma_sqr).float()

    erf = torch.erf(k_sigma)
    masked_erf = erf * mask
    masked_exp = stdev * torch.exp(-k_sigma_sqr) * mask

    loss = masked_exp.sum(0) * masked_erf.sum(0) / erf.sum(0)
    loss = (loss[0] + loss[-1]) / 2. + loss[1:-1].sum()
    return -torch.log(loss * CONST / len(variance))


def compute_loss(pred_encoded, gt_encoded, loss_weights=[1., 1., 1., 1.]):

    # Expand tuples
    score, pos_offsets, dim_offsets, ang_offsets = pred_encoded
    heatmaps, gt_pos_offsets, gt_dim_offsets, gt_ang_offsets, mask = gt_encoded
    score_weight, pos_weight, dim_weight, ang_weight = loss_weights

    # Compute losses
    score_loss = heatmap_loss(score, heatmaps)
    pos_loss = masked_l1_loss(pos_offsets, gt_pos_offsets, mask.unsqueeze(2))
    dim_loss = masked_l1_loss(dim_offsets, gt_dim_offsets, mask.unsqueeze(2))
    ang_loss = masked_l1_loss(ang_offsets, gt_ang_offsets, mask.unsqueeze(2))

    # Combine loss
    total_loss = score_loss * score_weight + pos_loss * pos_weight \
        + dim_loss * dim_weight + ang_loss * ang_weight

    # Store scalar losses in a dictionary
    loss_dict = {
        'score': float(score_loss), 'position': float(pos_loss),
        'dimension': float(dim_loss), 'angle': float(ang_loss),
        'total': float(total_loss)
    }

    return total_loss, loss_dict
