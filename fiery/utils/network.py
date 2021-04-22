import torch.nn as nn


def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


def preprocess_batch(batch, device, unsqueeze=False):
    for key, value in batch.items():
        if key != 'sample_token':
            batch[key] = value.to(device)
            if unsqueeze:
                batch[key] = batch[key].unsqueeze(0)


def set_module_grad(module, requires_grad=False):
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum