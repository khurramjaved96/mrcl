import torch
from torch import nn


def conv2d(param, adaptation, meta):
    w = nn.Parameter(torch.ones(param['out-channels'], param['in-channels'], param['kernal'], param['kernal']))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(param['out-channels']))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b


def linear(out_dim, in_dim, adaptation, meta):
    w = nn.Parameter(torch.ones(out_dim, in_dim))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(out_dim))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b

def linear_sparse(out_dim, in_dim, spread, adaptation, meta):
    w = nn.Parameter(torch.ones(spread, in_dim))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(out_dim))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b


def static_plasticity(parameters):
    weights = nn.Parameter(torch.ones(parameters))
    weights.adaptation = False
    weights.meta = True
    return weights
#

def local_connectivity(parameters):
    if len(parameters.shape) > 1 and parameters.shape[0] == parameters.shape[1]:
        weights = torch.zeros(parameters.shape)
        for elem in range(parameters.shape[0]):
            spread = 20
            left = max(0, elem - spread)
            right = min(parameters.shape[0], elem + spread)
            weights[elem - left:elem + right, elem - left: elem + right] = 1
            # weights[elem, elem] = 1
        weights = nn.Parameter(weights)
    else:
        weights = nn.Parameter(torch.ones(parameters.shape))
    weights.adaptation = False
    weights.meta = True
    return weights
