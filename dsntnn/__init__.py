# Copyright 2017 Aiden Nibali
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Differentiable DSNT operations for use in PyTorch computation graphs.
"""

from functools import reduce
from operator import mul

import torch
import torch.nn.functional
from torch.autograd import Variable


def _normalized_linspace(dim_size, type_as):
    first = -(dim_size - 1) / dim_size
    last = (dim_size - 1) / dim_size
    vec = torch.linspace(first, last, dim_size)
    if isinstance(type_as, Variable):
        vec = Variable(vec, requires_grad=False)
    return vec.type_as(type_as)


def _coord_expectation(heatmaps, dim, transform=None):
    dim_size = heatmaps.size()[dim]
    own_coords = _normalized_linspace(dim_size, type_as=heatmaps)
    if transform:
        own_coords = transform(own_coords)
    *first_dims, height, width = heatmaps.size()
    summed = heatmaps.view(-1, height, width)
    for i in range(-2, 0):
        if i != dim:
            summed = summed.sum(i, keepdim=True)
    summed = summed.view(summed.size(0), -1)
    expectations = torch.mv(summed, own_coords)
    if len(first_dims) > 0:
        expectations = expectations.view(*first_dims)
    return expectations


def dsnt(heatmaps, ndims=2):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations
        ndims (int): the number of dimensions in a heatmap

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """

    dim_range = range(-1, -(ndims + 1), -1)
    mu = torch.stack([_coord_expectation(heatmaps, dim) for dim in dim_range], -1)
    return mu


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (Tensor): Predictions ([batches x] n)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            ([batches x] n), defaults to including everything
    """

    if mask is not None:
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom


def flat_softmax(inp, ndims=2):
    """Compute the softmax with the last `ndims` tensor dimensions combined."""
    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[-ndims:]))
    flat = torch.nn.functional.softmax(flat)
    return flat.view(*orig_size)


def euclidean_losses(actual, target):
    """Calculate the average Euclidean loss for multi-point samples.

    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (Tensor): Predictions ([batches x] n x d)
        target (Tensor): Ground truth target ([batches x] n x d)
    """

    # Calculate Euclidean distances between actual and target locations
    diff = actual - target
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt()
    return dist


def make_gauss(means, size, sigma, normalize=True):
    """Draw Gaussians.

    This function is differential with respect to means.

    Note on ordering: `size` expects [..., depth, height, width], whereas
    `means` expects x, y, z, ...

    Args:
        means: coordinates containing the Gaussian means (units: normalized coordinates)
        size: size of the generated images (units: pixels)
        sigma: standard deviation of the Gaussian (units: normalized coordinates)
        normalize: when set to True, the returned Gaussians will be normalized
    """

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [_normalized_linspace(s, type_as=means) for s in reversed(size)]

    dists = [(x - mean) ** 2 for x, mean in zip(coords_list, means.split(1, -1))]
    # Reshape dists so they can be added together (with broadcast)
    unsqueezed_dists = [
        reduce(lambda t, d: t.unsqueeze(d), filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, dists)
    ]

    k = -0.5 * (1 / sigma) ** 2
    gauss = (sum(unsqueezed_dists) * k).exp()

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range, gauss) + 1e-24
    return gauss / val_sum


def _kl(p, q, ndims, eps=1e-24):
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims, eps=1e-24):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims, eps) + 0.5 * _kl(q, m, ndims, eps)


def kl_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Kullback-Leibler divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in normalized units)

    Returns:
        Per-location KL divergences.
    """

    ndims = mu_t.size(-1)
    gauss = make_gauss(mu_t, heatmaps.size()[-ndims:], sigma_t)
    divergences = _kl(heatmaps, gauss, ndims)
    return divergences


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Jensen-Shannon divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in normalized units)

    Returns:
        Per-location JS divergences.
    """

    ndims = mu_t.size(-1)
    gauss = make_gauss(mu_t, heatmaps.size()[-ndims:], sigma_t)
    divergences = _js(heatmaps, gauss, ndims)
    return divergences


def _coord_variance(heatmaps, dim):
    # NOTE: Works for any coordinate, not just xs
    # mu = E[X]
    mu_x = _coord_expectation(heatmaps, dim)
    # var_x = E[(X - mu_x)^2]
    var_x = _coord_expectation(heatmaps, dim, lambda x: (x - mu_x) ** 2)
    return var_x


def variance_reg_losses(heatmaps, sigma_t, ndims=2):
    """Calculate the loss between heatmap variances and target variance.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        sigma_t (float): Target standard deviation (in normalized units)
        ndims (int): Number of dimensions in a heatmap

    Returns:
        Per-location sum of square errors for variance.
    """

    dim_range = range(-1, -(ndims + 1), -1)
    variance = torch.stack([_coord_variance(heatmaps, dim) for dim in dim_range], -1)

    var_t = sigma_t ** 2
    sq_error = (variance - var_t) ** 2

    return sq_error.sum(-1, keepdim=False)
