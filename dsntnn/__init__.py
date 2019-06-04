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
DSNT (soft-argmax) operations for use in PyTorch computation graphs.
"""

from functools import reduce
from operator import mul

import torch
import torch.nn.functional


def linear_expectation(probs, values):
    assert(len(values) == probs.ndimension() - 2)
    expectation = []
    for i in range(2, probs.ndimension()):
        # Marginalise probabilities
        marg = probs
        for j in range(probs.ndimension() - 1, 1, -1):
            if i != j:
                marg = marg.sum(j, keepdim=False)
        # Calculate expectation along axis `i`
        expectation.append((marg * values[len(expectation)]).sum(-1, keepdim=False))
    return torch.stack(expectation, -1)


def normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector

    Returns:
        The generated vector
    """
    first = -(length - 1) / length
    last = (length - 1) / length
    return torch.linspace(first, last, length, dtype=dtype, device=device)


def soft_argmax(heatmaps, normalized_coordinates=True):
    if normalized_coordinates:
        values = [normalized_linspace(d, dtype=heatmaps.dtype, device=heatmaps.device)
                  for d in heatmaps.size()[2:]]
    else:
        values = [torch.arange(0, d, dtype=heatmaps.dtype, device=heatmaps.device)
                  for d in heatmaps.size()[2:]]
    return linear_expectation(heatmaps, values).flip(-1)


def dsnt(heatmaps, **kwargs):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """
    return soft_argmax(heatmaps, **kwargs)


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""

    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)


def euclidean_losses(actual, target):
    """Calculate the average Euclidean loss for multi-point samples.

    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    """

    assert actual.size() == target.size(), 'input tensors must have the same size'

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
        sigma: standard deviation of the Gaussian (units: pixels)
        normalize: when set to True, the returned Gaussians will be normalized
    """

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [normalized_linspace(s, dtype=means.dtype, device=means.device)
                   for s in reversed(size)]

    # PDF = exp(-(x - \mu)^2 / (2 \sigma^2))

    # dists <- (x - \mu)^2
    dists = [(x - mean) ** 2 for x, mean in zip(coords_list, means.split(1, -1))]

    # ks <- -1 / (2 \sigma^2)
    stddevs = [2 * sigma / s for s in reversed(size)]
    ks = [-0.5 * (1 / stddev) ** 2 for stddev in stddevs]

    exps = [(dist * k).exp() for k, dist in zip(ks, dists)]

    # Combine dimensions of the Gaussian
    gauss = reduce(mul, [
        reduce(lambda t, d: t.unsqueeze(d), filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, exps)
    ])

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range, gauss) + 1e-24
    return gauss / val_sum


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (Tensor): Predictions (B x L)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            (B x L), defaults to including everything
    """

    if mask is not None:
        assert mask.size() == losses.size(), 'mask must be the same size as losses'
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


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence):
    ndims = mu_t.size(-1)
    assert heatmaps.dim() == ndims + 2, 'expected heatmaps to be a {}D tensor'.format(ndims + 2)
    assert heatmaps.size()[:-ndims] == mu_t.size()[:-1]

    gauss = make_gauss(mu_t, heatmaps.size()[2:], sigma_t)
    divergences = divergence(heatmaps, gauss, ndims)
    return divergences


def kl_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Kullback-Leibler divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in pixels)

    Returns:
        Per-location KL divergences.
    """

    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _kl)


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Jensen-Shannon divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in pixels)

    Returns:
        Per-location JS divergences.
    """

    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js)


def variance_reg_losses(heatmaps, sigma_t):
    """Calculate the loss between heatmap variances and target variance.

    Note that this is slightly different from the version used in the
    DSNT paper. This version uses pixel units for variance, which
    produces losses that are larger by a constant factor.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        sigma_t (float): Target standard deviation (in pixels)

    Returns:
        Per-location sum of square errors for variance.
    """

    # mu = E[X]
    values = [normalized_linspace(d, dtype=heatmaps.dtype, device=heatmaps.device)
              for d in heatmaps.size()[2:]]
    mu = linear_expectation(heatmaps, values)
    # var = E[(X - mu)^2]
    values = [(a - b.squeeze(0)) ** 2 for a, b in zip(values, mu.split(1, -1))]
    var = linear_expectation(heatmaps, values)


    heatmap_size = torch.tensor(list(heatmaps.size()[2:]), dtype=var.dtype, device=var.device)
    actual_variance = var * (heatmap_size / 2) ** 2
    target_variance = sigma_t ** 2
    sq_error = (actual_variance - target_variance) ** 2

    return sq_error.sum(-1, keepdim=False)


def normalized_to_pixel_coordinates(coords, size):
    """Convert from normalized coordinates to pixel coordinates.

    Args:
        coords: Coordinate tensor, where elements in the last dimension are ordered as (x, y, ...).
        size: Number of pixels in each spatial dimension, ordered as (..., height, width).

    Returns:
        `coords` in pixel coordinates.
    """
    if torch.is_tensor(coords):
        size = coords.new_tensor(size).flip(-1)
    return 0.5 * ((coords + 1) * size - 1)


def pixel_to_normalized_coordinates(coords, size):
    """Convert from pixel coordinates to normalized coordinates.

    Args:
        coords: Coordinate tensor, where elements in the last dimension are ordered as (x, y, ...).
        size: Number of pixels in each spatial dimension, ordered as (..., height, width).

    Returns:
        `coords` in normalized coordinates.
    """
    if torch.is_tensor(coords):
        size = coords.new_tensor(size).flip(-1)
    return ((2 * coords + 1) / size) - 1
