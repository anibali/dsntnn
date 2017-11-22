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

import torch
import torch.nn.functional
from torch.autograd import Variable


def generate_xy(inp):
    """Generate matrices X and Y."""

    *first_dims, height, width = inp.size()

    first_x = -(width - 1) / width
    first_y = -(height - 1) / height
    last_x = (width - 1) / width
    last_y = (height - 1) / height

    sing_dims = [1] * len(first_dims)
    xs = torch.linspace(first_x, last_x, width).view(*sing_dims, 1, width).expand_as(inp)
    ys = torch.linspace(first_y, last_y, height).view(*sing_dims, height, 1).expand_as(inp)

    if isinstance(inp, Variable):
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)

    xs = xs.type_as(inp)
    ys = ys.type_as(inp)

    return xs, ys


def expectation_2d(values, probabilities):
    """Calculate the expected value over values in a 2D layout.

    Args:
        values (torch.Tensor): Values for each position.
        probabilities (torch.Tensor): Probabilities for each position.

    Returns:
        The expected values.
    """

    prod = values * probabilities
    *first_dims, height, width = prod.size()
    mean = prod.view(*first_dims, height * width).sum(-1, keepdim=False)
    return mean


def dsnt(heatmaps):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """

    xs, ys = generate_xy(heatmaps)
    output = torch.stack([expectation_2d(xs, heatmaps), expectation_2d(ys, heatmaps)], -1)
    return output


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


def softmax_2d(inp):
    """Compute the softmax with the last two tensor dimensions combined."""
    size = inp.size()
    flat = inp.view(-1, size[-1] * size[-2])
    flat = torch.nn.functional.softmax(flat)
    return flat.view(*size)


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


def make_gauss(coords, width, height, sigma):
    """Draw 2D Gaussians.

    This function is differential with respect to coords.

    Args:
        coords: coordinates containing the Gaussian means (units: normalized coordinates)
        width: width of the generated images (units: pixels)
        height: height of the generated images (units: pixels)
        sigma: standard deviation of the Gaussian (units: normalized coordinates)
    """

    first_x = -(width - 1) / width
    first_y = -(height - 1) / height
    last_x = (width - 1) / width
    last_y = (height - 1) / height

    sing_dims = [1] * (coords.dim() - 1)
    xs = torch.linspace(first_x, last_x, width).view(*sing_dims, 1, width).expand(*sing_dims, height, width)
    ys = torch.linspace(first_y, last_y, height).view(*sing_dims, height, 1).expand(*sing_dims, height, width)

    if isinstance(coords, Variable):
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)

    xs = xs.type_as(coords)
    ys = ys.type_as(coords)

    k = -0.5 * (1 / sigma)**2
    xs = (xs - coords.narrow(-1, 0, 1).unsqueeze(-1)) ** 2
    ys = (ys - coords.narrow(-1, 1, 1).unsqueeze(-1)) ** 2
    gauss = ((xs + ys) * k).exp()

    # Normalize the Gaussians
    val_sum = gauss.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-24
    gauss = gauss / val_sum

    return gauss


def _kl_2d(p, q, eps=1e-24):
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = unsummed_kl.sum(-1, keepdim=False).sum(-1, keepdim=False)
    return kl_values


def _js_2d(p, q, eps=1e-24):
    m = 0.5 * (p + q)
    return 0.5 * _kl_2d(p, m, eps) + 0.5 * _kl_2d(q, m, eps)


def kl_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate the average Kullback-Leibler divergence between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in normalized units)

    Returns:
        The average KL divergence.
    """

    gauss = make_gauss(mu_t, heatmaps.size(-1), heatmaps.size(-2), sigma_t)
    divergences = _kl_2d(heatmaps, gauss)
    return divergences


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate the average Jensen-Shannon divergence between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in normalized units)

    Returns:
        The average JS divergence.
    """

    gauss = make_gauss(mu_t, heatmaps.size(-1), heatmaps.size(-2), sigma_t)
    divergences = _js_2d(heatmaps, gauss)
    return divergences


def variance_reg_losses(heatmaps, sigma_t):
    """Calculate the mean-square-error between heatmap variances and target variance.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        sigma_t (float): Target standard deviation (in normalized units)

    Returns:
        The variance MSE.
    """

    # variance = E[(x - E[x])^2]
    xs, ys = generate_xy(heatmaps)
    mean_x = expectation_2d(xs, heatmaps).unsqueeze(-1).unsqueeze(-1)
    mean_y = expectation_2d(ys, heatmaps).unsqueeze(-1).unsqueeze(-1)
    sq_xs = (xs - mean_x) ** 2
    sq_ys = (ys - mean_y) ** 2
    variance = torch.stack([expectation_2d(sq_xs, heatmaps), expectation_2d(sq_ys, heatmaps)], -1)

    sq_error = (variance - (sigma_t ** 2)) ** 2
    sq_error_sum = sq_error.sum(-1, keepdim=False)

    return sq_error_sum
