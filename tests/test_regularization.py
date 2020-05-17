import pytest
import torch
from pytest import approx

from dsntnn import make_gauss, average_loss, kl_reg_losses, js_reg_losses, variance_reg_losses


def _test_reg_loss(loss_method, uses_mean=True):
    # Target mean and standard deviation
    target_mean = torch.tensor([[[0.0, 0.0]]])
    target_stddev = 1.0

    # Helper function to calculate the loss between the target and a Gaussian heatmap
    # parameterized by `mean` and `stddev`.
    def calc_loss(mean, stddev):
        hm = make_gauss(mean, [9, 9], sigma=stddev)
        args = [hm]
        if uses_mean: args.append(target_mean)
        args.append(target_stddev)
        return average_loss(loss_method(*args)).item()

    # Minimum loss occurs when the heatmap's mean and standard deviation are the same
    # as the target
    min_loss = calc_loss(target_mean, target_stddev)

    # Minimum loss should be close to zero
    assert min_loss == approx(0, abs=1e-6)

    # Loss should increase if the heatmap has a larger or smaller standard deviation than
    # the target
    assert calc_loss(target_mean, target_stddev + 0.2) > min_loss + 1e-3
    assert calc_loss(target_mean, target_stddev - 0.2) > min_loss + 1e-3

    if uses_mean:
        # Loss should increase if the heatmap has its mean location at a different
        # position than the target
        assert calc_loss(target_mean + 0.1, target_stddev) > min_loss + 1e-3
        assert calc_loss(target_mean - 0.1, target_stddev) > min_loss + 1e-3


def test_kl_reg_loss():
    _test_reg_loss(kl_reg_losses)


def test_kl_mask():
    t = torch.tensor([[
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.1, 0.8],
        ],
        [
            [0.8, 0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ]])
    coords = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    actual = average_loss(kl_reg_losses(t, coords, 2.0), mask)

    assert actual.item() == approx(1.2228811717796824)


def test_kl_rectangular():
    t = torch.tensor([[
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.8],
        ]
    ]])
    coords = torch.tensor([[[1.0, 1.0]]])

    actual = average_loss(kl_reg_losses(t, coords, 2.0))

    assert actual.item() == pytest.approx(1.26467538775)


def test_js_reg_loss():
    _test_reg_loss(js_reg_losses)


def test_variance_reg_loss():
    _test_reg_loss(variance_reg_losses, uses_mean=False)


def test_variance_exact():
    t = torch.tensor([[
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.6, 0.1],
            [0.0, 0.0, 0.1, 0.0],
        ]
    ]])

    actual = average_loss(variance_reg_losses(t, 2.0))
    assert actual.item() == 28.88


def test_variance_rectangular():
    t = torch.tensor([[
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.6, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        ]
    ]])

    actual = average_loss(variance_reg_losses(t, 2.0))
    assert actual.item() == 28.88


def test_variance_3d():
    t = torch.tensor([[
        [[
            [0.000035, 0.000002, 0.000000],
            [0.009165, 0.000570, 0.000002],
            [0.147403, 0.009165, 0.000035],
        ], [
            [0.000142, 0.000009, 0.000000],
            [0.036755, 0.002285, 0.000009],
            [0.591145, 0.036755, 0.000142],
        ], [
            [0.000035, 0.000002, 0.000000],
            [0.009165, 0.000570, 0.000002],
            [0.147403, 0.009165, 0.000035],
        ]]
    ]])
    actual = average_loss(variance_reg_losses(t, 0.6))
    assert actual.item() == approx(0.18564102213775013)


def test_variance_batch():
    t = torch.tensor([
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.6, 0.1],
            [0.0, 0.0, 0.1, 0.0],
        ]],
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.1, 0.5, 0.1, 0.0],
            [0.0, 0.1, 0.0, 0.0],
        ]]
    ])

    actual = average_loss(variance_reg_losses(t, 2.0))
    assert actual.item() == approx(28.54205)
