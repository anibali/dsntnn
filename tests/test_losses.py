import hypothesis
import numpy as np
import pytest
import torch
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats
from torch.testing import assert_allclose

from dsntnn import average_loss, euclidean_losses, l1_losses, mse_losses

_LOSSES_FNS = {
    'euclidean': euclidean_losses,
    'l1': l1_losses,
    'mse': mse_losses,
}


class TestLosses:
    """Common tests for all loss functions."""

    @pytest.fixture(params=['euclidean', 'l1', 'mse'])
    def losses_fn(self, request):
        return _LOSSES_FNS[request.param]

    def test_smoke(self, losses_fn):
        input_tensor = torch.randn(4, 3, 2)
        target = torch.randn(4, 3, 2)
        losses = losses_fn(input_tensor, target)
        assert losses.shape == (4, 3)

    @hypothesis.given(
        data=arrays(np.float32, array_shapes(min_dims=3, max_dims=3),
                    elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    )
    def test_same(self, data, losses_fn):
        input_tensor = torch.as_tensor(data)
        target = input_tensor.clone()
        losses = losses_fn(input_tensor, target)
        assert_allclose(losses, torch.zeros_like(losses))


class TestEuclideanLosses:
    def test_forward_and_backward(self):
        input_tensor = torch.tensor([
            [[3.0, 4.0], [3.0, 4.0]],
            [[3.0, 4.0], [3.0, 4.0]],
        ])

        target = torch.tensor([
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])

        in_var = input_tensor.detach().requires_grad_(True)

        expected_loss = 5.0
        actual_loss = average_loss(euclidean_losses(in_var, target))
        expected_grad = torch.tensor([
            [[0.15, 0.20], [0.15, 0.20]],
            [[0.15, 0.20], [0.15, 0.20]],
        ])
        actual_loss.backward()

        assert float(actual_loss) == expected_loss
        assert_allclose(expected_grad, in_var.grad)


def test_average_loss_mask():
    losses = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    mask = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    actual = average_loss(losses, mask)
    assert float(actual) == 0.0
