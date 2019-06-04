import hypothesis
import numpy as np
import torch
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats
from torch.testing import assert_allclose

from dsntnn import flat_softmax


def test_flat_softmax_example():
    in_var = torch.tensor([[[[10.0, 1.0], [5.0, 2.0]]]])
    expected = torch.tensor([[
        [[0.99285460, 0.00012253],
         [0.00668980, 0.00033307]],
    ]])
    assert_allclose(flat_softmax(in_var), expected)


@hypothesis.given(
    data=arrays(np.float32, array_shapes(min_dims=3, max_dims=5),
                elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_flat_softmax_gives_valid_distribution(data):
    inp = torch.from_numpy(data)
    res = flat_softmax(inp)
    # 1. Check that all probabilities are greater than zero.
    assert np.all(np.asarray(res) >= 0)
    # 2. Check that probabilities sum to one.
    res_flat = res.view(res.size(0) * res.size(1), -1)
    res_sum = res_flat.sum(-1)
    assert_allclose(res_sum, np.ones_like(res_sum))


@hypothesis.given(
    data=arrays(np.float32, array_shapes(min_dims=3, max_dims=5),
                elements=floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False,
                                width=32)),
)
def test_flat_softmax_preserves_ranking(data):
    inp = torch.from_numpy(data)
    res = flat_softmax(inp)
    inp_flat = inp.view(inp.size(0) * inp.size(1), -1)
    res_flat = res.view(res.size(0) * res.size(1), -1)
    assert_allclose(np.argsort(res_flat, kind='mergesort'), np.argsort(inp_flat, kind='mergesort'))
