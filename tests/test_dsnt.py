import os
from tempfile import TemporaryDirectory

import torch
from torch import nn, onnx
from torch.nn.functional import mse_loss
from torch.testing import assert_allclose

from dsntnn import dsnt, linear_expectation, normalized_to_pixel_coordinates, \
    pixel_to_normalized_coordinates, flat_softmax

SIMPLE_INPUT = torch.tensor([[[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1, 0.6, 0.1],
    [0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]]])

SIMPLE_OUTPUT = torch.tensor([[[0.4, 0.0]]])

SIMPLE_TARGET = torch.tensor([[[0.5, 0.5]]])

# Expected dloss/dinput when using MSE with target (0.5, 0.5)
SIMPLE_GRAD_INPUT = torch.tensor([[[
    [0.4800, 0.4400, 0.4000, 0.3600, 0.3200],
    [0.2800, 0.2400, 0.2000, 0.1600, 0.1200],
    [0.0800, 0.0400, 0.0000, -0.0400, -0.0800],
    [-0.1200, -0.1600, -0.2000, -0.2400, -0.2800],
    [-0.3200, -0.3600, -0.4000, -0.4400, -0.4800],
]]])


def test_dsnt_forward():
    expected = SIMPLE_OUTPUT
    actual = dsnt(SIMPLE_INPUT)
    assert_allclose(actual, expected)


def test_dsnt_trace():
    def op(inp):
        return dsnt(inp, normalized_coordinates=True)
    jit_op = torch.jit.trace(op, (SIMPLE_INPUT,))
    expected = op(SIMPLE_INPUT)
    actual = jit_op(SIMPLE_INPUT)
    assert_allclose(actual, expected)


def test_dsnt_forward_not_normalized():
    expected = torch.tensor([[[3.0, 2.0]]])
    actual = dsnt(SIMPLE_INPUT, normalized_coordinates=False)
    assert_allclose(actual, expected)


def test_dsnt_trace_not_normalized():
    def op(inp):
        return dsnt(inp, normalized_coordinates=False)
    jit_op = torch.jit.trace(op, (SIMPLE_INPUT,))
    expected = op(SIMPLE_INPUT)
    actual = jit_op(SIMPLE_INPUT)
    assert_allclose(actual, expected)


def test_dsnt_backward():
    in_var = SIMPLE_INPUT.detach().requires_grad_(True)
    output = dsnt(in_var)

    loss = mse_loss(output, SIMPLE_TARGET)
    loss.backward()

    assert_allclose(in_var.grad, SIMPLE_GRAD_INPUT)


def test_dsnt_cuda():
    mse = torch.nn.MSELoss()

    in_var = SIMPLE_INPUT.detach().cuda().requires_grad_(True)

    expected_output = SIMPLE_OUTPUT.cuda()
    output = dsnt(in_var)
    assert_allclose(output, expected_output)

    target_var = SIMPLE_TARGET.cuda()
    loss = mse(output, target_var)
    loss.backward()

    expected_grad = SIMPLE_GRAD_INPUT.cuda()
    assert_allclose(in_var.grad, expected_grad)


def test_dsnt_3d():
    inp = torch.tensor([[
        [[
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ], [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00],
        ], [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ]]
    ]])

    expected = torch.tensor([[[-2/3, 2/3, 0]]])
    assert_allclose(dsnt(inp), expected)


def test_dsnt_linear_expectation():
    probs = torch.tensor([[[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.3, 0.0],
        [0.0, 0.0, 0.3, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]]])
    values = [torch.arange(d, dtype=probs.dtype, device=probs.device) for d in probs.size()[2:]]

    expected = torch.tensor([[[1.5, 2.5]]])
    actual = linear_expectation(probs, values)
    assert_allclose(actual, expected)


def test_normalized_to_pixel_coordinates():
    expected = torch.tensor([0.5, 2.0])
    actual = normalized_to_pixel_coordinates(torch.tensor([0.0, 0.0]), (5, 2))
    assert_allclose(actual, expected)


def test_pixel_to_normalized_coordinates():
    expected = torch.tensor([0.0, 0.0])
    actual = pixel_to_normalized_coordinates(torch.tensor([0.5, 2.0]), (5, 2))
    assert_allclose(actual, expected)


def test_onnx():
    class SimpleModel(nn.Module):
        def forward(self, x):
            x = flat_softmax(x)
            x = dsnt(x, normalized_coordinates=True)
            return x
    dummy_input = torch.randn((2, 4, 32, 32))
    model = SimpleModel()
    model.eval()
    with TemporaryDirectory() as d:
        onnx_file = os.path.join(d, 'model.onnx')
        onnx.export(model, (dummy_input,), onnx_file, verbose=False)
        assert os.path.isfile(onnx_file)
