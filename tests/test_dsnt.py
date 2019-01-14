import torch
from tests.common import TestCase

from dsntnn import dsnt, linear_expectation


class TestDSNT(TestCase):
    SIMPLE_INPUT = torch.Tensor([[[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.6, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]]])

    SIMPLE_OUTPUT = torch.Tensor([[[0.4, 0.0]]])

    SIMPLE_TARGET = torch.Tensor([[[0.5, 0.5]]])

    # Expected dloss/dinput when using MSE with target (0.5, 0.5)
    SIMPLE_GRAD_INPUT = torch.Tensor([[[
        [0.4800, 0.4400, 0.4000, 0.3600, 0.3200],
        [0.2800, 0.2400, 0.2000, 0.1600, 0.1200],
        [0.0800, 0.0400, 0.0000, -0.0400, -0.0800],
        [-0.1200, -0.1600, -0.2000, -0.2400, -0.2800],
        [-0.3200, -0.3600, -0.4000, -0.4400, -0.4800],
    ]]])

    def test_forward(self):
        expected = self.SIMPLE_OUTPUT
        actual = dsnt(self.SIMPLE_INPUT)
        self.assertEqual(actual, expected)

    def test_backward(self):
        mse = torch.nn.MSELoss()

        in_var = self.SIMPLE_INPUT.detach().requires_grad_(True)
        output = dsnt(in_var)

        loss = mse(output, self.SIMPLE_TARGET)
        loss.backward()

        self.assertEqual(in_var.grad, self.SIMPLE_GRAD_INPUT)

    def test_cuda(self):
        mse = torch.nn.MSELoss()

        in_var = self.SIMPLE_INPUT.detach().cuda().requires_grad_(True)

        expected_output = self.SIMPLE_OUTPUT.cuda()
        output = dsnt(in_var)
        self.assertEqual(output, expected_output)

        target_var = self.SIMPLE_TARGET.cuda()
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = self.SIMPLE_GRAD_INPUT.cuda()
        self.assertEqual(in_var.grad, expected_grad)

    def test_3d(self):
        inp = torch.Tensor([[
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

        expected = torch.Tensor([[[-2/3, 2/3, 0]]])
        self.assertEqual(dsnt(inp), expected)

    def test_linear_expectation(self):
        probs = torch.Tensor([[[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.3, 0.0],
            [0.0, 0.0, 0.3, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]]])
        values = [torch.arange(d, dtype=probs.dtype, device=probs.device) for d in probs.size()[2:]]

        expected = torch.Tensor([[[1.5, 2.5]]])
        actual = linear_expectation(probs, values)
        self.assertEqual(actual, expected)
