import torch
from torch.autograd import Variable
from tests.common import TestCase

from dsntnn import average_loss, euclidean_losses


class TestEuclideanLoss(TestCase):
    def test_forward_and_backward(self):
        input_tensor = torch.Tensor([
            [[3, 4], [3, 4]],
            [[3, 4], [3, 4]],
        ])

        target = torch.Tensor([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ])

        in_var = Variable(input_tensor, requires_grad=True)

        expected_loss = 5.0
        actual_loss = average_loss(euclidean_losses(in_var, Variable(target)))
        expected_grad = torch.Tensor([
            [[0.15, 0.20], [0.15, 0.20]],
            [[0.15, 0.20], [0.15, 0.20]],
        ])
        actual_loss.backward()
        actual_grad = in_var.grad.data

        self.assertEqual(expected_loss, actual_loss.item())
        self.assertEqual(expected_grad, actual_grad)

    def test_mask(self):
        output = torch.Tensor([
            [[0, 0], [1, 1], [0, 0]],
            [[1, 1], [0, 0], [0, 0]],
        ])

        target = torch.Tensor([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ])

        mask = torch.Tensor([
            [1, 0, 1],
            [0, 1, 1],
        ])

        expected = 0.0
        actual = average_loss(
            euclidean_losses(Variable(output), Variable(target)),
            Variable(mask))

        self.assertEqual(expected, actual.item())
