import torch
from torch.autograd import Variable
from tests.common import TestCase

from dsntnn import make_gauss, kl_reg_loss, js_reg_loss, mse_reg_loss, variance_reg_loss


def _test_reg_loss(tc, loss_method, shift_mean=True):
    # Target mean and standard deviation
    target_mean = torch.Tensor([0, 0])
    target_stddev = 0.4

    # Helper function to calculate the loss between the target and a Gaussian heatmap
    # parameterized by `mean` and `stddev`.
    def calc_loss(mean, stddev):
        hm = make_gauss(mean, 5, 5, sigma=stddev)
        return loss_method(hm, target_mean, target_stddev, mask=None)

    # Minimum loss occurs when the heatmap's mean and standard deviation are the same
    # as the target
    min_loss = calc_loss(target_mean, target_stddev)

    # Minimum loss should be close to zero
    tc.assertEqual(min_loss, 0, 1e-3)

    # Loss should increase if the heatmap has a larger or smaller standard deviation than
    # the target
    tc.assertGreater(calc_loss(target_mean, target_stddev + 0.2), min_loss + 1e-3)
    tc.assertGreater(calc_loss(target_mean, target_stddev - 0.2), min_loss + 1e-3)

    if shift_mean:
        # Loss should increase if the heatmap has its mean location at a different
        # position than the target
        tc.assertGreater(calc_loss(target_mean + 0.1, target_stddev), min_loss + 1e-3)
        tc.assertGreater(calc_loss(target_mean - 0.1, target_stddev), min_loss + 1e-3)


class TestKLRegLoss(TestCase):
    def test_kl_reg_loss(self):
        _test_reg_loss(self, kl_reg_loss)

    def test_mask(self):
        t = torch.Tensor([
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
        ])
        coords = torch.Tensor([[1, 1], [0, 0]])
        mask = torch.Tensor([1, 0])

        actual = kl_reg_loss(Variable(t), Variable(coords), 1, Variable(mask))

        self.assertEqual(1.2228811717796824, actual.data[0])


class TestMSERegLoss(TestCase):
    def test_mse_reg_loss(self):
        _test_reg_loss(self, mse_reg_loss)


class TestJSRegLoss(TestCase):
    def test_js_reg_loss(self):
        _test_reg_loss(self, js_reg_loss)


class TestVarianceRegLoss(TestCase):
    def test_variance_reg_loss(self):
        _test_reg_loss(self, variance_reg_loss, shift_mean=False)
