import torch
from tests.common import TestCase

from dsntnn import make_gauss


class TestMakeGauss(TestCase):
    def test_2d(self):
        expected = torch.Tensor([
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
            [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
        ])
        actual = make_gauss(torch.Tensor([0, 0]), [5, 5], sigma=0.4)
        self.assertEqual(expected, actual, 1e-4)

    def test_3d(self):
        expected = torch.Tensor([[
            [0.0000, 0.0000, 0.0000],
            [0.0092, 0.0006, 0.0000],
            [0.1474, 0.0092, 0.0000]
        ], [
            [0.0001, 0.0000, 0.0000],
            [0.0368, 0.0023, 0.0000],
            [0.5911, 0.0368, 0.0001]
        ], [
            [0.0000, 0.0000, 0.0000],
            [0.0092, 0.0006, 0.0000],
            [0.1474, 0.0092, 0.0000]
        ]])
        actual = make_gauss(torch.Tensor([-1, 1, 0]), [3, 3, 3], sigma=0.4)
        self.assertEqual(expected, actual, 1e-4)

    def test_unnormalized(self):
        actual = make_gauss(torch.Tensor([0, 0]), [5, 5], sigma=1.0, normalize=False)
        self.assertEqual(1.0, actual.max())
