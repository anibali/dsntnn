import torch
from tests.common import TestCase

from dsntnn import make_gauss


class TestMakeGauss(TestCase):
    def test_make_gauss(self):
        expected = torch.Tensor([
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
            [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
        ])
        actual = make_gauss(torch.Tensor([0, 0]), 5, 5, sigma=0.4)
        self.assertEqual(expected, actual, 1e-4)
