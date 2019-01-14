import torch
from tests.common import TestCase

from dsntnn import flat_softmax


class TestFlatSoftmax(TestCase):
    def test_forward(self):
        in_var = torch.Tensor([[[[10, 1], [5, 2]]]])
        expected = torch.Tensor([[
            [[0.99285460, 0.00012253],
             [0.00668980, 0.00033307]],
        ]])
        self.assertEqual(flat_softmax(in_var), expected)
