import torch
from torch.testing import assert_allclose

from dsntnn import make_gauss


def test_make_gauss_2d():
    expected = torch.tensor([
        [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
        [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
        [0.021938, 0.098320, 0.162103, 0.098320, 0.021938],
        [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
        [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
    ])
    actual = make_gauss(torch.tensor([0.0, 0.0]), [5, 5], sigma=1.0)
    assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_make_gauss_3d():
    expected = torch.tensor([[
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
    ]])
    actual = make_gauss(torch.tensor([-1.0, 1.0, 0.0]), [3, 3, 3], sigma=0.6)
    assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_make_gauss_unnormalized():
    actual = make_gauss(torch.tensor([0.0, 0.0]), [5, 5], sigma=1.0, normalize=False)
    assert actual.max().item() == 1.0


def test_make_gauss_rectangular():
    expected = torch.tensor([
        [0.496683, 0.182719, 0.024728, 0.001231, 0.000023],
        [0.182719, 0.067219, 0.009097, 0.000453, 0.000008],
        [0.024728, 0.009097, 0.001231, 0.000061, 0.000001],
    ])
    actual = make_gauss(torch.tensor([-1.0, -1.0]), [3, 5], sigma=1.0)
    assert_allclose(actual, expected, rtol=0, atol=1e-5)
