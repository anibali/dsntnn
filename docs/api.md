# API docs

## Terminology

### Normalized units

Sizes and coordinates may be specified in terms of pixels or "normalized units".
In normalized units, the top-left corner of the image is (-1, -1) and the bottom-right
corner is (1, 1).

The equation for converting an x-coordinate from pixels to normalized units is

$$
x_{norm} = \dfrac{2x_{px} + 1}{width} - 1
$$

For sizes/distances, it's simply

$$
d_{norm} = \dfrac{2d_{px}}{width}
$$

### Dimensions

The following variables are used when denoting tensor sizes.

* `B` — batch size
* `L` — number of locations per image
* `H` — height
* `W` — width

Optional dimensions are denoted with square brackets (e.g. `[B] x L` denotes an optional
first dimension).

## Heatmap normalization

```python
dsntnn.softmax_2d(tensor)
```

## DSNT

```python
dsntnn.dsnt(heatmaps)
```

## Main loss term

### Euclidean loss

```python
dsntnn.euclidean_loss(actual, target, mask=None)
```

## Regularization

There are multiple regularization strategies included in this package.
We have found `js_reg_loss` to perform best in practice.

### Divergence regularization

```python
dsntnn.js_reg_loss(heatmaps, mu_t, sigma_t, mask=None) # Jensen-Shannon divergence
dsntnn.kl_reg_loss(heatmaps, mu_t, sigma_t, mask=None) # Kullback-Leibler divergence
```

Calculates the divergence between `heatmaps` and 2D spherical Gaussians
with standard deviation `sigma_t` means `mu_t`.

Arguments

* `heatmaps ([B] x L x H x W tensor)` — the predicted heatmaps
* `mu_t ([B] x L x 2 tensor)` — the ground truth location coordinates, in normalized units
* `sigma_t (float)` — the target standard deviation, in normalized units
* `mask ([B] x L tensor)` — location mask (see [Masked average](#masked-average))

## Utility

### Masked average

Combines per-location losses into an overall loss by taking the mean.

If the `mask` argument is specified, it must be a binary mask.
Locations where `mask` has value 0 will be excluded from the overall loss.

```python
dsntnn.masked_average(losses, mask=None)
```

Arguments

* `losses ([B] x L tensor)` — per-location losses
* `mask ([B] x L tensor, optional)` — binary mask of included locations

Returns

* `float` — the combined loss
