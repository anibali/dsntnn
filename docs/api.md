# API docs

## Terminology

### Normalized units

Sizes and coordinates may be specified in terms of pixels or "normalized units".
In normalized units, the top-left corner of a 2D image is (-1, -1) and the bottom-right
corner is (1, 1).

The equation for converting an x-coordinate from pixels to normalized units is

$$
x_{norm} = \dfrac{2x_{px} + 1}{width} - 1
$$

For sizes/distances, it's simply

$$
d_{norm} = \dfrac{2d_{px}}{width}
$$

### Tensor dimensions

The following variables are used when denoting tensor sizes.

* `B` — batch size
* `L` — number of locations per image
* `D` — the number of coordinate dimensions (2)
* `H` — height
* `W` — width

Optional dimensions are denoted with square brackets (e.g. `[B] x L` denotes an optional
first dimension).

## Heatmap normalization

```python
dsntnn.flat_softmax(tensor)
```

Calculates the softmax as if the last D dimensions were flattened.

## DSNT

```python
dsntnn.dsnt(heatmaps)
```

## Euclidean loss

```python
dsntnn.euclidean_losses(actual, target)
```

## Regularization

There are multiple regularization strategies included in this package.
We have found `js_reg_losses` to perform best in practice.

### Divergence regularization

```python
dsntnn.js_reg_losses(heatmaps, mu_t, sigma_t) # Jensen-Shannon divergence
dsntnn.kl_reg_losses(heatmaps, mu_t, sigma_t) # Kullback-Leibler divergence
```

Calculates the divergence between `heatmaps` and spherical Gaussians
with standard deviation `sigma_t` and means `mu_t`.

Arguments

* `heatmaps ([B] x L x H x W tensor)` — the predicted heatmaps
* `mu_t ([B] x L x D tensor)` — the ground truth location coordinates, in normalized units
* `sigma_t (float)` — the target standard deviation, in pixels

### Variance regularization

```python
dsntnn.variance_reg_losses(heatmaps, sigma_t)
```

Calculates the mean-square-error between the variance of `heatmaps` and `sigma_t ** 2`.

Note that this implementation works in pixel space, which is different from
the version used in the DSNT paper. The losses returned by this function
will be larger by a constant factor based on the size of the heatmap.

Arguments

* `heatmaps ([B] x L x H x W tensor)` — the predicted heatmaps
* `sigma_t (float)` — the target standard deviation, in pixels

## Utility

### Average loss

Combines per-location losses into an average loss by taking the mean.

If the `mask` argument is specified, it must be a binary mask.
Locations where `mask` has value 0 will be excluded from the overall loss.

```python
dsntnn.average_loss(losses, mask=None)
```

Arguments

* `losses ([B] x L tensor)` — per-location losses
* `mask ([B] x L tensor, optional)` — binary mask of included locations

Returns

* `loss (float)` — the combined loss
