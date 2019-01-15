# Basic DSNT usage guide

_You can use [Pweave](http://mpastell.com/pweave/) to execute the code in this
document and produce a HTML report._

```python
import torch
torch.manual_seed(12345)
```

## Building a coordinate regression model

```python
from torch import nn
import dsntnn
```

The bulk of the model can be any sort of fully convolutional network (FCN).
Here we'll just use a custom network with three convolutional layers.

```python
class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
```

Using the DSNT layer, we can very simply extend any FCN to tackle
coordinate regression tasks.

```python
class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps
```

## Training the model

```python
from torch import optim
import matplotlib.pyplot as plt
import scipy.misc
```

To demonstrate the model in action, we're going to train on an image of a
raccoon's eye.

```python
image_size = [40, 40]
raccoon_face = scipy.misc.imresize(scipy.misc.face()[200:400, 600:800, :], image_size)
eye_x, eye_y = 24, 26

plt.imshow(raccoon_face)
plt.scatter([eye_x], [eye_y], color='red', marker='X')
plt.show()
```

The input and target need to be put into PyTorch tensors. Importantly,
the target coordinates are normalized so that they are in the range (-1, 1).
The DSNT layer always outputs coordinates in this range.

```python
raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = input_tensor.cuda()

eye_coords_tensor = torch.Tensor([[[eye_x, eye_y]]])
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
target_var = target_tensor.cuda()

print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))
```

The coordinate regression model needs to be told ahead of time how many
locations to predict coordinates for. In this case we'll intantiate a
model to output 1 location per image.

```python
model = CoordRegressionNetwork(n_locations=1).cuda()
```

Doing a forward pass with the model is the same as with any PyTorch model.
The results aren't very good yet since the model is completely untrained.

```python
coords, heatmaps = model(input_var)

print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()
```

Now we'll train the model to overfit the location of the eye. Of course,
for real applications the model should be trained and evaluated using
separate training and validation datasets!

```python
optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)

for i in range(400):
    # Forward pass
    coords, heatmaps = model(input_var)

    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
    # Combine losses into an overall loss
    loss = dsntnn.average_loss(euc_losses + reg_losses)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters with RMSprop
    optimizer.step()

# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()
```
