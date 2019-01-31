import torch
from pytest import fixture

torch.set_default_tensor_type('torch.DoubleTensor')

# Get the slow CUDA initialisation out of the way
for i in range(torch.cuda.device_count()):
    torch.empty(0).to(torch.device('cuda', i))


@fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    yield
