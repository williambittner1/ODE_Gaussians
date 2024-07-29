import torch

# Define the dimensions of the tensor
shape = (3, 3, 5)  # Example shape: (depth, height, width)

# Create a 3D tensor with random values
tensor_random = torch.rand(shape)

print(tensor_random)
