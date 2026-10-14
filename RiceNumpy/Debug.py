import numpy as np
from Layers.Conv import Conv2D

np.random.seed(0)

conv = Conv2D(
    in_channels=3,
    out_channels=2,
    kernel_size=3,
    padding=1
)

x = np.random.randn(2,3,5,5)

# forward
out = conv.forward(x)

# fake upstream gradient
dout = np.random.randn(*out.shape)

# backward
dx = conv.backward(dout)

# numerical dx check
eps = 1e-5

idx = (0,0,0,0)

original = x[idx]

x[idx] = original + eps
plus = np.sum(conv.forward(x) * dout)

x[idx] = original - eps
minus = np.sum(conv.forward(x) * dout)

x[idx] = original

numerical_dx = (plus - minus) / (2 * eps)

print("Analytical dx:", dx[idx])
print("Numerical dx :", numerical_dx)