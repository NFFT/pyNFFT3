import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.fastsum import *

d = 2
N = 5
M = 10
kernel = "multiquadric"
c = 1 / np.sqrt(N)
eps_B = 1 / 16

# Create a FASTSUM object in Python
plan = FASTSUM(d, N, M, kernel, c)

# Generate source nodes in circle of radius 0.25 - eps_B / 2
r = np.sqrt(np.random.rand(N)) * (0.25 - eps_B / 2)
phi = np.random.rand(N) * (2 * np.pi)
X = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
plan.x = X

# Generate coefficients alpha_k
alpha = np.random.rand(N) + 1j * np.random.rand(N)
plan.alpha = alpha

# Generate target nodes in circle of radius 0.25 - eps_B / 2
r = np.sqrt(np.random.rand(M)) * (0.25 - eps_B / 2)
phi = np.random.rand(M) * (2 * np.pi)
Y = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
plan.y = Y

print("X: ",plan.x)
print("Y: ",plan.y)
print("alpha: ",plan.alpha)

# Test trafo
plan.fastsum_trafo()
f1 = np.copy(plan.f)
print("f1: ",f1)

# Test trafo exact
plan.fastsum_trafo_exact()
f2 = np.copy(plan.f)
print("f2: ",f2)

# Calculate the error vector
error_vector = f1 - f2
print("Error vector: ", error_vector)

# Calculate the norms
E_2 = np.linalg.norm(error_vector) / np.linalg.norm(f1)
E_infty = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(plan.alpha, 1)

# Print the errors
print("E_2: ", E_2)
print("E_infty: ", E_infty)