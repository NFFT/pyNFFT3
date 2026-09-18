import os
import sys

import numpy as np

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.fastsum import *
from pyNFFT3.flags import *

d = 2
N = 20000
M = 20000
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

# Test trafo
plan.fastsum_trafo()
f1 = np.copy(plan.f)

# Test trafo exact
plan.fastsum_trafo_exact()
f2 = np.copy(plan.f)

# Calculate the error vector
error_vector = f1 - f2

# Calculate and print norms
E_2 = np.linalg.norm(error_vector) / np.linalg.norm(f1)
E_infty = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(plan.alpha, 1)
print("E_2: ", E_2)
print("E_infty: ", E_infty)

assert (
    E_2 < 1e-5
), f"TEST FAILED: Euclidiean norm ({E_2}) for trafo test is not less than 1e-5"
assert (
    E_infty < 1e-5
), f"TEST FAILED: Infinity norm ({E_infty}) for trafo test is not less than 1e-5"

# Regression test for d=1 (the x/y setters used to crash with an AxisError
# for 1D coordinate arrays, since np.linalg.norm(..., axis=1) requires ndim>=2)
d1 = 1
N1 = 200
M1 = 200
kernel1 = "gaussian"
c1 = 1 / np.sqrt(N1)

plan1 = FASTSUM(d1, N1, M1, kernel1, c1, eps_B=eps_B)

x1 = (np.random.rand(N1) - 0.5) * (0.25 - eps_B / 2)
plan1.x = x1

alpha1 = np.random.rand(N1) + 1j * np.random.rand(N1)
plan1.alpha = alpha1

y1 = (np.random.rand(M1) - 0.5) * (0.25 - eps_B / 2)
plan1.y = y1

plan1.fastsum_trafo()
f1_1d = np.copy(plan1.f)

plan1.fastsum_trafo_exact()
f2_1d = np.copy(plan1.f)

error_vector_1d = f1_1d - f2_1d
E_2_1d = np.linalg.norm(error_vector_1d) / np.linalg.norm(f1_1d)
print("E_2 (d=1): ", E_2_1d)
assert (
    E_2_1d < 1e-5
), f"TEST FAILED: Euclidiean norm ({E_2_1d}) for d=1 trafo test is not less than 1e-5"
