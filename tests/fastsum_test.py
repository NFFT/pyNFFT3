import numpy as np
import sys, os

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.fastsum import *

d = 2
N = 3
M = 5
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
