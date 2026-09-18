import os
import sys

import numpy as np

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.NFSFT import *

N = 100
M = 100

# pseudo-random nodes
X = np.random.rand(2, M)
X[0, :] -= 0.5
X[1, :] *= 0.5

# NOTE: each operation below uses its own, freshly initialized plan. The
# underlying NFSFT library corrupts its internal state if a *_direct()
# method is called on the same plan after trafo()/adjoint() has already run
# on it (returns silently wrong results, or worse) - so direct and fast
# variants must never share a plan.

# test trafo direct
plan_direct = NFSFT(N, M)
plan_direct.x = X

# generate pseudo-random Fourier coefficients
fhat = np.zeros((2 * plan_direct.N + 2) ** 2, dtype=np.complex128)
for k in range(N + 1):
    for n in range(-k, k + 1):
        index = plan_direct.nfsft_index(k, n)
        fhat[index] = (np.random.rand() - 0.5) + 1j * (np.random.rand() - 0.5)

plan_direct.fhat = fhat
plan_direct.trafo_direct()
f1 = np.copy(plan_direct.f)

# test trafo
plan_fast = NFSFT(N, M)
plan_fast.x = X
plan_fast.fhat = fhat
plan_fast.trafo()
f2 = np.copy(plan_fast.f)

# test adjoint direct (using f2 as input, like the trafo() output would be used downstream)
plan_adj_direct = NFSFT(N, M)
plan_adj_direct.x = X
plan_adj_direct.f = f2
plan_adj_direct.adjoint_direct()
f3 = np.zeros_like(plan_adj_direct.fhat, dtype=np.complex128)
for k in range(plan_adj_direct.N + 1):
    for n in range(-k, k + 1):
        index = plan_adj_direct.nfsft_index(k, n)
        f3[index] = plan_adj_direct.fhat[index]

# test fast approximate adjoint
plan_adj_fast = NFSFT(N, M)
plan_adj_fast.x = X
plan_adj_fast.f = f2
plan_adj_fast.adjoint()
f4 = np.zeros_like(plan_adj_fast.fhat, dtype=np.complex128)
for k in range(0, plan_adj_fast.N + 1):
    for n in range(-k, k + 1):
        index = plan_adj_fast.nfsft_index(k, n)
        f4[index] = plan_adj_fast.fhat[index]

# calculate the error vectors
error_vector_traf = f1 - f2
error_vector_adj = f3 - f4

# calculate and print norms for trafo test
E_2_traf = np.linalg.norm(error_vector_traf) / np.linalg.norm(f1)
E_infty_traf = np.linalg.norm(error_vector_traf, np.inf) / np.linalg.norm(fhat, 1)
print("E_2 (trafo test): ", E_2_traf)
print("E_infty (trafo test): ", E_infty_traf)
assert (
    E_2_traf < 1e-9
), f"TRAFO TEST FAILED: Euclidiean norm ({E_2_traf}) for trafo test is not less than 1e-9"
assert (
    E_infty_traf < 1e-9
), f"TRAFO TEST FAILED: Infinity norm ({E_infty_traf}) for trafo test is not less than 1e-9"

# calculate and print norms for adjoint test
E_2_adj = np.linalg.norm(error_vector_adj) / np.linalg.norm(f3)
E_infty_adj = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(f2, 1)
print("E_2 (adjoint test): ", E_2_adj)
print("E_infty (adjoint test): ", E_infty_adj)
assert (
    E_2_adj < 1e-9
), f"ADJOINT TEST FAILED: Euclidiean norm ({E_2_adj}) for adjoint test is not less than 1e-9"
assert (
    E_infty_adj < 1e-9
), f"ADJOINT TEST FAILED: Infinity norm ({E_infty_adj}) for adjoint test is not less than 1e-9"
