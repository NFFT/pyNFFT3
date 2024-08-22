import numpy as np
import sys, os

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.NFSFT import *

N = 100
M = 100

# pseudo-random nodes
X = np.random.rand(2,M)
X[0,:] -= 0.5
X[1,:] *= 0.5

# test init and setting x
plan = NFSFT(N, M)
plan.x = X

# generate pseudo-random Fourier coefficients
fhat = np.zeros((2*plan.N+2)**2, dtype=np.complex128)
for k in range(N + 1):
    for n in range(-k, k + 1):
        index = plan.nfsft_index(k, n)
        fhat[index] = (np.random.rand() - 0.5) + 1j * (np.random.rand() - 0.5)
plan.fhat = fhat

# test trafo direct
plan.trafo_direct()
f1 = np.copy(plan.f)
# print("Vector f (NDSFT):")
# for j in range(M):
#     print(f"f[{j:+2d}] = {plan.f[j].real:5.3f} {plan.f[j].imag:+5.3f}*I")

# test trafo
plan.trafo()
f2 = np.copy(plan.f)
# print("Vector f (NFSFT):")
# for j in range(M):
#     print(f"f[{j:+2d}] = {plan.f[j].real:5.3f} {plan.f[j].imag:+5.3f}*I")

# test adjoint direct
plan.adjoint_direct()
f3 = np.zeros_like(plan.fhat, dtype=np.complex128)
# print("Vector fhat (NDSFT):")
for k in range(plan.N + 1):
    for n in range(-k, k + 1):
        index = plan.nfsft_index(k, n)
        # print(f"fhat[{k:+2d},{n:+2d}] = {plan.fhat[index].real:5.8f} {plan.fhat[index].imag:+5.8f}*I")
        f3[index] = plan.fhat[index]

# test fast approximate adjoint
plan.adjoint()
f4 = np.zeros_like(plan.fhat, dtype=np.complex128)
# print("Vector fhat (NFSFT):")
for k in range(0, plan.N + 1):
    for n in range(-k, k + 1):
        index = plan.nfsft_index(k, n)
        # print(f"fhat[{k:+2d},{n:+2d}] = {plan.fhat[index].real:5.8f} {plan.fhat[index].imag:+5.8f}*I")
        f4[index] = plan.fhat[index]

# calculate the error vectors
error_vector_traf = f1 - f2
error_vector_adj = f3 - f4

# calculate and print norms for trafo test
E_2_traf = np.linalg.norm(error_vector_traf) / np.linalg.norm(f1)
E_infty_traf = np.linalg.norm(error_vector_traf, np.inf) / np.linalg.norm(plan.fhat, 1)
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
E_infty_adj = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(plan.f, 1)
print("E_2 (adjoint test): ", E_2_adj)
print("E_infty (adjoint test): ", E_infty_adj)
assert (
    E_2_adj < 1e-9
), f"ADJOINT TEST FAILED: Euclidiean norm ({E_2_adj}) for adjoint test is not less than 1e-9"
assert (
    E_infty_adj < 1e-9
), f"ADJOINT TEST FAILED: Infinity norm ({E_infty_adj}) for adjoint test is not less than 1e-9"