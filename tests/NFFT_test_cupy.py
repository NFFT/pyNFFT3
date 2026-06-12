import os
import sys
import time

import numpy as np

import cupy as cp

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.NFFT import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([64, 32, 16], dtype="int32")  # 3d

M = 1000
d = len(N)
Ns = np.prod(N)

X = np.ascontiguousarray((np.random.rand(3, M) - 0.5).T)
fhat = np.random.rand(Ns) + 1.0j * np.random.rand(Ns)

# declare data for GPU usage

X_gpu = cp.asarray(X)
fhat_gpu = cp.asarray(fhat)

k0 = cp.arange(-N[0]//2, N[0]//2)
k1 = cp.arange(-N[1]//2, N[1]//2)
k2 = cp.arange(-N[2]//2, N[2]//2)

t0 = time.perf_counter()
I_gpu = cp.stack(
    cp.meshgrid(k0, k1, k2, indexing="ij"),
    axis=-1
).reshape(-1, 3)

dots = X_gpu @ I_gpu.T

F_gpu = cp.exp(-2j * cp.pi * dots)

f3 = F_gpu @ fhat_gpu
cp.cuda.Stream.null.synchronize()

t1 = time.perf_counter()
t_gpu = t1 - t0

# test init and setting
t0 = time.perf_counter()

plan = NFFT(N, M)
plan.x = X
plan.fhat = fhat

plan.trafo()  # value is in plan.f

t1 = time.perf_counter()
t_nfft = t1 - t0

# compare with directly computed
t0 = time.perf_counter()

if d == 1:
    I = [[k] for k in range(int(-N[0] / 2), int(N[0] / 2))]
elif d == 2:
    I = [
        [k, i]
        for k in range(int(-N[0] / 2), int(N[0] / 2))
        for i in range(int(-N[1] / 2), int(N[1] / 2))
    ]
elif d == 3:
    I = [
        [k, i, j]
        for k in range(int(-N[0] / 2), int(N[0] / 2))
        for i in range(int(-N[1] / 2), int(N[1] / 2))
        for j in range(int(-N[2] / 2), int(N[2] / 2))
    ]

F = np.array(
    [
        [np.exp(-2 * np.pi * 1j * np.dot(X.T[:, j], I[l])) for l in range(0, Ns)]
        for j in range(0, M)
    ]
)
F_mat = np.asmatrix(F)
f1 = F @ fhat
t1 = time.perf_counter()
t_cpu = t1 - t0


# get errors from trafo test
f2 = plan.f
f3 = cp.asnumpy(f3)
# Comparisons 
print(f'Executions time: \n Explicit computation: {t_cpu} \n NFFT: {t_nfft} \n Using CuPy: {t_gpu}')

error_vector = f1 - f2
norm_euclidean_traf = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_traf = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
print("Euclidean norm for trafo test:", norm_euclidean_traf)
print("Infinity norm: for trafo test", norm_infinity_traf)
assert (
    norm_euclidean_traf < 1e-10
), f"TEST FAILED: Euclidiean norm ({norm_euclidean_traf}) for trafo test is not less than 1e-10"
assert (
    norm_infinity_traf < 1e-10
), f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for trafo test is not less than 1e-10"

error_vector_gpu = f1 - f3
norm_euclidean_traf_gpu = np.linalg.norm(error_vector_gpu) / np.linalg.norm(f1)
norm_infinity_traf_gpu = np.linalg.norm(error_vector_gpu, np.inf) / np.linalg.norm(fhat, 1)
print("Euclidean norm for trafo test (GPU):", norm_euclidean_traf_gpu)
print("Infinity norm for trafo test (GPU)", norm_infinity_traf_gpu)
assert (
    norm_euclidean_traf_gpu < 1e-10
), f"TEST FAILED: Euclidiean norm ({norm_euclidean_traf_gpu}) for trafo test is not less than 1e-10"
assert (
    norm_infinity_traf_gpu < 1e-10
), f"TEST FAILED: Infinity norm ({norm_infinity_traf_gpu}) for trafo test is not less than 1e-10"