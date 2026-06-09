import os
import sys
import time

import numpy as np

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.NFCT import *

import pykeops
from pykeops.numpy import LazyTensor

pykeops.test_numpy_bindings()
print(pykeops.config.gpu_available)
pykeops.set_verbose(True)

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([32, 16, 8], dtype="int32")  # 3d

M = 500
d = len(N)
Ns = np.prod(N)

X = np.ascontiguousarray((np.random.rand(3, M) * 0.5).T)
fhat = np.random.rand(Ns)

# test init and setting
plan = NFCT(N, M)
plan.x = X
plan.fhat = fhat

# test trafo and get execution time 
t0 = time.perf_counter()

plan.trafo()  # value is in plan.f

t1 = time.perf_counter()
t_nfct = t1 - t0

# Explicit calculation using NumPy
if d == 1:
    I = [[k] for k in range(0, N[0])]
elif d == 2:
    I = [[k, i] for k in range(0, N[0]) for i in range(0, N[1])]
elif d == 3:
    I = [
        [k, i, j]
        for k in range(0, N[0])
        for i in range(0, N[1])
        for j in range(0, N[2])
    ]


def cosine_product(x, i, d):
    result = 1
    for k in range(d):
        result *= np.cos(2 * np.pi * x[k] * i[k])
    return result


F = np.array([[cosine_product(X[j], I[l], d) for l in range(Ns)] for j in range(M)])
F_mat = np.asmatrix(F)

## Compute and get execution time
t0 = time.perf_counter()

f1 = F @ fhat

t1 = time.perf_counter()
t_np = t1 - t0

f2 = plan.f

# Using keops and GPU computing
I = np.array(I, dtype=np.float64)
X_i = LazyTensor(X[:, None, :])      # (M,1,d)
I_j = LazyTensor(I[None, :, :])      # (1,Ns,d)

Kx = (2*np.pi * X_i[0] * I_j[0]).cos()
Ky = (2*np.pi * X_i[1] * I_j[1]).cos()
Kz = (2*np.pi * X_i[2] * I_j[2]).cos()

K = Kx * Ky * Kz

fhat_j = LazyTensor(
    fhat.astype(np.float64)[None, :, None]
)

# Warmup (triggers compilation)
_ = (K * fhat_j).sum(axis=1)

# Timed run
t0 = time.perf_counter()

f_keops = (K * fhat_j).sum(axis=1)

t1 = time.perf_counter()
t_keops = t1 - t0

f_keops = np.asarray(f_keops).reshape(-1)

# Comparisons
print(f'Executions time: \n Explicit computation: {t_np} \n NFCT: {t_nfct} \n Using KeOps: {t_keops}')
norm_euclidean_traf = np.linalg.norm(f1 - plan.f)
norm_infinity_traf = np.linalg.norm(f1 - plan.f, np.inf)
print("Euclidean norm for trafo test:", norm_euclidean_traf)
print("Infinity norm: for trafo test", norm_infinity_traf)

assert (
    norm_euclidean_traf < 1e-10
), f"TEST FAILED: Euclidiean norm ({norm_euclidean_traf}) for trafo test is not less than 1e-10"
assert (
    norm_infinity_traf < 1e-10
), f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for trafo test is not less than 1e-10"

norm_euclidean_keops = np.linalg.norm(f_keops - plan.f)
norm_infinity_keops = np.linalg.norm(f_keops - plan.f, np.inf)
print("Euclidean norm for KeOps test:", norm_euclidean_keops)
print("Infinity norm: for keops test", norm_infinity_keops)

assert (
    norm_euclidean_keops < 1e-10
), f"TEST FAILED: Euclidiean norm ({norm_euclidean_keops}) for keops test is not less than 1e-10"
assert (
    norm_infinity_keops < 1e-10
), f"TEST FAILED: Infinity norm ({norm_infinity_keops}) for keops test is not less than 1e-10"

# test transpose
plan.nfct_transposed()
f1 = F_mat.H @ plan.f
f2 = plan.fhat

# get errors from transpose test
norm_euclidean_adj = np.linalg.norm(f1 - plan.fhat)
norm_infinity_adj = np.linalg.norm(f1 - plan.fhat, np.inf)
print("Euclidean norm for transpose test:", norm_euclidean_adj)
print("Infinity norm for transpose test:", norm_infinity_adj)
assert (
    norm_euclidean_adj < 1e-9
), f"TEST FAILED: Euclidiean norm ({norm_euclidean_adj}) for transpose test is not less than 1e-9"
assert (
    norm_infinity_adj < 1e-9
), f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for transpose test is not less than 1e-9"
