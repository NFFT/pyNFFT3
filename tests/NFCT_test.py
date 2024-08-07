import numpy as np
import sys, os

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.NFCT import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([16, 8, 4], dtype="int32")  # 3d

M = 100
d = len(N)
Ns = np.prod(N)

X = np.ascontiguousarray((np.random.rand(3, M) * 0.5).T)
fhat = np.random.rand(Ns)

# test init and setting
plan = NFCT(N, M)
plan.x = X
plan.fhat = fhat

# test trafo
plan.trafo()  # value is in plan.f

# compare with directly computed, using N[k]-1 but range is not inclusive
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

# get errors from trafo test
f1 = F @ fhat
f2 = plan.f
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
