import numpy as np
from src.pyNFFT3.flags import *
from src.pyNFFT3.NFCT import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([16, 8, 4], dtype='int32') # 3d

M = 100
d = len(N)
Ns = np.prod(N)

X = np.array([[abs(np.sin(i + j)) for j in range(d)] for i in range(M)])
fhat = np.array([np.cos(k) * np.sin(k) for k in range(Ns)], dtype=np.float64)
f = np.array([np.sin(m) * np.cos(m) for m in range(M)])

# test init and setting
plan_traf = NFCT(N,M)
plan_traf.x = X
plan_traf.f = f # this gets overwritten
plan_traf.fhat = fhat

plan_adj = NFCT(N,M)
plan_adj.x = X
plan_adj.f = f # this gets overwritten
plan_adj.fhat = fhat

# test trafo
plan_traf.trafo() # value is in plan.f

# test transpose
plan_adj.nfct_transposed()

# compare with directly computed, using N[k]-1 but range is not inclusive
if d == 1:
    I = [[k] for k in range(0, N[0])]
elif d == 2:
    I = [[k, i] for k in range(0, N[0]) for i in range(0, N[1])]
elif d == 3:
    I = [[k, i, j] for k in range(0, N[0]) for i in range(0, N[1]) for j in range(0, N[2])]

def cosine_product(x, i, d):
    result = 1
    for k in range(d):
        result *= np.cos(2 * np.pi * x[k] * i[k])
    return result

F = np.array([[cosine_product(X[j], I[l], d) for l in range(Ns)] for j in range(M)])
F_mat = np.asmatrix(F)

## norm values should be ~e-12
# for testing trafo
f1 = F @ fhat
norm_euclidean_traf = np.linalg.norm(f1 - plan_traf.f)
norm_infinity_traf = np.linalg.norm(f1 - plan_traf.f, np.inf)
print("Euclidean norm for trafo test:", norm_euclidean_traf)
print("Infinity norm: for trafo test", norm_infinity_traf)

# for testing transpose
f1 = F_mat.H @ f
norm_euclidean_adj = np.linalg.norm(f1 - plan_adj.fhat)
norm_infinity_adj = np.linalg.norm(f1 - plan_adj.fhat, np.inf)
print("Euclidean norm for transpose test:", norm_euclidean_adj)
print("Infinity norm for transpose test:", norm_infinity_adj)

assert norm_euclidean_traf < 1e-10, f"TEST FAILED: Euclidiean norm ({norm_euclidean_traf}) for trafo test is not less than 1e-10"
assert norm_infinity_traf < 1e-10, f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for trafo test is not less than 1e-10"

assert norm_euclidean_adj < 1e-10, f"TEST FAILED: Euclidiean norm ({norm_euclidean_adj}) for transpose test is not less than 1e-10"
assert norm_infinity_adj < 1e-10, f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for transpose test is not less than 1e-10"