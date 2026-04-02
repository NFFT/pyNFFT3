import os
import sys

import numpy as np

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.NFMT import BASES, NFMT, nfmt_index_set_without_zeros


def getphi(basis_vect, x, k):
    p = 1.0 + 0.0j
    for idx, basis in enumerate(basis_vect):
        if BASES[basis] == 1:
            if k[idx] != 0:
                p *= np.sqrt(2.0) * np.cos(np.pi * k[idx] * x[idx])
        elif BASES[basis] == 2:
            if k[idx] != 0:
                p *= np.sqrt(2.0) * np.cos(k[idx] * np.arccos(2.0 * x[idx] - 1.0))
        elif BASES[basis] == 0:
            p *= np.exp(-2.0j * np.pi * k[idx] * x[idx])
        else:
            raise ValueError(f"Unknown basis: {basis}")
    return p


def _dim_frequency_values(basis, n_j):
    if BASES[basis] > 0:
        return list(range(0, int(n_j // 2)))
    half = int(n_j // 2)
    return list(range(-half, half))


def getk(basis_vect, bandwidths, i):
    d = len(basis_vect)
    vals = [_dim_frequency_values(basis_vect[j], bandwidths[j]) for j in range(d)]
    lens = [len(v) for v in vals]

    pv = np.ones(d, dtype=np.int64)
    for j in range(d - 2, -1, -1):
        pv[j] = pv[j + 1] * lens[j + 1]

    idx0 = i - 1
    k = np.zeros(d, dtype=np.int64)
    for j in range(d):
        pos = (idx0 // pv[j]) % lens[j]
        k[j] = vals[j][pos]
    return k


def getMat(bandwidths, M, X, basis_vect):
    n = nfmt_index_set_without_zeros(bandwidths, basis_vect)
    if n.ndim == 1:
        n = n.reshape(1, -1)
    n = n.shape[1]
    F = np.array(
        [[getphi(basis_vect, X[:, j], getk(basis_vect, bandwidths, l)) for l in range(1, n + 1)] for j in range(M)],
        dtype=np.complex128,
    )
    return F


N = np.array([20, 8, 12, 4], dtype=np.int32)
basis_vect = ["exp", "alg", "cos", "alg"]
M = 100

X_jl = np.random.rand(4, M)
X_jl[0, :] -= 0.5  # exponential dim in [-0.5, 0.5)
X = np.ascontiguousarray(X_jl.T)

nf = int(np.prod(N - 1))
a = sum(BASES[b] > 0 for b in basis_vect)
nf = int(np.prod(N) // (2**a))
fhat = (np.random.rand(nf) + 1j * np.random.rand(nf)).astype(np.complex128)

plan = NFMT(N, M, basis_vect)
plan.x = X
plan.fhat = fhat

# p * nfmt_get_coefficient_array(fhat, p)
f3 = plan.trafo()

# nfmt_get_LinearMap(...)
F = getMat(N, M, X_jl, basis_vect)
f4 = F @ fhat

# nfmt_trafo(p)
plan.nfmt_trafo()
f2 = plan.f

f1 = F @ fhat

error_vector = f1 - f2
norm_euclidean_traf = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_traf = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
print("Euclidean norm for trafo test:", norm_euclidean_traf)
print("Infinity norm for trafo test:", norm_infinity_traf)
assert (
    norm_euclidean_traf < 1e-12
), f"TEST FAILED: Euclidean norm ({norm_euclidean_traf}) for trafo test is not less than 1e-12"
assert (
    norm_infinity_traf < 1e-12
), f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for trafo test is not less than 1e-12"

error_vector = f1 - f3
norm_euclidean_traf = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_traf = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
assert (
    norm_euclidean_traf < 1e-12
), f"TEST FAILED: Euclidean norm ({norm_euclidean_traf}) for p*fhat test is not less than 1e-12"
assert (
    norm_infinity_traf < 1e-12
), f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for p*fhat test is not less than 1e-12"

error_vector = f1 - f4
norm_euclidean_traf = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_traf = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
assert (
    norm_euclidean_traf < 1e-12
), f"TEST FAILED: Euclidean norm ({norm_euclidean_traf}) for LinearMap test is not less than 1e-12"
assert (
    norm_infinity_traf < 1e-12
), f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for LinearMap test is not less than 1e-12"

# p' * p.f (coefficient vector path)
f3_adj = F.conj().T @ plan.f

# L' * p.f
f4_adj = F.conj().T @ plan.f

# nfmt_adjoint(p)
plan.nfmt_adjoint()
f2_adj = plan.fhat

f1_adj = F.conj().T @ plan.f
error_vector_adj = f1_adj - f2_adj
norm_euclidean_adj = np.linalg.norm(error_vector_adj) / np.linalg.norm(f1_adj)
norm_infinity_adj = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(plan.f, 1)
print("Euclidean norm for adjoint test:", norm_euclidean_adj)
print("Infinity norm for adjoint test:", norm_infinity_adj)
assert (
    norm_euclidean_adj < 1e-12
), f"TEST FAILED: Euclidean norm ({norm_euclidean_adj}) for adjoint test is not less than 1e-12"
assert (
    norm_infinity_adj < 1e-12
), f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for adjoint test is not less than 1e-12"

error_vector_adj = f1_adj - f3_adj
norm_euclidean_adj = np.linalg.norm(error_vector_adj) / np.linalg.norm(f1_adj)
norm_infinity_adj = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(plan.f, 1)
assert (
    norm_euclidean_adj < 1e-12
), f"TEST FAILED: Euclidean norm ({norm_euclidean_adj}) for p'*f test is not less than 1e-12"
assert (
    norm_infinity_adj < 1e-12
), f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for p'*f test is not less than 1e-12"

error_vector_adj = f1_adj - f4_adj
norm_euclidean_adj = np.linalg.norm(error_vector_adj) / np.linalg.norm(f1_adj)
norm_infinity_adj = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(plan.f, 1)
assert (
    norm_euclidean_adj < 1e-12
), f"TEST FAILED: Euclidean norm ({norm_euclidean_adj}) for L'*f test is not less than 1e-12"
assert (
    norm_infinity_adj < 1e-12
), f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for L'*f test is not less than 1e-12"
