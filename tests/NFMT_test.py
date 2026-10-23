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


# NOTE: unlike the other pyNFFT3 plan classes, NFMT has no compiled C
# counterpart of its own - it reduces the mixed transform to a plain NFFT or
# NFCT (remapped nodes + expanded/scaled coefficients, see
# src/pyNFFT3/NFMT.py and NFFT3.jl's NFMT.jl). A complex fhat/f, or any exp
# dimension, picks the NFFT backend; a real fhat/f with every dimension
# cos/alg picks the cheaper, real-valued NFCT backend instead - see the
# "real-valued backend" block further down. Either way trafo()/adjoint()
# therefore run the real, approximate fast algorithm (not an exact
# computation), so the tolerances here are the usual NFFT/NFCT accuracy, not
# machine epsilon - trafo_direct()/adjoint_direct() run the real direct
# transform and are exact up to rounding.
N = np.array([20, 8, 12, 4], dtype=np.int32)
basis_vect = ["exp", "alg", "cos", "alg"]
M = 100

X_jl = np.random.rand(4, M)
X_jl[0, :] -= 0.5  # exponential dim in [-0.5, 0.5)
X = np.ascontiguousarray(X_jl.T)

a = sum(BASES[b] > 0 for b in basis_vect)
nf = int(np.prod(N) // (2**a))
fhat = (np.random.rand(nf) + 1j * np.random.rand(nf)).astype(np.complex128)

plan = NFMT(N, M, basis_vect)
plan.x = X
plan.fhat = fhat

# reference: dense matrix built independently from getphi/getk
F = getMat(N, M, X_jl, basis_vect)
f1 = F @ fhat

# fast trafo (real NFFT under the hood)
plan.trafo()
f2 = plan.f

error_vector = f1 - f2
norm_euclidean_traf = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_traf = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
print("Euclidean norm for fast trafo test:", norm_euclidean_traf)
print("Infinity norm for fast trafo test:", norm_infinity_traf)
assert (
    norm_euclidean_traf < 1e-6
), f"TEST FAILED: Euclidean norm ({norm_euclidean_traf}) for fast trafo test is not less than 1e-6"
assert (
    norm_infinity_traf < 1e-6
), f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for fast trafo test is not less than 1e-6"

# direct trafo (real NDFT under the hood) - exact up to rounding
plan.trafo_direct()
f3 = plan.f

error_vector = f1 - f3
norm_euclidean_direct = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_direct = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
print("Euclidean norm for direct trafo test:", norm_euclidean_direct)
print("Infinity norm for direct trafo test:", norm_infinity_direct)
assert (
    norm_euclidean_direct < 1e-9
), f"TEST FAILED: Euclidean norm ({norm_euclidean_direct}) for direct trafo test is not less than 1e-9"
assert (
    norm_infinity_direct < 1e-9
), f"TEST FAILED: Infinity norm ({norm_infinity_direct}) for direct trafo test is not less than 1e-9"

# adjoint: fast vs. direct, both against the dense reference's adjoint
f1_adj = F.conj().T @ f3

plan.adjoint()
f2_adj = plan.fhat

error_vector_adj = f1_adj - f2_adj
norm_euclidean_adj = np.linalg.norm(error_vector_adj) / np.linalg.norm(f1_adj)
norm_infinity_adj = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(plan.f, 1)
print("Euclidean norm for fast adjoint test:", norm_euclidean_adj)
print("Infinity norm for fast adjoint test:", norm_infinity_adj)
assert (
    norm_euclidean_adj < 1e-6
), f"TEST FAILED: Euclidean norm ({norm_euclidean_adj}) for fast adjoint test is not less than 1e-6"
assert (
    norm_infinity_adj < 1e-6
), f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for fast adjoint test is not less than 1e-6"

plan.adjoint_direct()
f3_adj = plan.fhat

error_vector_adj = f1_adj - f3_adj
norm_euclidean_adj_direct = np.linalg.norm(error_vector_adj) / np.linalg.norm(f1_adj)
norm_infinity_adj_direct = np.linalg.norm(error_vector_adj, np.inf) / np.linalg.norm(plan.f, 1)
print("Euclidean norm for direct adjoint test:", norm_euclidean_adj_direct)
print("Infinity norm for direct adjoint test:", norm_infinity_adj_direct)
assert (
    norm_euclidean_adj_direct < 1e-9
), f"TEST FAILED: Euclidean norm ({norm_euclidean_adj_direct}) for direct adjoint test is not less than 1e-9"
assert (
    norm_infinity_adj_direct < 1e-9
), f"TEST FAILED: Infinity norm ({norm_infinity_adj_direct}) for direct adjoint test is not less than 1e-9"

# --- real-valued backend: when every dimension is cos/alg (no exp dimension)
# --- and fhat/f is given as a real float64 array, NFMT realises the mixed
# --- transform with a plain NFCT instead of an NFFT (half the data, real
# --- arithmetic throughout) - see _all_cos_alg()/_ensure_backend() in
# --- src/pyNFFT3/NFMT.py. Exercise that path here against the same kind of
# --- independent dense reference used above.
N_real = np.array([8, 4], dtype=np.int32)
basis_real = ["cos", "alg"]
M_real = 60

X_real_jl = np.random.rand(2, M_real)
X_real = np.ascontiguousarray(X_real_jl.T)

nf_real = int(np.prod(N_real) // 4)
fhat_real = np.random.rand(nf_real)

plan_real = NFMT(N_real, M_real, basis_real)
plan_real.x = X_real
plan_real.fhat = fhat_real  # real dtype + all cos/alg -> picks the NFCT backend

F_real = getMat(N_real, M_real, X_real_jl, basis_real)
f1_real = (F_real @ fhat_real.astype(np.complex128)).real

plan_real.trafo()
assert plan_real.f.dtype == np.float64, "real fhat + all cos/alg should yield real f (NFCT backend)"
err_real = np.linalg.norm(plan_real.f - f1_real) / np.linalg.norm(f1_real)
print("Euclidean norm for real-backend (NFCT) fast trafo test:", err_real)
assert (
    err_real < 1e-6
), f"TEST FAILED: relative error ({err_real}) for real-backend fast trafo test is not less than 1e-6"

plan_real.trafo_direct()
err_real_direct = np.linalg.norm(plan_real.f - f1_real) / np.linalg.norm(f1_real)
print("Euclidean norm for real-backend (NFCT) direct trafo test:", err_real_direct)
assert (
    err_real_direct < 1e-9
), f"TEST FAILED: relative error ({err_real_direct}) for real-backend direct trafo test is not less than 1e-9"

f1_real_adj = (F_real.conj().T @ f1_real.astype(np.complex128)).real
plan_real.adjoint()
assert plan_real.fhat.dtype == np.float64, "adjoint on the NFCT backend should yield real fhat"
err_real_adj = np.linalg.norm(plan_real.fhat - f1_real_adj) / np.linalg.norm(f1_real_adj)
print("Euclidean norm for real-backend (NFCT) fast adjoint test:", err_real_adj)
assert (
    err_real_adj < 1e-6
), f"TEST FAILED: relative error ({err_real_adj}) for real-backend fast adjoint test is not less than 1e-6"

# a complex-dtype fhat on an all-cos/alg plan still uses the general NFFT path
plan_mixed_dtype = NFMT(N_real, M_real, basis_real)
plan_mixed_dtype.x = X_real
fhat_complex = (np.random.rand(nf_real) + 1j * np.random.rand(nf_real)).astype(np.complex128)
plan_mixed_dtype.fhat = fhat_complex
plan_mixed_dtype.trafo()
assert plan_mixed_dtype.f.dtype == np.complex128, "complex fhat should always use the NFFT backend"
f1_complex = F_real @ fhat_complex
err_complex = np.linalg.norm(plan_mixed_dtype.f - f1_complex) / np.linalg.norm(f1_complex)
print("Euclidean norm for all-cos/alg plan with complex fhat (NFFT backend) trafo test:", err_complex)
assert (
    err_complex < 1e-6
), f"TEST FAILED: relative error ({err_complex}) for complex-on-cos/alg trafo test is not less than 1e-6"

# once a plan has picked the real NFCT backend, complex data must be rejected
try:
    plan_real.fhat = fhat_complex
    raise AssertionError("TEST FAILED: assigning complex fhat to an NFCT-backed plan did not raise")
except RuntimeError:
    pass
