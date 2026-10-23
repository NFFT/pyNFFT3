import os
import sys

import numpy as np

# Ensure src directory is in the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join((os.path.dirname(__file__)), "..", "src"))
)

from pyNFFT3.flags import *
from pyNFFT3.FSFT import *

# NOTE on FSFT and *_direct(): nfsft_trafo_direct()/nfsft_adjoint_direct()
# (shared with NFSFT) unconditionally read plan->x for node positions. FSFT
# has no x setter of its own (the whole point of NFSFT_EQUISPACED is that the
# grid is implicit in N) - fsft_init() now sets it to the equispaced grid
# itself (mirroring the "Set equispaced nodes" code in nfsft_init_guru(),
# kernel/nfsft/nfsft.c), since the bundled library does not reliably do this
# on its own (verified: two separately-constructed FSFT plans with the same N
# used to disagree on trafo_direct() output most of the time before this
# fix, 0/60 after). That makes trafo_direct()/adjoint_direct() trustworthy
# cross-plan references for any N, not just a same-plan smoke test.

# IMPORTANT ordering note: NFSFT/FSFT share one process-global "wisdom"
# precompute (see nfsft_precompute()/wisdom.initialized in
# kernel/nfsft/nfsft.c). It is computed once per process and sized for
# whichever bandwidth N is used *first* - nfsft_precompute() is a no-op on
# every later call, even for a larger N. Any later plan whose N exceeds that
# first N silently gets NaN from trafo()/adjoint() (wisdom.N_MAX check).
# So the larger bandwidth must run first in this file.


def random_fhat(plan, N):
    fhat = np.zeros((2 * plan.N + 2) ** 2, dtype=np.complex128)
    for k in range(N + 1):
        for n in range(-k, k + 1):
            fhat[plan.fsft_index(k, n)] = (np.random.rand() - 0.5) + 1j * (
                np.random.rand() - 0.5
            )
    return fhat


def valid_entries(plan, N):
    """
    Only indices (k, n) with |n| <= k <= N correspond to actual spherical
    Fourier coefficients; without NFSFT_ZERO_F_HAT (not part of the default
    flags), adjoint()/adjoint_direct() leave every other entry in the padded
    N_total-sized fhat array as unspecified scratch content, which the two
    algorithms are not expected to agree on (see NFSFT_ZERO_F_HAT in
    src/pyNFFT3/flags.py) - so comparisons must only look at these indices,
    exactly like tests/NFSFT_test.py does for NFSFT.
    """
    out = np.zeros_like(plan.fhat)
    for k in range(N + 1):
        for n in range(-k, k + 1):
            index = plan.fsft_index(k, n)
            out[index] = plan.fhat[index]
    return out


def check(label, a, b, tol):
    err = np.linalg.norm(a - b) / np.linalg.norm(a)
    print(f"{label}: {err}")
    assert err < tol, f"TEST FAILED: {label} = {err} is not less than {tol}"


# --- at/above NFSFT_BREAK_EVEN (5): trafo()/adjoint() use a distinct,
# --- FFT-based fast algorithm here, genuinely independent of trafo_direct().
N = 6

plan = FSFT(N, 1)  # M is ignored/overridden by FSFT for the actual grid size
fhat = random_fhat(plan, N)
plan.fhat = fhat

plan.trafo_direct()
f1 = np.copy(plan.f)

# nfsft_default does not set NFSFT_PRESERVE_F_HAT, so fsft_trafo() overwrites
# plan.fhat in place with internal FPT/FFT scratch data (expected behaviour
# of the underlying library, see simpleTest/example_nfsft.py) - so
# trafo_direct() (which only reads fhat) must run before trafo() on a shared
# plan, as done here.
plan.trafo()
f2 = np.copy(plan.f)
check("E_2 (trafo test, N >= NFSFT_BREAK_EVEN)", f1, f2, 1e-9)

plan.adjoint_direct()
f3 = valid_entries(plan, N)

plan.adjoint()
f4 = valid_entries(plan, N)
check("E_2 (adjoint test, N >= NFSFT_BREAK_EVEN)", f3, f4, 1e-9)

# --- below NFSFT_BREAK_EVEN: trafo()/adjoint() literally call
# --- trafo_direct()/adjoint_direct() internally (see kernel/nfsft/nfsft.c),
# --- so this is an exact-agreement check, not just an accuracy check. Must
# --- run after the N=6 block above (see ordering note).
N_small = 3

plan_small = FSFT(N_small, 1)
fhat_small = random_fhat(plan_small, N_small)
plan_small.fhat = fhat_small

plan_small.trafo_direct()
f1_small = np.copy(plan_small.f)

plan_small.trafo()
f2_small = np.copy(plan_small.f)
assert np.array_equal(
    f1_small, f2_small
), "TRAFO TEST FAILED: fsft_trafo() does not exactly match fsft_trafo_direct() for N < NFSFT_BREAK_EVEN"

plan_small.adjoint_direct()
f3_small = valid_entries(plan_small, N_small)

plan_small.adjoint()
f4_small = valid_entries(plan_small, N_small)
assert np.array_equal(
    f3_small, f4_small
), "ADJOINT TEST FAILED: fsft_adjoint() does not exactly match fsft_adjoint_direct() for N < NFSFT_BREAK_EVEN"

print("FSFT trafo/adjoint tests (both N >= and N < NFSFT_BREAK_EVEN) passed")
