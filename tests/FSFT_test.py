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
# (shared with NFSFT) unconditionally read/write plan->x for node positions,
# but FSFT never initializes x (there is no x setter - the whole point of
# NFSFT_EQUISPACED is that the grid is implicit in N). Below
# NFSFT_BREAK_EVEN (bandwidth 5, see kernel/nfsft/nfsft.c), the "fast"
# trafo()/adjoint() calls literally delegate to trafo_direct()/adjoint_direct()
# internally, so comparing them there is meaningful (also a reasonable
# end-to-end smoke test of the Python<->C bridge). At/above the break-even
# bandwidth, trafo()/adjoint() use a distinct FFT-based algorithm that never
# touches x at all, so there is currently no independent, trustworthy
# reference to validate it against from Python - only a smoke test (no
# NaN/Inf, deterministic across repeated calls) is done for that case.

# IMPORTANT ordering note: NFSFT/FSFT share one process-global "wisdom"
# precompute (see nfsft_precompute()/wisdom.initialized in
# kernel/nfsft/nfsft.c). It is computed once per process and sized for
# whichever bandwidth N is used *first* - nfsft_precompute() is a no-op on
# every later call, even for a larger N. Any later plan whose N exceeds that
# first N silently gets NaN from trafo()/adjoint() (wisdom.N_MAX check).
# So the larger bandwidth must run first in this file.

# --- at/above NFSFT_BREAK_EVEN (5): only a smoke test - trafo()/adjoint() use
# --- a distinct FFT-based algorithm here that never touches x at all, so
# --- there is currently no independent, trustworthy reference to validate it
# --- against from Python.
N = 6

plan = FSFT(N, 1)  # M is ignored/overridden by FSFT for the actual grid size
fhat = np.zeros((2 * plan.N + 2) ** 2, dtype=np.complex128)
for k in range(N + 1):
    for n in range(-k, k + 1):
        fhat[plan.fsft_index(k, n)] = (np.random.rand() - 0.5) + 1j * (
            np.random.rand() - 0.5
        )
plan.fhat = fhat

plan.trafo()
f_first = np.copy(plan.f)
assert np.all(np.isfinite(f_first)), "TRAFO TEST FAILED: fsft_trafo() produced NaN/Inf"

# nfsft_default does not set NFSFT_PRESERVE_F_HAT, so fsft_trafo() has just
# overwritten plan.fhat in place with internal FPT/FFT scratch data (expected
# behaviour of the underlying library, see simpleTest/example_nfsft.py) -
# fhat must be set again before trafo() can be trusted to reproduce the same
# result.
assert not np.array_equal(
    plan.fhat, fhat
), "fhat was unexpectedly left untouched by trafo() (NFSFT_PRESERVE_F_HAT not set)"
plan.fhat = fhat
plan.trafo()
f_second = np.copy(plan.f)
assert np.array_equal(
    f_first, f_second
), "TRAFO TEST FAILED: fsft_trafo() is not deterministic for identical input"

print(
    "fsft_trafo() smoke test (N >= NFSFT_BREAK_EVEN) passed: finite and deterministic"
)

# --- below NFSFT_BREAK_EVEN: trafo() literally calls trafo_direct() (see
# --- kernel/nfsft/nfsft.c), so comparing them here is meaningful - it doesn't
# --- validate the FFT-based path above, but does exercise the Python/C bridge
# --- end to end. Must run after the N=6 block above (see ordering note).
N_small = 3

plan_direct = FSFT(N_small, 1)
fhat_small = np.zeros((2 * plan_direct.N + 2) ** 2, dtype=np.complex128)
for k in range(N_small + 1):
    for n in range(-k, k + 1):
        fhat_small[plan_direct.fsft_index(k, n)] = (np.random.rand() - 0.5) + 1j * (
            np.random.rand() - 0.5
        )

plan_direct.fhat = fhat_small
plan_direct.trafo_direct()
f1_small = np.copy(plan_direct.f)

plan_fast = FSFT(N_small, 1)
plan_fast.fhat = fhat_small
plan_fast.trafo()
f2_small = np.copy(plan_fast.f)

E_2_small = np.linalg.norm(f1_small - f2_small) / np.linalg.norm(f1_small)
print("E_2 (trafo test, N < NFSFT_BREAK_EVEN): ", E_2_small)
assert (
    E_2_small < 1e-9
), f"TRAFO TEST FAILED: Euclidean norm ({E_2_small}) for small-N trafo test is not less than 1e-9"

# TODO: fsft_adjoint()/fsft_adjoint_direct() still need a trustworthy,
# independent reference before they can be tested properly - same
# uninitialized-x caveat as above, plus adjoint_direct() shares the
# trafo_direct() limitation.
