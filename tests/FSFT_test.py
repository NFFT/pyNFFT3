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
# but FSFT never initializes x itself (there is no x setter - the whole point
# of NFSFT_EQUISPACED is that the grid is implicit in N). The underlying C
# library is documented to auto-fill plan->x with the equispaced grid during
# init precisely so trafo_direct() still works (see the "Set equispaced
# nodes" comment in nfsft_init_guru(), kernel/nfsft/nfsft.c) - but empirically
# (see the FSFT PR this fix was made in) two separately-constructed FSFT
# plans with the *same* N disagree on trafo_direct()/adjoint_direct() output
# most of the time, while repeated calls *on the same plan* always agree
# exactly (checked 15/15). That means plan->x is not being populated
# consistently across plan instances in the bundled library build - possibly
# an older build than its "3.6.0" label suggests - so trafo_direct()/
# adjoint_direct() can only be smoke-tested (finite, deterministic across
# repeated calls on one plan), never compared across two different plans, for
# *any* N - not just at/above NFSFT_BREAK_EVEN (bandwidth 5) where trafo()
# additionally switches to a distinct FFT-based algorithm that never touches
# x at all.

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

# --- below NFSFT_BREAK_EVEN: trafo() literally calls trafo_direct() on the
# --- SAME plan internally (see kernel/nfsft/nfsft.c), so this is still only a
# --- same-plan smoke test (finite, deterministic across repeated calls) - NOT
# --- a comparison across two separately-constructed plans, which the
# --- plan->x inconsistency documented above makes meaningless for any N.
# --- Must run after the N=6 block above (see ordering note).
N_small = 3

plan_small = FSFT(
    N_small, 1
)  # M is ignored/overridden by FSFT for the actual grid size
fhat_small = np.zeros((2 * plan_small.N + 2) ** 2, dtype=np.complex128)
for k in range(N_small + 1):
    for n in range(-k, k + 1):
        fhat_small[plan_small.fsft_index(k, n)] = (np.random.rand() - 0.5) + 1j * (
            np.random.rand() - 0.5
        )
plan_small.fhat = fhat_small

plan_small.trafo()  # N < NFSFT_BREAK_EVEN -> internally calls trafo_direct()
f_small_first = np.copy(plan_small.f)
assert np.all(
    np.isfinite(f_small_first)
), "TRAFO TEST FAILED: fsft_trafo() produced NaN/Inf for N < NFSFT_BREAK_EVEN"

plan_small.fhat = fhat_small  # trafo() overwrote fhat in place, see note above
plan_small.trafo()
f_small_second = np.copy(plan_small.f)
assert np.array_equal(
    f_small_first, f_small_second
), "TRAFO TEST FAILED: fsft_trafo() is not deterministic for identical input (N < NFSFT_BREAK_EVEN)"

print("fsft_trafo() smoke test (N < NFSFT_BREAK_EVEN) passed: finite and deterministic")

# TODO: fsft_adjoint()/fsft_adjoint_direct() still need a trustworthy,
# independent reference before they can be tested properly - same plan->x
# caveat as above, plus adjoint_direct() shares the trafo_direct() limitation.
