import ctypes

PRE_PHI_HUT = 1 << 0
"""
PRE_PHI_HUT

The deconvolution step (the multiplication with the diagonal matrix :math:`D`) uses precomputed values of the Fourier transformed window function.
"""

FG_PSI = 1 << 1
"""
FG_PSI

The convolution step (the multiplication with the sparse matrix :math:`B`) uses particular properties of the Gaussian window function to trade multiplications for direct calls to exponential function.
"""

PRE_LIN_PSI = 1 << 2
"""
PRE_LIN_PSI

Linear interpolation of the window function from a lookup table.
"""

PRE_FG_PSI = 1 << 3
"""
PRE_FG_PSI

The convolution step (the multiplication with the sparse matrix :math:`B`) uses particular properties of the Gaussian window function to trade multiplications for direct calls to exponential function.
"""

PRE_PSI = 1 << 4
"""
PRE_PSI

The convolution step (the multiplication with the sparse matrix :math:`B`) uses :math:`(2m+2)dM` precomputed values of the window function.
"""

PRE_FULL_PSI = 1 << 5
"""
PRE_FULL_PSI

The convolution step (the multiplication with the sparse matrix :math:`B`) uses :math:`(2m+2)^d M` precomputed values of the window function, 
in addition indices of source and target vectors are stored.
"""

MALLOC_X = 1 << 6
"""
MALLOC_X

Allocate memory for node :math:`x_j`.
"""

MALLOC_F_HAT = 1 << 7
"""
MALLOC_F_HAT

Allocate memory for coefficient :math:`hat{f}_k`.
"""

MALLOC_F = 1 << 8
"""
MALLOC_F

Allocate memory for approximate function value :math:`f_j`.
"""

FFT_OUT_OF_PLACE = 1 << 9
"""
FFT_OUT_OF_PLACE

FFTW uses disjoint input/output vector.
"""

FFTW_INIT = 1 << 10
"""
FFTW_INIT

Initialize FFTW plan.
"""

PRE_ONE_PSI = PRE_LIN_PSI | PRE_FG_PSI | PRE_PSI | PRE_FULL_PSI
"""
PRE_ONE_PSI

Summarises if precomputation is used within the convolution step (the multiplication with the sparse matrix :math:`B`). 
If testing against this flag is positive, nfft_precompute_one_psi has to be called.
"""

NFFT_SORT_NODES = 1 << 11
"""
NFFT_SORT_NODES

Internal sorting of the nodes :math:`x_j` that may increase performance.
"""

NFFT_OMP_BLOCKWISE_ADJOINT = 1 << 12
"""
NFFT_OMP_BLOCKWISE_ADJOINT

Blockwise calculation for adjoint NFFT in the case of OpenMP.
"""
# NFCT flags
NFCT_SORT_NODES = 1 << 11
"""
NFCT_SORT_NODES

Internal sorting of the nodes :math:`x_j` that may increase performance.
"""

NFCT_OMP_BLOCKWISE_ADJOINT = 1 << 12
"""
NFCT_OMP_BLOCKWISE_ADJOINT

Blockwise calculation for adjoint NFFT in the case of OpenMP.
"""
# NFST flags
NFST_SORT_NODES = 1 << 11
"""
NFST_SORT_NODES

Internal sorting of the nodes :math:`x_j` that may increase performance.
"""

NFST_OMP_BLOCKWISE_ADJOINT = 1 << 12
"""
NFST_OMP_BLOCKWISE_ADJOINT

Blockwise calculation for adjoint NFFT in the case of OpenMP.
"""
# FFTW flags
FFTW_MEASURE = 0
"""
FFTW_MEASURE

Find optimal plan by executing several FFTs and compare times.
"""

FFTW_DESTROY_INPUT = 1 << 0
"""
FFTW_DESTROY_INPUT

An out-of-place transform is allowed to overwrite the input array with arbitrary data.
"""

FFTW_UNALIGNED = 1 << 1
"""
FFTW_UNALIGNED

The algorithm may not impose any unusual alignment requirements on the input/output arrays (not necessary in most context).
"""

FFTW_CONSERVE_MEMORY = 1 << 2
"""
FFTW_CONSERVE_MEMORY

Conserving memory.
"""

FFTW_EXHAUSTIVE = 1 << 3
"""
FFTW_EXHAUSTIVE

Behaves like FFTW_PATIENT with an even wider range of tests.
"""

FFTW_PRESERVE_INPUT = 1 << 4
"""
FFTW_PRESERVE_INPUT

Input vector is preserved and unchanged.
"""

FFTW_PATIENT = 1 << 5
"""
FFTW_PATIENT

Behaves like FFTW_MEASURE with a wider range of tests.
"""

FFTW_ESTIMATE = 1 << 6
"""
FFTW_ESTIMATE

Use simple heuristic instead of measurements to pick a plan.
"""

FFTW_WISDOM_ONLY = 1 << 21
"""
FFTW_WISDOM_ONLY

A plan is only created if wisdom from tests is available.
"""
# NFSFT flags
NFSFT_NORMALIZED = 1 << 0
"""
NFSFT_NORMALIZED

Multiply Fourier coefficients with normalization weight.
"""

NFSFT_USE_NDFT = 1 << 1
"""
NFSFT_USE_NDFT

The fast NFSFT algorithms will use internally the exact but usually slower direct NDFT algorithm in favor of fast but approximative NFFT algorithm.
"""

NFSFT_USE_DPT = 1 << 2
"""
NFSFT_USE_DPT

The fast NFSFT algorithms will use internally the usually slower direct DPT algorithm in favor of the fast FPT algorithm.
"""

NFSFT_MALLOC_X = 1 << 3
"""
NFSFT_MALLOC_X

The init methods will allocate memory and the method nfsft_finalize will free the array x for you.
"""

NFSFT_MALLOC_F_HAT = 1 << 5
"""
NFSFT_MALLOC_F_HAT

The init methods will allocate memory and the method nfsft_finalize will free the array fhat for you.
"""

NFSFT_MALLOC_F = 1 << 6
"""
NFSFT_MALLOC_F

The init methods will allocate memory and the method nfsft_finalize will free the array f for you.
"""

NFSFT_PRESERVE_F_HAT = 1 << 7
"""
NFSFT_PRESERVE_F_HAT

Guarantee that during an execution of nfsft_direct_trafo or nfsft_trafo, the content of fhat remains unchanged.
"""

NFSFT_PRESERVE_X = 1 << 8
"""
NFSFT_PRESERVE_X

Guarantee that during an execution of nfsft_direct_trafo, nfsft_trafo or nfsft_direct_adjoint, nfsft_adjoint the content of x remains unchanged.
"""

NFSFT_PRESERVE_F = 1 << 9
"""
NFSFT_PRESERVE_F

Guarantee that during an execution of nfsft_direct_adjoint or nfsft_adjoint the content of f remains unchanged.
"""

NFSFT_DESTROY_F_HAT = 1 << 10
"""
NFSFT_DESTROY_F_HAT

Explicitly allow that during an execution of nfsft_direct_trafo or nfsft_trafo, the content of fhat may be changed.
"""

NFSFT_DESTROY_X = 1 << 11
"""
NFSFT_DESTROY_X

Explicitly allow that during an execution of nfsft_direct_trafo, nfsft_trafo, nfsft_direct_adjoint, or nfsft_adjoint, the content of x may be changed.
"""

NFSFT_DESTROY_F = 1 << 12
"""
NFSFT_DESTROY_F

Explicitly allow that during an execution of nfsft_direct_adjoint or nfsft_adjoint, the content of f may be changed.
"""

NFSFT_NO_DIRECT_ALGORITHM = 1 << 13
"""
NFSFT_NO_DIRECT_ALGORITHM

The transforms nfsft_direct_trafo and nfsft_direct_adjoint do not work. Setting this flag saves some memory for precomputed data.
"""

NFSFT_NO_FAST_ALGORITHM = 1 << 14
"""
NFSFT_NO_FAST_ALGORITHM

The transforms nfsft_trafo and nfsft_adjoint do not work. Setting this flag saves memory for precomputed data.
"""

NFSFT_ZERO_F_HAT = 1 << 16
"""
NFSFT_ZERO_F_HAT

The transforms nfsft_adjoint and nfsft_direct_adjoint set all unused entries in fhat not corresponding to spherical Fourier coefficients to zero.
"""

NFSFT_EQUISPACED = 1 << 17
"""
NFSFT_EQUISPACED

Use the equispaced FFT instead of the NFFT.
"""

# Default flag values
f1_default_1d = (
    PRE_PHI_HUT
    | PRE_PSI
    | MALLOC_X
    | MALLOC_F_HAT
    | MALLOC_F
    | FFTW_INIT
    | FFT_OUT_OF_PLACE
)
"""
f1_default_1d

Default NFFT flags for 1 dimension cases.
"""

f1_default = (
    PRE_PHI_HUT
    | PRE_PSI
    | MALLOC_X
    | MALLOC_F_HAT
    | MALLOC_F
    | FFTW_INIT
    | FFT_OUT_OF_PLACE
    | NFCT_SORT_NODES
    | NFCT_OMP_BLOCKWISE_ADJOINT
)
"""
f1_default

Default NFFT flags for multidimensional cases. 
"""

f2_default = ctypes.c_uint32(FFTW_ESTIMATE | FFTW_DESTROY_INPUT)
"""
f2_default

Default FFTW flags.
"""

nfsft_default = NFSFT_MALLOC_X | NFSFT_MALLOC_F | NFSFT_MALLOC_F_HAT

"""
nfsft_default

Default NFSFT flags.
"""

nfsft_nfft_default = ctypes.c_uint32(
    PRE_PHI_HUT | PRE_PSI | FFTW_INIT | NFFT_OMP_BLOCKWISE_ADJOINT,
)
"""
nfsft_nfft_default

Default NFFT flags for NFSFT calculations.
"""

default_window_cut_off = 8  # TODO: Add proper nfft_get_default_window_cut_off ccall
"""
default_window_cut_off

The default NFFT window size cutoff.
"""

nfsft_default_nfft_cut_off = 6  # TODO: Add proper nfsft_default_nfft_cut_off ccall
"""
nfsft_default_nfft_cut_off

The default NFFT cutoff parameter.
"""

nfsft_default_threshold = 1000  # TODO: Add proper nfsft_default_nfft_cut_off ccall
"""
nfsft_default_threshold

The default threshold for the FPT.
"""

BASES = {"exp": 0, "cos": 1, "alg": 2}
