import ctypes
import numpy as np
from .flags import *
from . import nfctlib
from . import nfct_plan

# # NFCT plan struct
# """
#     NFCT{D}

# A NFCT (nonequispaced fast cosine transform) plan, where D is the dimension. 

# The NFCT realizes a direct and fast computation of the discrete nonequispaced cosine transform. The aim is to compute

# ```math
# f^c(\pmb{x}_j) = \sum_{\pmb{k} \in I_{\pmb{N},\mathrm{c}}^D} \hat{f}_{\pmb{k}}^c \, \cos(2\pi \, \pmb{k} \odot \pmb{x}_j)
# ```

# at given arbitrary knots ``\pmb{x}_j \in [0,0.5]^D, j = 1, \cdots, M``, for coefficients ``\hat{f}^{c}_{\pmb{k}} \in \mathbb{R}``, ``\pmb{k} \in I_{\pmb{N},\mathrm{c}}^D \coloneqq \left\{ \pmb{k} \in \mathbb{Z}^D: 1 \leq k_i \leq N_i - 1, \, i = 1,2,\ldots,D \right\}``, and a multibandlimit vector ``\pmb{N} \in \mathbb{N}^{D}``. Note that we define ``\cos(\pmb{k} \circ \pmb{x}) \coloneqq \prod_{i=1}^D \cos(k_i \cdot x_i)``. The transposed problem reads as

# ```math
# \hat{h}^c_{\pmb{k}} = \sum_{j=1}^M f^c_j \, \cos(2\pi \, \pmb{k} \odot \pmb{x}_j)
# ```

# for the frequencies ``\pmb{k} \in I_{\pmb{N},\mathrm{c}}^D`` with given coefficients ``f^c_j \in \mathbb{R}, j = 1,2,\ldots,M``.

# # Fields
# * `N` - the multibandlimit ``(N_1, N_2, \ldots, N_D)`` of the trigonometric polynomial ``f^s``.
# * `M` - the number of nodes.
# * `n` - the oversampling ``(n_1, n_2, \ldots, n_D)`` per dimension.
# * `m` - the window size. A larger m results in more accuracy but also a higher computational cost. 
# * `f1` - the NFCT flags.
# * `f2` - the FFTW flags.
# * `init_done` - indicates if the plan is initialized.
# * `finalized` - indicates if the plan is finalized.
# * `x` - the nodes ``x_j \in [0,0.5]^D, \, j = 1, \ldots, M``.
# * `f` - the values ``f^c(\pmb{x}_j)`` for the NFCT or the coefficients ``f_j^c \in \mathbb{R}, j = 1, \ldots, M,`` for the transposed NFCT.
# * `fhat` - the Fourier coefficients ``\hat{f}_{\pmb{k}}^c \in \mathbb{R}`` for the NFCT or the values ``\hat{h}_{\pmb{k}}^c, \pmb{k} \in I_{\pmb{N},\mathrm{c}}^D,`` for the adjoint NFFT.
# * `plan` - plan (C pointer).

# # Constructor
#     NFCT{D}( N::NTuple{D,Int32}, M::Int32, n::NTuple{D,Int32}, m::Int32, f1::UInt32, f2::UInt32 ) where {D}

# # Additional Constructor
#     NFCT( N::NTuple{D,Int32}, M::Int32, n::NTuple{D,Int32}, m::Int32, f1::UInt32, f2::UInt32) where {D}
#     NFCT( N::NTuple{D,Int32}, M::Int32) where {D}

# # See also
# [`NFFT`](@ref)
# """

# Set arugment and return types for functions
nfctlib.jnfct_init.argtypes = [ctypes.POINTER(nfct_plan), 
                               ctypes.c_int32, 
                               ctypes.POINTER(ctypes.c_int32), 
                               ctypes.c_int32, 
                               ctypes.POINTER(ctypes.c_int32), 
                               ctypes.c_int32, 
                               ctypes.c_uint32, 
                               ctypes.c_uint32]

nfctlib.jnfct_alloc.restype = ctypes.POINTER(nfct_plan)
nfctlib.jnfct_finalize.argtypes = [ctypes.POINTER(nfct_plan)]

nfctlib.jnfct_set_x.argtypes = [ctypes.POINTER(nfct_plan), np.ctypeslib.ndpointer(np.float64, flags='C')]
nfctlib.jnfct_set_f.argtypes = [ctypes.POINTER(nfct_plan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')] 
nfctlib.jnfct_set_fhat.argtypes = [ctypes.POINTER(nfct_plan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')] 

nfctlib.jnfct_trafo.argtypes = [ctypes.POINTER(nfct_plan)]
nfctlib.jnfct_adjoint.argtypes = [ctypes.POINTER(nfct_plan)]
nfctlib.jnfct_trafo_direct.argtypes = [ctypes.POINTER(nfct_plan)]
nfctlib.jnfct_adjoint_direct.argtypes = [ctypes.POINTER(nfct_plan)]

class NFCT:
    def __init__(self, N, M, n=None, m=default_window_cut_off, f1=None, f2=f2_default):
        self.plan = nfctlib.jnfct_alloc()
        self.N = N  # bandwidth tuple
        N_ct = N.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self.M = M  # number of nodes
        self.n = n  # oversampling per dimension
        self.m = m  # window size
        self.D = len(N)  # dimensions

        if any(x <= 0 for x in N):
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        
        if sum(x % 2 for x in N) != 0:
            raise ValueError(f"Invalid N: {N}. Argument must be an even integer")
        
        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")

        if n is None:
            self.n = (2 ** (np.ceil( np.log(self.N) / np.log(2)) +1 )).astype('int32')
            n_ct = self.n.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        if any(x <= 0 for x in self.n):
            raise ValueError(f"Invalid n: {self.n}. Argument must be a positive integer")
        
        if any(x <= y for x, y in zip(self.n, N)):
            raise ValueError(f"Invalid n: {self.n}. Argument must fulfil n_i > N_i")
        
        if sum(x % 2 for x in self.n) != 0:
            raise ValueError(f"Invalid n: {self.n}. Argument must be an even integer")
        
        if m <= 0:
            raise ValueError(f"Invalid m: {m}. Argument must be a positive integer")

        if f1 is None:
            self.f1 = f1_default if self.D > 1 else f1_default_1d
        else:
            self.f1 = f1

        self.f2 = f2  # FFTW flags
        nfctlib.jnfct_init(self.plan, self.D, N_ct, self.M, n_ct, self.m, self.f1, self.f2)
        self.init_done = True  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function values
        self.fhat = None  # Fourier coefficients

    def __del__(self):
        self.finalize_plan()

    # # finalizer
    # """
    #     nfct_finalize_plan(P::NFCT{D})

    # destroys a NFCT plan structure.

    # # Input
    # * `P` - a NFCT plan structure.

    # # See also
    # [`NFCT{D}`](@ref), [`nfct_init`](@ref)
    # """

    def nfct_finalize_plan(self):
        nfctlib.jnfct_finalize.argtypes = (
            ctypes.POINTER(nfct_plan),   # P
        )

        if not self.init_done:
            raise ValueError("NFST not initialized.")

        if not self.finalized:
            self.finalized = True
            nfctlib.jnfct_finalize(self.plan)

    def finalize_plan(self):
        return self.nfct_finalize_plan()
    
    # # allocate plan memory and init with D,N,M,n,m,f1,f2
    # """
    #     nfct_init(P)

    # intialises the NFCT plan in C. This function does not have to be called by the user.

    # # Input
    # * `P` - a NFCT plan structure.

    # # See also
    # [`NFCT{D}`](@ref), [`nfct_finalize_plan`](@ref)
    # """

    def nfct_init(self):
        # Convert N and n to numpy arrays for passing them to C
        Nv = np.array(self.N, dtype=np.int32)
        n = np.array(self.n, dtype=np.int32)

        # Call init for memory allocation
        ptr = nfctlib.jnfct_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(nfct_plan))

        # Initialize values
        nfctlib.jnfct_init(
            self.plan,
            ctypes.c_int(self.D),
            ctypes.cast(Nv.ctypes.data, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.M),
            ctypes.cast(n.ctypes.data, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.m),
            ctypes.c_uint(self.f1),
            ctypes.c_uint(self.f2)
        )
        self.init_done = True

    def init(self):
        return self.nfct_init()
    
    @property
    def x(self):
        return self._X

    @x.setter 
    def x(self, value):
        if value is not None:
            if not (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.flags['C']):
                raise RuntimeError("x has to be C-continuous, numpy float64 array")
            if self.D == 1:
                nfctlib.jnfct_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=self.M, flags='C')
            else:
                nfctlib.jnfct_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(self.M,self.D), flags='C')
            self._X = nfctlib.jnfct_set_x(self.plan, value)
    
    @property
    def f(self):
        return self._f
    
    @f.setter 
    def f(self, value):
        if value is not None:
            if not (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.flags['C']):
                raise RuntimeError("f has to be C-continuous, numpy float64 array")
            nfctlib.jnfct_set_f.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=self.M, flags='C') 
            self._f = nfctlib.jnfct_set_f(self.plan, value)

    @property
    def fhat(self):
        return self._fhat
    
    @fhat.setter 
    def fhat(self, value):
        Ns = np.prod(self.N)
        if value is not None:
            if not (isinstance(value, np.ndarray) and value.dtype == np.float64):
                raise RuntimeError("fhat has to be a numpy float64 array")
            if not value.flags['C']:
                raise RuntimeError("fhat has to be C-continuous")
            if value.size != Ns:
                raise ValueError(f"fhat has to be an array of size {Ns}")
            nfctlib.jnfct_set_fhat.argtypes = [ctypes.POINTER(nfct_plan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')]
            nfctlib.jnfct_set_fhat.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=Ns, flags='C_CONTIGUOUS') 
            self._fhat = nfctlib.jnfct_set_fhat(self.plan, value)

    @property
    def num_threads(self):
        return nfctlib.nfft_get_num_threads()
    
    # # nfct trafo direct [call with NFCT.trafo_direct outside module]
    # """
    #     nfct_trafo_direct(P)

    # computes the NDCT via naive matrix-vector multiplication for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``\hat{f}_{\pmb{k}}^c \in \mathbb{R}, \pmb{k} \in I_{\pmb{N},\mathrm{c}}^D,`` in `P.fhat`.

    # # Input
    # * `P` - a NFCT plan structure.

    # # See also
    # [`NFCT{D}`](@ref), [`nfct_trafo`](@ref)
    # """

    def nfct_trafo_direct(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if self.fhat is None:
            raise ValueError("fhat has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = nfctlib.jnfct_trafo_direct(self.plan)
        self.f = ptr

    def trafo_direct(self):
        return self.nfct_trafo_direct()
    
    # # adjoint trafo direct [call with NFCT.adjoint_direct outside module]
    # """
    #     nfct_transposed_direct(P)

    # computes the transposed NDCT via naive matrix-vector multiplication for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``f_j^c \in \mathbb{R}, j =1,2,\dots,M,`` in `P.f`.

    # # Input
    # * `P` - a NFCT plan structure.

    # # See also
    # [`NFCT{D}`](@ref), [`nfct_transposed`](@ref)
    # """

    def nfct_transposed_direct(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if self.f is None:
            raise ValueError("f has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")
        
        ptr = nfctlib.jnfct_adjoint_direct(self.plan)
        self.fhat = ptr

    def nfct_adjoint_direct(self):
        return self.nfct_transposed_direct()

    def adjoint_direct(self):
        return self.nfct_adjoint_direct()
    
    # # nfct trafo [call with NFCT.trafo outside module]
    # """
    #     nfct_trafo(P)

    # computes the NDCT via the fast NFCT algorithm for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``\hat{f}_{\pmb{k}}^c \in \mathbb{R}, \pmb{k} \in I_{\pmb{N},\mathrm{c}}^D,`` in `P.fhat`.

    # # Input
    # * `P` - a NFCT plan structure.

    # # See also
    # [`NFCT{D}`](@ref), [`nfct_trafo_direct`](@ref)
    # """

    def nfct_trafo(self):
        Ns = np.prod(self.N)
        nfctlib.jnfct_trafo.restype = np.ctypeslib.ndpointer(np.float64, shape=Ns, flags='C')
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if not hasattr(self, 'fhat'):
            raise ValueError("fhat has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")

        ptr = nfctlib.jnfct_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        return self.nfct_trafo()
    
    # # adjoint trafo [call with NFCT.adjoint outside module]
    # """
    #     nfct_transposed(P)

    # computes the transposed NDCT via the fast transposed NFCT algorithm for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``f_j^c \in \mathbb{R}, j =1,2,\dots,M,`` in `P.f`.

    # # Input
    # * `P` - a NFCT plan structure.

    # # See also
    # [`NFCT{D}`](@ref), [`nfct_transposed_direct`](@ref)
    # """

    def nfct_transposed(self):
        Ns = np.prod(self.N)
        nfctlib.jnfct_adjoint.restype = np.ctypeslib.ndpointer(np.float64, shape=Ns, flags='C')
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if self.f is None:
            raise ValueError("f has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")
               
        ptr = nfctlib.jnfct_adjoint(self.plan)
        self.fhat = ptr

    def nfct_adjoint(self):
        return self.nfct_transposed()

    def adjoint(self):
        return self.nfct_adjoint()