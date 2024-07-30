import warnings
import ctypes
import numpy as np
from .flags import *
from . import fastsumlib
from . import fastsum_plan

# """
#     FASTSUM

# The fast summation algorithm evaluates the function

# ```math
# f(\pmb{y}) \coloneqq \sum^{N}_{k = 1} \alpha_k \, \mathscr{K}(\pmb{y} - \pmb{x}_k) = \sum^{N}_{k = 1} \alpha_k \, K(\lVert \pmb{y} - \pmb{x}_k \rVert_2)
# ```

# for given arbitrary source knots ``\pmb{x}_k \in \mathbb{R}^d, k = 1,2, \cdots, N`` and a given kernel function ``\mathscr{K}(\cdot) = K (\lVert \cdot \rVert_2), \; \pmb{x} \in \mathbb{R}^d``, 
# which is an even, real univariate function which is infinitely differentiable at least in ``\mathbb{R} \setminus \{ 0 \}``. 
# If ``K`` is infinitely differentiable at zero as well, then ``\mathscr{K}`` is defined on ``\mathbb{R}^d`` and is called 
# nonsingular kernel function. The evaluation is done at ``M`` different points ``\pmb{y}_j \in \mathbb{R}^d, j = 1, \cdots, M``. 

# # Fields
# * `d` - dimension.
# * `N` - number of source nodes.
# * `M` - number of target nodes.
# * `n` - expansion degree.
# * `p` - degree of smoothness.
# * `kernel` - name of kernel function ``K``.
# * `c` - kernel parameters.
# * `eps_I` - inner boundary.
# * `eps_B` - outer boundary.
# * `nn_x` - oversampled nn in x.
# * `nn_y` - oversampled nn in y.
# * `m_x` - NFFT-cutoff in x.
# * `m_y` - NFFT-cutoff in y.
# * `init_done` - bool for plan init.
# * `finalized` - bool for finalizer.
# * `flags` - flags.
# * `x` - source nodes.
# * `y` - target nodes.
# * `alpha` - source coefficients.
# * `f` - target evaluations.
# * `plan` - plan (C pointer).

# # Constructor
#     FASTSUM( d::Integer, N::Integer, M::Integer, n::Integer, p::Integer, kernel::String, c::Vector{<:Real}, eps_I::Real, eps_B::Real, nn_x::Integer, nn_y::Integer, m_x::Integer, m_y::Integer, flags::UInt32 )

# # Additional Constructor
#     FASTSUM( d::Integer, N::Integer, M::Integer, n::Integer, p::Integer, kernel::String, c::Real, eps_I::Real, eps_B::Real, nn::Integer, m::Integer )

# # See also
# [`NFFT`](@ref)
# """

# Set arugment and return types for functions
fastsumlib.jfastsum_init.argtypes = [ctypes.POINTER(fastsum_plan), 
                               ctypes.c_int,
                               ctypes.c_char_p,
                               np.ctypeslib.ndpointer(np.float64, flags='C'),
                               ctypes.c_uint,
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_double,
                               ctypes.c_float,
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_int, 
                               ctypes.c_int,
                               ctypes.c_int]

fastsumlib.jfastsum_alloc.restype = ctypes.POINTER(fastsum_plan)
fastsumlib.jfastsum_finalize.argtypes = [ctypes.POINTER(fastsum_plan)]

fastsumlib.jfastsum_set_x.argtypes = [ctypes.POINTER(fastsum_plan), np.ctypeslib.ndpointer(np.float64, flags='F')]
fastsumlib.jfastsum_set_y.argtypes = [ctypes.POINTER(fastsum_plan), np.ctypeslib.ndpointer(np.float64, flags='F')] 
fastsumlib.jfastsum_set_alpha.argtypes = [ctypes.POINTER(fastsum_plan), np.ctypeslib.ndpointer(np.complex128, flags='F')]

fastsumlib.jfastsum_trafo.argtypes = [ctypes.POINTER(fastsum_plan)]
fastsumlib.jfastsum_exact.argtypes = [ctypes.POINTER(fastsum_plan)]

class FASTSUM:
    def __init__(self, d, N, M, kernel, c, n=256, p=8, eps_I=8/256, eps_B=1/16, nn=512, m=8):
        self.plan = fastsumlib.jfastsum_alloc()
        
        if N <= 0:
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")
        if n <= 0:
            raise ValueError(f"Invalid n: {n}. Argument must be a positive integer")
        if m <= 0:
            raise ValueError(f"Invalid m: {m}. Argument must be a positive integer")

        self.d = d  # dimension
        self.N = N  # number of source nodes
        self.M = M  # number of target nodes
        self.n = n  # expansion degree
        self.p = p  # degree of smoothness
        self.kernel = ctypes.create_string_buffer(kernel.encode("utf-8"))  # name of kernel and encode as string
        self.c = np.array([c], dtype=np.float64) if isinstance(c, (int, float)) else np.array(c, dtype=np.float64)  # kernel parameters
        self.eps_I = eps_I # inner boundary
        self.eps_B = eps_B  # outer boundary
        self.nn_x = nn  # oversampled nn in x
        self.nn_y = nn  # oversampled nn in y
        self.m_x = m  # NFFT-cutoff in x
        self.m_y = m  # NFFT-cutoff in y
        self.flags = 0  # flags

        self.finalized = False  # bool for finalizer
        self.init_done = False  # bool for plan init
        self.x = None  # source nodes
        self.y = None  # target nodes
        self.alpha = None  # source coefficients
        self.f = None  # target evaluations

    def __del__(self):
        self.finalize_plan()
    
    def fastsum_finalize_plan(self):
        fastsumlib.jfastsum_finalize.argtypes = (
            ctypes.POINTER(fastsum_plan),   # P
        )

        if not self.init_done:
            raise ValueError("FASTSUM not initialized.")

        if not self.finalized:
            self.finalized = True
            fastsumlib.jfastsum_finalize(self.plan)

    def finalize_plan(self):
        return self.fastsum_finalize_plan()
    
    # """
    #     fastsum_init(P)

    # intialises a transform plan.

    # # Input
    # * `P` - a FASTSUM plan structure.

    # # See also
    # [`FASTSUM{D}`](@ref), [`fastsum_finalize_plan`](@ref)
    # """

    def fastsum_init(self):
        # Convert c to numpy array for passing them to C
        Cv = np.array(self.c, dtype=np.float64)

        # Call init for memory allocation
        ptr = fastsumlib.jfastsum_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(fastsum_plan))

        # Initialize values
        code = fastsumlib.jfastsum_init(
            self.plan,
            ctypes.c_int(self.d),
            self.kernel,
            Cv,
            ctypes.c_uint(self.flags),
            ctypes.c_int(self.n),
            ctypes.c_int(self.p),
            ctypes.c_double(self.eps_I),
            ctypes.c_float(self.eps_B),
            ctypes.c_int(self.N),
            ctypes.c_int(self.M),
            ctypes.c_int(self.nn_x),
            ctypes.c_int(self.nn_y),
            ctypes.c_int(self.m_x),
            ctypes.c_int(self.m_y)
        )
        self.init_done = True

        if code == 1:
            raise RuntimeError("Unkown kernel")

    def init(self):
        return self.fastsum_init()
    
    @property
    def x(self):
        return np.ascontiguousarray(self._X).T

    @x.setter 
    def x(self, value):
        if value is not None:
            if self.init_done is False:
                self.init()
            X_fort = np.asfortranarray(value)
            if self.d == 1:
                if not (isinstance(value, np.ndarray) and value.dtype == np.float64):
                    raise RuntimeError("x has to be a numpy float64 array")
                if not value.flags['C']:
                    raise RuntimeError("x has to be C-continuous")
                if value.size != self.N:
                    raise ValueError(f"x has to be an array of size {self.N}")
                fastsumlib.jfastsum_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=self.N)
            else:
                if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != 2:
                    raise ValueError("x has to be a Float64 matrix.")
                if value.shape[0] != self.N or value.shape[1] != self.d:
                    raise ValueError(f"x has to be a Float64 matrix of size {self.N}")
                fastsumlib.jfastsum_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(self.N,self.d))
            self._X = fastsumlib.jfastsum_set_x(self.plan, X_fort)

    @property
    def y(self):
        return np.ascontiguousarray(self._Y).T

    @y.setter 
    def y(self, value):
        if value is not None:
            if self.init_done is False:
                self.init()
            Y_fort = np.asfortranarray(value)
            if self.d == 1:
                if not (isinstance(value, np.ndarray) and value.dtype == np.float64):
                    raise RuntimeError("y has to be a numpy float64 array")
                if not value.flags['C']:
                    raise RuntimeError("y has to be C-continuous")
                if value.size != self.M:
                    raise ValueError(f"y has to be an array of size {self.M}")
                fastsumlib.jfastsum_set_y.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=self.M)
            else:
                if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != 2:
                    raise ValueError("y has to be a Float64 matrix.")
                if value.shape[0] != self.M or value.shape[1] != self.d:
                    raise ValueError(f"y has to be a Float64 matrix of size {self.M}")
                fastsumlib.jfastsum_set_y.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(self.M,self.d))
            self._Y = fastsumlib.jfastsum_set_y(self.plan, Y_fort)

    @property
    def alpha(self):
        return np.ascontiguousarray(self._Alpha)

    @alpha.setter 
    def alpha(self, value):
        if value is not None:
            if self.init_done is False:
                self.init()
            if not (isinstance(value, np.ndarray) and value.dtype == np.complex128):
                raise RuntimeError("alpha has to be a numpy complex128 array")
            if value.size != self.N:
                raise ValueError(f"alpha has to be an array of size {self.N}")
            
            # Create a copy of the array to modify
            alpha_array = np.copy(value)
            
            alpha_fort = np.asfortranarray(alpha_array)

            fastsumlib.jfastsum_set_alpha.restype = np.ctypeslib.ndpointer(np.complex128, shape=self.N)
            self._Alpha = fastsumlib.jfastsum_set_alpha(self.plan, alpha_fort)

    # """
    #     fastsum_trafo(P)

    # fast NFFT-based summation.

    # # Input
    # * `P` - a FASTSUM plan structure.

    # # See also
    # [`FASTSUM{D}`](@ref), [`fastsum_trafo_exact`](@ref)
    # """

    def fastsum_trafo(self):
        fastsumlib.jfastsum_trafo.restype = np.ctypeslib.ndpointer(np.complex128, shape=self.M, flags='F')
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("FASTSUM already finalized")

        if not hasattr(self, 'y'):
            raise ValueError("y has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")
        
        if not hasattr(self, 'alpha'):
            raise ValueError("alpha has not been set.")
        
        print("c=",self.c)

        ptr = fastsumlib.jfastsum_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        return self.fastsum_trafo()
    
    # """
    #     fastsum_trafo_exact(P)

    # direct computation of sums.

    # # Input
    # * `P` - a FASTSUM plan structure.

    # # See also
    # [`FASTSUM{D}`](@ref), [`fastsum_trafo`](@ref)
    # """

    def fastsum_trafo_exact(self):
        fastsumlib.jfastsum_exact.restype = np.ctypeslib.ndpointer(np.complex128, shape=self.M, flags='F')
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("FASTSUM already finalized")

        if not hasattr(self, 'y'):
            raise ValueError("y has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")
        
        if not hasattr(self, 'alpha'):
            raise ValueError("alpha has not been set.")
        
        print("c=",self.c)
        
        ptr = fastsumlib.jfastsum_exact(self.plan)
        self.f = ptr

    def trafoexact(self):
        return self.fastsum_trafo_exact()