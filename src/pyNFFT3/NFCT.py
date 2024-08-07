import ctypes
import numpy as np
from .flags import *
from . import _nfctlib
from . import nfct_plan

# Set arugment and return types for functions
_nfctlib.jnfct_init.argtypes = [
    ctypes.POINTER(nfct_plan),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_uint32,
]

_nfctlib.jnfct_alloc.restype = ctypes.POINTER(nfct_plan)
_nfctlib.jnfct_finalize.argtypes = [ctypes.POINTER(nfct_plan)]

_nfctlib.jnfct_set_x.argtypes = [
    ctypes.POINTER(nfct_plan),
    np.ctypeslib.ndpointer(np.float64, flags="C"),
]
_nfctlib.jnfct_set_f.argtypes = [
    ctypes.POINTER(nfct_plan),
    np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C"),
]
_nfctlib.jnfct_set_fhat.argtypes = [
    ctypes.POINTER(nfct_plan),
    np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C"),
]

_nfctlib.jnfct_trafo.argtypes = [ctypes.POINTER(nfct_plan)]
_nfctlib.jnfct_adjoint.argtypes = [ctypes.POINTER(nfct_plan)]
_nfctlib.jnfct_trafo_direct.argtypes = [ctypes.POINTER(nfct_plan)]
_nfctlib.jnfct_adjoint_direct.argtypes = [ctypes.POINTER(nfct_plan)]


class NFCT:
    def __init__(self, N, M, n=None, m=default_window_cut_off, f1=None, f2=f2_default):
        """
        Class to perform non-equispaced fast cosine transforms (NFCT)
        with a dimension of **D**.
        Just **N** and **M** are required for initializing a plan.
        """
        self.plan = _nfctlib.jnfct_alloc()
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
            self.n = (2 ** (np.ceil(np.log(self.N) / np.log(2)) + 1)).astype("int32")
            n_ct = self.n.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        if any(x <= 0 for x in self.n):
            raise ValueError(
                f"Invalid n: {self.n}. Argument must be a positive integer"
            )

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
        _nfctlib.jnfct_init(
            self.plan, self.D, N_ct, self.M, n_ct, self.m, self.f1, self.f2
        )
        self.init_done = True  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function values
        self.fhat = None  # Fourier coefficients

    def __del__(self):
        self.finalize_plan()

    def nfct_finalize_plan(self):
        """
        Finalizes an NFCT plan.
        This function does not have to be called by the user.
        """
        _nfctlib.jnfct_finalize.argtypes = (ctypes.POINTER(nfct_plan),)  # P

        if not self.init_done:
            raise ValueError("NFST not initialized.")

        if not self.finalized:
            self.finalized = True
            _nfctlib.jnfct_finalize(self.plan)

    def finalize_plan(self):
        """
        Alternative call for **nfct_finalize_plan()**
        """
        return self.nfct_finalize_plan()

    def nfct_init(self):
        """
        Initializes the NFCT plan in C.
        This function does not have to be called by the user.
        """
        # Convert N and n to numpy arrays for passing them to C
        Nv = np.array(self.N, dtype=np.int32)
        n = np.array(self.n, dtype=np.int32)

        # Call init for memory allocation
        ptr = _nfctlib.jnfct_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(nfct_plan))

        # Initialize values
        _nfctlib.jnfct_init(
            self.plan,
            ctypes.c_int(self.D),
            ctypes.cast(Nv.ctypes.data, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.M),
            ctypes.cast(n.ctypes.data, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.m),
            ctypes.c_uint(self.f1),
            ctypes.c_uint(self.f2),
        )
        self.init_done = True

    def init(self):
        """
        Alternative call for **nfft_init()**
        """
        return self.nfct_init()

    @property
    def x(self):
        return self._X

    @x.setter
    def x(self, value):
        if value is not None:
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.flags["C"]
            ):
                raise RuntimeError("x has to be C-continuous, numpy float64 array")
            if self.D == 1:
                _nfctlib.jnfct_set_x.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=self.M, flags="C"
                )
            else:
                _nfctlib.jnfct_set_x.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=(self.M, self.D), flags="C"
                )
            self._X = _nfctlib.jnfct_set_x(self.plan, value)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        if value is not None:
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.flags["C"]
            ):
                raise RuntimeError("f has to be C-continuous, numpy float64 array")
            _nfctlib.jnfct_set_f.restype = np.ctypeslib.ndpointer(
                np.float64, ndim=1, shape=self.M, flags="C"
            )
            self._f = _nfctlib.jnfct_set_f(self.plan, value)

    @property
    def fhat(self):
        return self._fhat

    @fhat.setter
    def fhat(self, value):
        Ns = np.prod(self.N)
        if value is not None:
            if not (isinstance(value, np.ndarray) and value.dtype == np.float64):
                raise RuntimeError("fhat has to be a numpy float64 array")
            if not value.flags["C"]:
                raise RuntimeError("fhat has to be C-continuous")
            if value.size != Ns:
                raise ValueError(f"fhat has to be an array of size {Ns}")
            _nfctlib.jnfct_set_fhat.argtypes = [
                ctypes.POINTER(nfct_plan),
                np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C"),
            ]
            _nfctlib.jnfct_set_fhat.restype = np.ctypeslib.ndpointer(
                np.float64, ndim=1, shape=Ns, flags="C_CONTIGUOUS"
            )
            self._fhat = _nfctlib.jnfct_set_fhat(self.plan, value)

    @property
    def num_threads(self):
        return _nfctlib.nfft_get_num_threads()

    def nfct_trafo(self):
        """
        Computes the NDCT via the fast NFCT algorithm for the provided nodes in **x** and coefficients in **fhat**.
        """
        Ns = np.prod(self.N)
        _nfctlib.jnfct_trafo.restype = np.ctypeslib.ndpointer(
            np.float64, shape=Ns, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if not hasattr(self, "fhat"):
            raise ValueError("fhat has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfctlib.jnfct_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        """
        Alternative call for **nfct_trafo()**
        """
        return self.nfct_trafo()

    def nfct_trafo_direct(self):
        """
        Computes the NDCT via naive matrix-vector multiplication for the provided nodes in **x** and coefficients in **fhat**.
        """
        _nfctlib.jnfct_trafo_direct.restype = np.ctypeslib.ndpointer(
            np.float64, shape=self.M, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if self.fhat is None:
            raise ValueError("fhat has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = _nfctlib.jnfct_trafo_direct(self.plan)
        self.f = ptr

    def trafo_direct(self):
        """
        Alternative call for **nfct_trafo_direct()**
        """
        return self.nfct_trafo_direct()

    def nfct_transposed(self):
        """
        Computes the transposed NDCT via the fast transposed NFCT algorithm for the provided nodes in **x** and coefficients in **f**.
        """
        Ns = np.prod(self.N)
        _nfctlib.jnfct_adjoint.restype = np.ctypeslib.ndpointer(
            np.float64, shape=Ns, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if self.f is None:
            raise ValueError("f has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = _nfctlib.jnfct_adjoint(self.plan)
        self.fhat = ptr

    def nfct_transposed_direct(self):
        """
        Computes the transposed NDCT via naive matrix-vector multiplication for provided nodes for the provided nodes in **x** and coefficients in **f**.
        """
        Ns = np.prod(self.N)
        _nfctlib.jnfct_adjoint_direct.restype = np.ctypeslib.ndpointer(
            np.float64, shape=Ns, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFCT already finalized")

        if self.f is None:
            raise ValueError("f has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = _nfctlib.jnfct_adjoint_direct(self.plan)
        self.fhat = ptr

    def nfct_adjoint(self):
        """
        Alternative call for **nfct_transposed()**
        """
        return self.nfct_transposed()

    def nfct_adjoint_direct(self):
        """
        Alternative call for **nfct_transposed_direct()**
        """
        return self.nfct_transposed_direct()

    def adjoint(self):
        """
        Alternative call for **nfct_adjoint()**
        """
        return self.nfct_adjoint()

    def adjoint_direct(self):
        """
        Alternative call for **nfct_adjoint_direct()**
        """
        return self.nfct_adjoint_direct()
