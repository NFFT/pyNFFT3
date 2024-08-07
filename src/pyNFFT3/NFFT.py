import ctypes
import numpy as np
from .flags import *
from . import _nfftlib
from . import nfft_plan

# Set arugment and return types for functions
_nfftlib.jnfft_init.argtypes = [
    ctypes.POINTER(nfft_plan),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_uint32,
]

_nfftlib.jnfft_alloc.restype = ctypes.POINTER(nfft_plan)
_nfftlib.jnfft_finalize.argtypes = [ctypes.POINTER(nfft_plan)]

_nfftlib.jnfft_set_x.argtypes = [
    ctypes.POINTER(nfft_plan),
    np.ctypeslib.ndpointer(np.float64, flags="C"),
]
_nfftlib.jnfft_set_f.argtypes = [
    ctypes.POINTER(nfft_plan),
    np.ctypeslib.ndpointer(np.complex128, ndim=1, flags="C"),
]
_nfftlib.jnfft_set_fhat.argtypes = [
    ctypes.POINTER(nfft_plan),
    np.ctypeslib.ndpointer(np.complex128, ndim=1, flags="C"),
]

_nfftlib.jnfft_trafo.argtypes = [ctypes.POINTER(nfft_plan)]
_nfftlib.jnfft_adjoint.argtypes = [ctypes.POINTER(nfft_plan)]
_nfftlib.jnfft_trafo_direct.argtypes = [ctypes.POINTER(nfft_plan)]
_nfftlib.jnfft_adjoint_direct.argtypes = [ctypes.POINTER(nfft_plan)]


class NFFT:
    """
    Class to perform non-equispaced fast Fourier transforms (NFFT)
    considering a **D**-dimensional trigonometric polynomial.
    Just **N** and **M** are required for initializing a plan.
    """

    def __init__(self, N, M, n=None, m=default_window_cut_off, f1=None, f2=f2_default):
        self.plan = _nfftlib.jnfft_alloc()
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
        _nfftlib.jnfft_init(
            self.plan, self.D, N_ct, self.M, n_ct, self.m, self.f1, self.f2
        )
        self.init_done = True  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function values
        self.fhat = None  # Fourier coefficients

    def __del__(self):
        self.finalize_plan()

    def nfft_finalize_plan(self):
        """
        Finalizes an NFFT plan.
        This function does not have to be called by the user.
        """
        _nfftlib.jnfft_finalize.argtypes = (ctypes.POINTER(nfft_plan),)  # P

        if not self.init_done:
            raise ValueError("NFFT not initialized.")

        if not self.finalized:
            self.finalized = True
            _nfftlib.jnfft_finalize(self.plan)

    def finalize_plan(self):
        """
        Alternate call for **nfft_finalize_plan()**
        """
        return self.nfft_finalize_plan()

    def nfft_init(self):
        """
        Initializes the NFFT plan in C.
        This function does not have to be called by the user.
        """
        # Convert N and n to numpy arrays for passing them to C
        Nv = np.array(self.N, dtype=np.int32)
        n = np.array(self.n, dtype=np.int32)

        # Call init for memory allocation
        ptr = _nfftlib.jnfft_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(nfft_plan))

        # Initialize values
        _nfftlib.jnfft_init(
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
        Alternate call for **nfft_init()**
        """
        return self.nfft_init()

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
                _nfftlib.jnfft_set_x.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=self.M, flags="C"
                )
            else:
                _nfftlib.jnfft_set_x.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=(self.M, self.D), flags="C"
                )
            self._X = _nfftlib.jnfft_set_x(self.plan, value)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        if value is not None:
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.complex128
                and value.flags["C"]
            ):
                raise RuntimeError("f has to be C-continuous, numpy complex128 array")
            _nfftlib.jnfft_set_f.restype = np.ctypeslib.ndpointer(
                np.complex128, ndim=1, shape=self.M, flags="C"
            )
            self._f = _nfftlib.jnfft_set_f(self.plan, value)

    @property
    def fhat(self):
        return self._fhat

    @fhat.setter
    def fhat(self, value):
        if value is not None:
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.complex128
                and value.flags["C"]
            ):
                raise RuntimeError(
                    "fhat has to be C-continuous, numpy complex128 array"
                )
            Ns = np.prod(self.N)
            _nfftlib.jnfft_set_fhat.restype = np.ctypeslib.ndpointer(
                np.complex128, ndim=1, shape=Ns, flags="C"
            )
            self._fhat = _nfftlib.jnfft_set_fhat(self.plan, value)

    @property
    def num_threads(self):
        return _nfftlib.nfft_get_num_threads()

    def nfft_trafo(self):
        """
        Computes the NDFT using the fast NFFT algorithm for the provided nodes in **x** and coefficients in **fhat**.
        """
        _nfftlib.jnfft_trafo.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=self.M, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, "fhat"):
            raise ValueError("fhat has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfftlib.jnfft_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        """
        Alternative call for **nfft_trafo()**
        """
        return self.nfft_trafo()

    def nfft_trafo_direct(self):
        """
        Computes the NDFT via naive matrix-vector multiplication for the provided nodes in **x** and coefficients in **fhat**.
        """
        _nfftlib.jnfft_trafo_direct.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=self.M, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if self.fhat is None:
            raise ValueError("fhat has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = _nfftlib.jnfft_trafo_direct(self.plan)
        self.f = ptr

    def trafo_direct(self):
        """
        Alternative call for **nfft_trafo_direct()**
        """
        return self.nfft_trafo_direct()

    def nfft_adjoint(self):
        """
        Computes the adjoint NDFT using the fast adjoint NFFT algorithm for the provided nodes in **x** and coefficients in **f**.
        """
        Ns = np.prod(self.N)
        _nfftlib.jnfft_adjoint.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=Ns, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, "f"):
            raise ValueError("f has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfftlib.jnfft_adjoint(self.plan)
        self.fhat = ptr

    def adjoint(self):
        """
        Alternative call for **nfft_adjoint()**
        """
        return self.nfft_adjoint()

    def nfft_adjoint_direct(self):
        """
        Computes the adjoint NDFT using naive matrix-vector multiplication for the provided nodes in **x** and coefficients in **f**.
        """
        Ns = np.prod(self.N)
        _nfftlib.jnfft_adjoint_direct.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=Ns, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, "f"):
            raise ValueError("f has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfftlib.jnfft_adjoint_direct(self.plan)
        self.fhat = ptr

    def adjoint_direct(self):
        """
        Alternative call for **nfft_adjoint_direct()**
        """
        return self.nfft_adjoint_direct()
