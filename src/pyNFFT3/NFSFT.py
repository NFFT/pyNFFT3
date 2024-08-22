import ctypes
import numpy as np
from .flags import *
from . import _nfsftlib
from . import nfsft_plan

# Set arugment and return types for functions
_nfsftlib.jnfsft_init.argtypes = [
    ctypes.POINTER(nfsft_plan),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_int32,
]

_nfsftlib.jnfsft_alloc.restype = ctypes.POINTER(nfsft_plan)
_nfsftlib.jnfsft_finalize.argtypes = [ctypes.POINTER(nfsft_plan)]

_nfsftlib.jnfsft_set_x.argtypes = [
    ctypes.POINTER(nfsft_plan),
    np.ctypeslib.ndpointer(np.float64, flags="C"),
]
_nfsftlib.jnfsft_set_f.argtypes = [
    ctypes.POINTER(nfsft_plan),
    np.ctypeslib.ndpointer(np.complex128, flags="C"),
]
_nfsftlib.jnfsft_set_fhat.argtypes = [
    ctypes.POINTER(nfsft_plan),
    np.ctypeslib.ndpointer(np.complex128, flags="C"),
]

_nfsftlib.jnfsft_trafo.argtypes = [ctypes.POINTER(nfsft_plan)]
_nfsftlib.jnfsft_adjoint.argtypes = [ctypes.POINTER(nfsft_plan)]
_nfsftlib.jnfsft_trafo_direct.argtypes = [ctypes.POINTER(nfsft_plan)]
_nfsftlib.jnfsft_adjoint_direct.argtypes = [ctypes.POINTER(nfsft_plan)]


class NFSFT:
    """
    Class to perform nonequispaced fast spherical Fourier transforms (NFSFT).
    Just **N** and **M** are required for initializing a plan.
    """

    def __init__(
        self,
        N: int,
        M: int,
        flags: ctypes.c_uint32 = nfsft_default,
        nfft_flags: ctypes.c_uint32 = nfsft_nfft_default,
        nfft_cutoff: int = nfsft_default_nfft_cut_off,
    ):
        self.plan = None
        self.N = N  # bandwidth
        self.M = M  # number of nodes
        self.flags = flags
        self.nfft_flags = nfft_flags
        self.nfft_cutoff = nfft_cutoff

        if N <= 0:
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")

        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")

        self.init_done = False  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function coefficients
        self.fhat = None  # spherical Fourier coefficients

    def __del__(self):
        self.finalize_plan()

    def nfsft_finalize_plan(self):
        """
        Finalizes an NFSFT plan.
        This function does not have to be called by the user.
        """
        _nfsftlib.jnfsft_finalize.argtypes = (ctypes.POINTER(nfsft_plan),)  # P

        if not self.init_done:
            raise ValueError("NFSFT not initialized.")

        if not self.finalized:
            self.finalized = True
            _nfsftlib.jnfsft_finalize(self.plan)

    def finalize_plan(self):
        """
        Alternate call for **nfsft_finalize_plan()**
        """
        return self.nfsft_finalize_plan()

    def nfsft_init(self):
        """
        Initializes the NFSFT plan in C.
        This function does not have to be called by the user.
        """
        # Call init for memory allocation
        ptr = _nfsftlib.jnfsft_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(nfsft_plan))

        # Initialize values
        _nfsftlib.jnfsft_init(
            self.plan,
            ctypes.c_int(self.N),
            ctypes.c_int(self.M),
            self.flags,
            self.nfft_flags,
            ctypes.c_int(self.nfft_cutoff),
        )
        self.init_done = True

    def init(self):
        """
        Alternate call for **nfsft_init()**
        """
        return self.nfsft_init()

    @property
    def x(self) -> np.ndarray:
        return self._X

    @x.setter
    def x(self, value: np.ndarray):
        if value is not None:
            if not self.init_done:
                self.nfsft_init()
            if self.finalized:
                raise RuntimeError("Plan already finalized")
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.flags["C"]
                and value.ndim >= 2
            ):
                raise RuntimeError("x must be a 2D C-contiguous numpy float64 array")
            elif not (value.shape[0] == 2 and value.shape[1] == self.M):
                raise RuntimeError(f"x must be of size 2x{self.M}")
            _nfsftlib.jnfsft_set_x.restype = np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=2, shape=(2, self.M), flags="C"
            )
            ptr = _nfsftlib.jnfsft_set_x(self.plan, np.ascontiguousarray(value.T))
            self._X = ptr

    @property
    def f(self) -> np.ndarray:
        return self._f

    @f.setter
    def f(self, value: np.ndarray):
        if value is not None:
            if not self.init_done:
                self.nfsft_init()
            if self.finalized:
                raise RuntimeError("Plan already finalized")
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.complex128
                and value.flags["C"]
                and value.size == self.M
            ):
                raise RuntimeError(
                    f"f has to be C-continuous, numpy complex128 array of size M ({self.M})"
                )
            _nfsftlib.jnfsft_set_f.restype = np.ctypeslib.ndpointer(
                np.complex128, ndim=1, shape=self.M, flags="C"
            )
            self._f = _nfsftlib.jnfsft_set_f(self.plan, value)

    @property
    def fhat(self) -> np.ndarray:
        return self._fhat

    @fhat.setter
    def fhat(self, value: np.ndarray):
        if value is not None:
            N_total = (2 * self.N + 2) ** 2
            if not self.init_done:
                self.nfsft_init()
            if self.finalized:
                raise RuntimeError("Plan already finalized")
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.complex128
                and value.flags["C"]
            ):
                raise RuntimeError(
                    "fhat has to be C-continuous, numpy complex128 array"
                )
            if value.shape != (N_total,):
                raise RuntimeError(f"fhat must be of size N_total ({N_total})")
            _nfsftlib.jnfsft_set_fhat.restype = np.ctypeslib.ndpointer(
                np.complex128, shape=N_total, flags="C"
            )
            self._fhat = _nfsftlib.jnfsft_set_fhat(self.plan, value)

    @property
    def num_threads(self) -> int:
        return _nfsftlib.nfft_get_num_threads()

    def nfsft_index(self, k: int, n: int) -> int:
        """
        Calculates the index for f_hat calculations.
        This function does not have to be called by the user.
        """
        return (2 * self.N + 2) * (self.N - n + 1) + (self.N + k + 1)

    def nfsft_trafo(self):
        """
        Computes the NFSFT using the fast approximate transform for the provided nodes in **x** and coefficients in **fhat**.
        """
        _nfsftlib.jnfsft_trafo.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=self.M, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFSFT already finalized")

        if not hasattr(self, "fhat"):
            raise ValueError("fhat has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfsftlib.jnfsft_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        """
        Alternative call for **nfsft_trafo()**
        """
        return self.nfsft_trafo()

    def nfsft_trafo_direct(self):
        """
        Computes the NDSFT using direct transformation for the provided nodes in **x** and coefficients in **fhat**.
        """
        _nfsftlib.jnfsft_trafo_direct.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=self.M, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFSFT already finalized")

        if self.fhat is None:
            raise ValueError("fhat has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = _nfsftlib.jnfsft_trafo_direct(self.plan)
        self.f = ptr

    def trafo_direct(self):
        """
        Alternative call for **nfsft_trafo_direct()**
        """
        return self.nfsft_trafo_direct()

    def nfsft_adjoint(self):
        """
        Computes the adjoint NFSFT using the fast approximate transform for the provided nodes in **x** and coefficients in **f**.
        """
        N_total = (2 * self.N + 2) ** 2
        _nfsftlib.jnfsft_adjoint.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=N_total, flags="C"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, "f"):
            raise ValueError("f has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfsftlib.jnfsft_adjoint(self.plan)
        self.fhat = ptr

    def adjoint(self):
        """
        Alternative call for **nfsft_adjoint()**
        """
        return self.nfsft_adjoint()

    def nfsft_adjoint_direct(self):
        """
        Computes the adjoint NDFST using direct transformation for the provided nodes in **x** and coefficients in **f**.
        """
        N_total = (2 * self.N + 2) ** 2
        _nfsftlib.jnfsft_adjoint_direct.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=N_total, flags="C"
        )

        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, "f"):
            raise ValueError("f has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        ptr = _nfsftlib.jnfsft_adjoint_direct(self.plan)
        self.fhat = ptr

    def adjoint_direct(self):
        """
        Alternative call for **nfsft_adjoint_direct()**
        """
        return self.nfsft_adjoint_direct()
