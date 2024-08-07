import ctypes
import numpy as np
from .flags import *
from . import _fastsumlib
from . import fastsum_plan

# Set arugment and return types for functions
_fastsumlib.jfastsum_init.argtypes = [
    ctypes.POINTER(fastsum_plan),
    ctypes.c_int,
    ctypes.c_char_p,
    np.ctypeslib.ndpointer(np.float64, flags="C"),
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
    ctypes.c_int,
]

_fastsumlib.jfastsum_alloc.restype = ctypes.POINTER(fastsum_plan)
_fastsumlib.jfastsum_finalize.argtypes = [ctypes.POINTER(fastsum_plan)]

_fastsumlib.jfastsum_set_x.argtypes = [
    ctypes.POINTER(fastsum_plan),
    np.ctypeslib.ndpointer(np.float64, flags="F"),
]
_fastsumlib.jfastsum_set_y.argtypes = [
    ctypes.POINTER(fastsum_plan),
    np.ctypeslib.ndpointer(np.float64, flags="F"),
]
_fastsumlib.jfastsum_set_alpha.argtypes = [
    ctypes.POINTER(fastsum_plan),
    np.ctypeslib.ndpointer(np.complex128, flags="F"),
]

_fastsumlib.jfastsum_trafo.argtypes = [ctypes.POINTER(fastsum_plan)]
_fastsumlib.jfastsum_exact.argtypes = [ctypes.POINTER(fastsum_plan)]


class FASTSUM:
    """
    Class to perform the fast summation algorithm.
    Check the `NFFT3.jl documentation <https://nfft.github.io/NFFT3.jl/stable/fastsum.html>`_ for the mathematical equations.
    Just **d**, **N**, **M**, **kernel**, and **c** are required for initializing a plan.
    """

    def __init__(
        self, d, N, M, kernel, c, n=256, p=8, eps_I=8 / 256, eps_B=1 / 16, nn=512, m=8
    ):
        self.plan = _fastsumlib.jfastsum_alloc()

        if N <= 0:
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")
        if n <= 0:
            raise ValueError(f"Invalid n: {n}. Argument must be a positive integer")
        if m <= 0:
            raise ValueError(f"Invalid m: {m}. Argument must be a positive integer")
        if 0 > eps_B > 0.5:
            raise ValueError(
                f"Invalid eps_B: {eps_B}. Argument must be between 0.0 and 0.5"
            )

        self.d = d  # dimension
        self.N = N  # number of source nodes
        self.M = M  # number of target nodes
        self.n = n  # expansion degree
        self.p = p  # degree of smoothness
        self.kernel = ctypes.create_string_buffer(
            kernel.encode("utf-8")
        )  # name of kernel and encode as string
        self.c = (
            np.array([c], dtype=np.float64)
            if isinstance(c, (int, float))
            else np.array(c, dtype=np.float64)
        )  # kernel parameters
        self.eps_I = eps_I  # inner boundary
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
        """
        Finalizes a fastsum plan.
        This function does not have to be called by the user.
        """
        _fastsumlib.jfastsum_finalize.argtypes = (ctypes.POINTER(fastsum_plan),)  # P

        if not self.init_done:
            raise ValueError("FASTSUM not initialized.")

        if not self.finalized:
            self.finalized = True
            _fastsumlib.jfastsum_finalize(self.plan)

    def finalize_plan(self):
        """
        Alternative call for fastsum_finalize_plan()
        """
        return self.fastsum_finalize_plan()

    def fastsum_init(self):
        """
        Initializes a fastsum  plan.
        This function does not have to be called by the user.
        """
        # Convert c to numpy array for passing them to C
        Cv = np.array(self.c, dtype=np.float64)

        # Call init for memory allocation
        ptr = _fastsumlib.jfastsum_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(fastsum_plan))

        # Initialize values
        code = _fastsumlib.jfastsum_init(
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
            ctypes.c_int(self.m_y),
        )
        self.init_done = True

        if code == 1:
            raise RuntimeError("Unkown kernel")

    def init(self):
        """
        Alternative call for fastsum_init()
        """
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
            norm_x = np.linalg.norm(X_fort, axis=1)
            max_allowed_norm = 0.5 * (0.5 - self.eps_B)
            if np.any(norm_x > max_allowed_norm):
                raise ValueError(
                    f"All x values must satisfy the norm condition: ||x_k|| <= {max_allowed_norm:.4f}"
                )
            if self.d == 1:
                if not (isinstance(value, np.ndarray) and value.dtype == np.float64):
                    raise RuntimeError("x has to be a numpy float64 array")
                if not value.flags["C"]:
                    raise RuntimeError("x has to be C-continuous")
                if value.size != self.N:
                    raise ValueError(f"x has to be an array of size {self.N}")
                _fastsumlib.jfastsum_set_x.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=self.N
                )
            else:
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.float64
                    or value.ndim != 2
                ):
                    raise ValueError("x has to be a Float64 matrix.")
                if value.shape[0] != self.N or value.shape[1] != self.d:
                    raise ValueError(f"x has to be a Float64 matrix of size {self.N}")
                _fastsumlib.jfastsum_set_x.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=(self.N, self.d)
                )
            self._X = _fastsumlib.jfastsum_set_x(self.plan, X_fort)

    @property
    def y(self):
        return np.ascontiguousarray(self._Y).T

    @y.setter
    def y(self, value):
        if value is not None:
            if self.init_done is False:
                self.init()
            Y_fort = np.asfortranarray(value)
            norm_y = np.linalg.norm(Y_fort, axis=1)
            max_allowed_norm = 0.5 * (0.5 - self.eps_B)
            if np.any(norm_y > max_allowed_norm):
                raise ValueError(
                    f"All y values must satisfy the norm condition: ||y_k|| <= {max_allowed_norm:.4f}"
                )
            if self.d == 1:
                if not (isinstance(value, np.ndarray) and value.dtype == np.float64):
                    raise RuntimeError("y has to be a numpy float64 array")
                if not value.flags["C"]:
                    raise RuntimeError("y has to be C-continuous")
                if value.size != self.M:
                    raise ValueError(f"y has to be an array of size {self.M}")
                _fastsumlib.jfastsum_set_y.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=self.M
                )
            else:
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.float64
                    or value.ndim != 2
                ):
                    raise ValueError("y has to be a Float64 matrix.")
                if value.shape[0] != self.M or value.shape[1] != self.d:
                    raise ValueError(f"y has to be a Float64 matrix of size {self.M}")
                _fastsumlib.jfastsum_set_y.restype = np.ctypeslib.ndpointer(
                    dtype=np.float64, ndim=2, shape=(self.M, self.d)
                )
            self._Y = _fastsumlib.jfastsum_set_y(self.plan, Y_fort)

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

            _fastsumlib.jfastsum_set_alpha.restype = np.ctypeslib.ndpointer(
                np.complex128, shape=self.N
            )
            self._Alpha = _fastsumlib.jfastsum_set_alpha(self.plan, alpha_fort)

    def fastsum_trafo(self):
        """
        Performs fast NFFT-based summation.
        """
        _fastsumlib.jfastsum_trafo.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=self.M, flags="F"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("FASTSUM already finalized")

        if not hasattr(self, "y"):
            raise ValueError("y has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        if not hasattr(self, "alpha"):
            raise ValueError("alpha has not been set.")

        ptr = _fastsumlib.jfastsum_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        """
        Alternative call for fastsum_trafo()
        """
        return self.fastsum_trafo()

    def fastsum_trafo_exact(self):
        """
        Performs direct computation of sums.
        """
        _fastsumlib.jfastsum_exact.restype = np.ctypeslib.ndpointer(
            np.complex128, shape=self.M, flags="F"
        )
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("FASTSUM already finalized")

        if not hasattr(self, "y"):
            raise ValueError("y has not been set.")

        if not hasattr(self, "x"):
            raise ValueError("x has not been set.")

        if not hasattr(self, "alpha"):
            raise ValueError("alpha has not been set.")

        ptr = _fastsumlib.jfastsum_exact(self.plan)
        self.f = ptr

    def trafoexact(self):
        """
        Alternative call for fastsum_trafo_exact()
        """
        return self.fastsum_trafo_exact()
