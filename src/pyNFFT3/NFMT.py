import numpy as np

from .flags import BASES, default_window_cut_off, f2_default
from .NFFT import NFFT


def _validate_bandwidths(N):
    N = np.asarray(N, dtype=np.int32)
    if N.ndim != 1:
        raise ValueError("N has to be a one-dimensional int32 array")
    if np.any(N <= 1):
        raise ValueError("All bandwidth entries must be > 1")
    if np.any(N % 2 != 0):
        raise ValueError("All bandwidth entries must be even")
    return N


def _validate_bases(bases, d):
    if bases is None or len(bases) != d:
        raise ValueError("bases must have one entry per dimension")
    for basis in bases:
        if basis not in BASES:
            raise ValueError(f"Unknown basis '{basis}'. Allowed: {list(BASES.keys())}")
    return list(bases)


def nfmt_index_set_without_zeros(N, bases):
    """
    The one-sided (reduced) frequency index set for the mixed basis: the full
    range {-N/2, ..., N/2-1} for an exp dimension, but only {0, ..., N/2-1}
    for a cos/alg dimension (cos(k*.) is symmetric in k, so +-k carry the same
    coefficient and only one of them needs to be stored).
    """
    N = _validate_bandwidths(N)
    bases = _validate_bases(bases, len(N))

    ranges = []
    for bw, basis in zip(N, bases):
        if BASES[basis] == 0:
            half = int(bw // 2)
            ranges.append(list(range(-half, half)))
        else:
            ranges.append(list(range(0, int(bw // 2))))

    if len(N) == 0:
        return np.array([0], dtype=np.int64)
    if len(N) == 1:
        return np.array(ranges[0], dtype=np.int64)

    mesh = np.array(np.meshgrid(*ranges, indexing="ij"), dtype=np.int64)
    return mesh.reshape(len(N), -1)


def _reduced_size(N, bases):
    a = sum(BASES[b] > 0 for b in bases)
    return int(np.prod(N) // (2**a))


def _prefix_products_full(N):
    d = len(N)
    pv = np.ones(d, dtype=np.int64)
    for i in range(d - 2, -1, -1):
        pv[i] = pv[i + 1] * int(N[i + 1])
    return pv


def _prefix_products_reduced(N, bases):
    d = len(N)
    pv = np.ones(d, dtype=np.int64)
    for i in range(d - 2, -1, -1):
        n_next = int(N[i + 1])
        if BASES[bases[i + 1]] > 0:
            n_next //= 2
        pv[i] = pv[i + 1] * n_next
    return pv


def _decode_full_index(i, N, bases, pv_exp, pv_red):
    """
    For linear index i into the FULL (NFFT-sized) fhat array, return
    (idx_red, e, valid): the matching index into the reduced array, the
    exponent for the sqrt(2)**e normalisation factor, and whether i
    corresponds to a Nyquist frequency that has no reduced counterpart
    (in which case it must read/contribute as zero).
    """
    idx_red = 0
    e = 0
    for j in range(len(N)):
        basis = bases[j]
        n_j = int(N[j])
        k_exp = (i // pv_exp[j]) % n_j
        if BASES[basis] > 0:
            k = abs(k_exp - (n_j // 2))
            if k == n_j // 2:
                return 0, 0, False
            if k != 0:
                e -= 1
            idx_red += k * pv_red[j]
        else:
            idx_red += k_exp * pv_red[j]
    return idx_red, e, True


def _expand_reduced_to_full_fhat(fhat_reduced, N, bases):
    """
    Build the full, NFFT-sized coefficient array from the reduced
    (one-sided) mixed-basis coefficients, so the actual transform can be
    carried out by a plain NFFT. See NFFT3.jl's NFMT.jl (Base.setproperty!
    for :fhat) for the reference this mirrors.
    """
    d = len(N)
    p = int(np.prod(N))
    pv_exp = _prefix_products_full(N)
    pv_red = _prefix_products_reduced(N, bases)
    full = np.zeros(p, dtype=np.complex128)

    for i in range(p):
        idx_red, e, valid = _decode_full_index(i, N, bases, pv_exp, pv_red)
        if valid:
            full[i] = (np.sqrt(2.0) ** e) * fhat_reduced[idx_red]

    return full


def _reduce_full_to_reduced_fhat(fhat_full, N, bases):
    """
    The inverse (adjoint) of _expand_reduced_to_full_fhat: folds the full,
    NFFT-sized coefficient array back down into the reduced mixed-basis
    representation by summing the contributions of every full index that
    maps to a given reduced index. Because of how the sqrt(2)**e factors are
    chosen, reduce(expand(x)) == x exactly - see NFFT3.jl's NFMT.jl
    (Base.getproperty for :fhat).
    """
    d = len(N)
    p = int(np.prod(N))
    pv_exp = _prefix_products_full(N)
    pv_red = _prefix_products_reduced(N, bases)
    reduced = np.zeros(_reduced_size(N, bases), dtype=np.complex128)

    for i in range(p):
        idx_red, e, valid = _decode_full_index(i, N, bases, pv_exp, pv_red)
        if valid:
            reduced[idx_red] += (np.sqrt(2.0) ** e) * fhat_full[i]

    return reduced


def _check_node_domain(col, basis, dim):
    if BASES[basis] == 0:
        if np.min(col) < -0.5 or np.max(col) >= 0.5:
            raise ValueError(
                f"Nodes for exp basis in dimension {dim} must be in [-0.5, 0.5)."
            )
    elif BASES[basis] in (1, 2):
        if np.min(col) < 0.0 or np.max(col) > 1.0:
            raise ValueError(
                f"Nodes for {basis} basis in dimension {dim} must be in [0, 1]."
            )
    else:
        raise ValueError(f"Unknown basis id for {basis}")


def _node_to_nfft_domain(col, basis):
    b = BASES[basis]
    if b == 0:  # exp: already on the NFFT's native domain [-0.5, 0.5)
        return col
    elif b == 1:  # cos on [0, 1] -> NFFT domain via a half-scale
        return col / 2.0
    elif b == 2:  # alg (Chebyshev-type) on [0, 1] via arccos substitution
        return np.arccos(2.0 * col - 1.0) / (2.0 * np.pi)
    else:
        raise ValueError(f"Unknown basis id for {basis}")


class NFMT:
    """
    Class to perform nonequispaced fast mixed transforms (NFMT): a
    D-dimensional trigonometric polynomial where each dimension independently
    uses an exp(onential), cos(ine) or alg(ebraic/Chebyshev) basis.

    Internally this reduces the mixed transform to a plain NFFT: node
    coordinates are remapped into the NFFT's native domain per dimension
    (cos/alg nodes via a scale/arccos substitution), and the one-sided
    mixed-basis coefficients are expanded into/folded back from the full,
    centered NFFT coefficient array - see nfmt.jl in NFFT3.jl for the
    reference this mirrors. trafo()/adjoint() therefore run the real fast
    NFFT algorithm, and trafo_direct()/adjoint_direct() the real direct NDFT,
    both from the underlying C library - unlike a dense matrix-vector
    reimplementation, this scales like NFFT, not like a naive O(M*N) sum.
    """

    def __init__(
        self,
        N: np.ndarray,
        M: int,
        bases,
        n: np.ndarray = None,
        m: int = default_window_cut_off,
        f1=None,
        f2=f2_default,
    ):
        self.N = _validate_bandwidths(N)
        self.D = len(self.N)
        self.bases = _validate_bases(bases, self.D)
        self.M = int(M)

        if self.M <= 0:
            raise ValueError("M must be positive")

        self._nfft = NFFT(
            tuple(int(v) for v in self.N), self.M, n=n, m=m, f1=f1, f2=f2
        )
        self._x = None  # cached, untransformed nodes as given by the caller
        self.finalized = False

    def __del__(self):
        self.finalize_plan()

    def nfmt_finalize_plan(self):
        """
        Finalizes an NFMT plan (and the NFFT plan it wraps).
        This function does not have to be called by the user.
        """
        # getattr-guarded: __del__ runs even if __init__ raised before
        # self.finalized/self._nfft were ever assigned (e.g. invalid N/M/bases).
        if getattr(self, "finalized", True):
            return
        self.finalized = True
        nfft = getattr(self, "_nfft", None)
        if nfft is not None and nfft.init_done:
            nfft.finalize_plan()

    def finalize_plan(self):
        """
        Alternate call for **nfmt_finalize_plan()**
        """
        return self.nfmt_finalize_plan()

    @property
    def num_threads(self) -> int:
        return self._nfft.num_threads

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, value: np.ndarray):
        if value is None:
            self._x = None
            return
        if self.finalized:
            raise RuntimeError("Plan already finalized")

        arr = np.asarray(value, dtype=np.float64)
        if self.D == 1:
            arr = arr.reshape(-1)
            if arr.shape != (self.M,):
                raise ValueError(f"x must have shape ({self.M},)")
            _check_node_domain(arr, self.bases[0], 0)
            transformed = _node_to_nfft_domain(arr, self.bases[0])
        else:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape != (self.M, self.D):
                raise ValueError(f"x must have shape ({self.M}, {self.D})")
            transformed = np.empty_like(arr)
            for j, basis in enumerate(self.bases):
                _check_node_domain(arr[:, j], basis, j)
                transformed[:, j] = _node_to_nfft_domain(arr[:, j], basis)

        self._nfft.x = np.ascontiguousarray(transformed)
        self._x = arr.copy()

    @property
    def f(self) -> np.ndarray:
        return self._nfft.f

    @f.setter
    def f(self, value: np.ndarray):
        self._nfft.f = value

    @property
    def fhat(self) -> np.ndarray:
        full = self._nfft.fhat
        if full is None:
            return None
        return _reduce_full_to_reduced_fhat(full, self.N, self.bases)

    @fhat.setter
    def fhat(self, value: np.ndarray):
        if value is None:
            self._nfft.fhat = None
            return
        arr = np.asarray(value, dtype=np.complex128)
        nf = _reduced_size(self.N, self.bases)
        if arr.ndim != 1 or arr.shape[0] != nf:
            raise ValueError(f"fhat must be a one-dimensional array of size {nf}")
        full = _expand_reduced_to_full_fhat(arr, self.N, self.bases)
        self._nfft.fhat = full

    def nfmt_trafo(self):
        """
        Computes the NFMT using the fast approximate NFFT algorithm for the
        provided nodes in **x** and coefficients in **fhat**.
        """
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        self._nfft.trafo()
        return self._nfft.f

    def trafo(self):
        """
        Alternative call for **nfmt_trafo()**
        """
        return self.nfmt_trafo()

    def nfmt_trafo_direct(self):
        """
        Computes the NDMT using direct (naive) transformation for the
        provided nodes in **x** and coefficients in **fhat**.
        """
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        self._nfft.trafo_direct()
        return self._nfft.f

    def trafo_direct(self):
        """
        Alternative call for **nfmt_trafo_direct()**
        """
        return self.nfmt_trafo_direct()

    def nfmt_adjoint(self):
        """
        Computes the adjoint NFMT using the fast approximate NFFT algorithm
        for the provided nodes in **x** and coefficients in **f**.
        """
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        self._nfft.adjoint()
        return self.fhat

    def adjoint(self):
        """
        Alternative call for **nfmt_adjoint()**
        """
        return self.nfmt_adjoint()

    def nfmt_adjoint_direct(self):
        """
        Computes the adjoint NDMT using direct (naive) transformation for the
        provided nodes in **x** and coefficients in **f**.
        """
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        self._nfft.adjoint_direct()
        return self.fhat

    def adjoint_direct(self):
        """
        Alternative call for **nfmt_adjoint_direct()**
        """
        return self.nfmt_adjoint_direct()
