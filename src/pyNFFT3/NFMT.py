import numpy as np

from .flags import BASES, default_window_cut_off, f2_default
from .NFCT import NFCT
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


def _all_cos_alg(bases):
    """True if no dimension uses the exp basis, i.e. the transform is
    real-valued and can be realised with a plain NFCT instead of an NFFT."""
    return all(BASES[b] > 0 for b in bases)


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
    p = int(np.prod(N))
    pv_exp = _prefix_products_full(N)
    pv_red = _prefix_products_reduced(N, bases)
    reduced = np.zeros(_reduced_size(N, bases), dtype=np.complex128)

    for i in range(p):
        idx_red, e, valid = _decode_full_index(i, N, bases, pv_exp, pv_red)
        if valid:
            reduced[idx_red] += (np.sqrt(2.0) ** e) * fhat_full[i]

    return reduced


def _cos_alg_scale_factors(N, bases):
    """
    sqrt(2)**e per reduced index, e = number of dimensions with k_j != 0 -
    the mixed-basis normalisation needed on top of a plain NFCT. Only valid
    when every dimension is cos/alg (_all_cos_alg(bases)): NFCT's own N_j is
    then exactly N_j/2 (see _real_backend_bandwidths) and its native linear
    index already equals this module's reduced index one for one (same
    row-major strides, see _prefix_products_reduced), with no folding or
    Nyquist exclusion needed - contrast with _expand_reduced_to_full_fhat,
    which is needed whenever an exp dimension is mixed in.
    """
    nf = _reduced_size(N, bases)
    pv_red = _prefix_products_reduced(N, bases)
    half_N = [int(n) // 2 for n in N]
    scale = np.empty(nf, dtype=np.float64)
    for i in range(nf):
        e = 0
        for j in range(len(N)):
            k_j = (i // pv_red[j]) % half_N[j]
            if k_j != 0:
                e += 1
        scale[i] = np.sqrt(2.0) ** e
    return scale


def _real_backend_bandwidths(N):
    half = [int(n) // 2 for n in N]
    if any(h % 2 != 0 for h in half):
        raise ValueError(
            f"Real-valued NFMT (NFCT backend) requires N/2 to be even in "
            f"every dimension, since NFCT itself needs an even bandwidth; "
            f"got N={list(N)} -> N/2={half}. Use an N that is a multiple of "
            f"4 in every dimension, or pass complex128 fhat/f to use the "
            f"general NFFT-based path instead."
        )
    return tuple(half)


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


def _node_to_backend_domain(col, basis):
    """
    Maps nodes into the native domain [-0.5, 0.5) (NFFT) resp. [0, 0.5]
    (NFCT) - both backends share this image exactly for cos/alg dimensions,
    so this transform is the same regardless of which backend ends up being
    used (see NFMT.jl's Base.setproperty! for :x in NFFT3.jl).
    """
    b = BASES[basis]
    if b == 0:  # exp: already on the NFFT's native domain [-0.5, 0.5)
        return col
    elif b == 1:  # cos on [0, 1] -> native domain via a half-scale
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

    Internally this reduces the mixed transform to a plain NFFT or NFCT: node
    coordinates are remapped into the backend's native domain per dimension
    (cos/alg nodes via a scale/arccos substitution), and the one-sided
    mixed-basis coefficients are expanded into/folded back from the backend's
    own coefficient array - see nfmt.jl in NFFT3.jl for the reference this
    mirrors. trafo()/adjoint() therefore run the real fast algorithm, and
    trafo_direct()/adjoint_direct() the real direct transform, both from the
    underlying C library - unlike a dense matrix-vector reimplementation,
    this scales like NFFT/NFCT, not like a naive O(M*N) sum.

    fhat/f may be given as either a complex128 or a real float64 array. The
    backend is chosen lazily from whichever of fhat/f is assigned first: a
    real array is realised with a plain NFCT (cheaper: half the data, real
    arithmetic throughout) whenever every dimension uses a cos/alg basis -
    the only case where the result is guaranteed real - and with an NFFT
    (upcast to complex128) otherwise, e.g. when any dimension is exp, or
    when complex data is passed explicitly. Once chosen for a plan, the
    backend cannot change: assigning complex data to a plan that already
    picked the real NFCT backend raises an error.
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

        self._n = n
        self._m = m
        self._f1 = f1
        self._f2 = f2
        self._backend = None  # "nfft" or "nfct", chosen lazily - see fhat/f setters
        self._plan = None  # the wrapped NFFT or NFCT instance
        self._x_raw = None  # cached, untransformed nodes pending a backend choice
        self.finalized = False

    def __del__(self):
        self.finalize_plan()

    def nfmt_finalize_plan(self):
        """
        Finalizes an NFMT plan (and the NFFT/NFCT plan it wraps).
        This function does not have to be called by the user.
        """
        # getattr-guarded: __del__ runs even if __init__ raised before
        # self.finalized/self._plan were ever assigned (e.g. invalid N/M/bases).
        if getattr(self, "finalized", True):
            return
        self.finalized = True
        plan = getattr(self, "_plan", None)
        if plan is not None and plan.init_done:
            plan.finalize_plan()

    def finalize_plan(self):
        """
        Alternate call for **nfmt_finalize_plan()**
        """
        return self.nfmt_finalize_plan()

    def _ensure_backend(self, want_complex: bool):
        if self._backend == "nfct" and want_complex:
            raise RuntimeError(
                "This NFMT plan already uses the real-valued NFCT backend "
                "(its first fhat/f assignment was real); it cannot accept "
                "complex data afterwards. Create a new NFMT plan to use "
                "complex coefficients on it."
            )
        if self._backend is not None:
            return

        if want_complex or not _all_cos_alg(self.bases):
            self._backend = "nfft"
            self._plan = NFFT(
                tuple(int(v) for v in self.N),
                self.M,
                n=self._n,
                m=self._m,
                f1=self._f1,
                f2=self._f2,
            )
        else:
            self._backend = "nfct"
            self._plan = NFCT(
                _real_backend_bandwidths(self.N),
                self.M,
                n=self._n,
                m=self._m,
                f1=self._f1,
                f2=self._f2,
            )

        if self._x_raw is not None:
            self._plan.x = np.ascontiguousarray(self._transform_x(self._x_raw))

    def _transform_x(self, arr: np.ndarray) -> np.ndarray:
        if self.D == 1:
            return _node_to_backend_domain(arr, self.bases[0])
        transformed = np.empty_like(arr)
        for j, basis in enumerate(self.bases):
            transformed[:, j] = _node_to_backend_domain(arr[:, j], basis)
        return transformed

    @property
    def num_threads(self) -> int:
        if self._plan is None:
            raise ValueError("NFMT not initialized.")
        return self._plan.num_threads

    @property
    def x(self) -> np.ndarray:
        return self._x_raw

    @x.setter
    def x(self, value: np.ndarray):
        if value is None:
            self._x_raw = None
            return
        if self.finalized:
            raise RuntimeError("Plan already finalized")

        arr = np.asarray(value, dtype=np.float64)
        if self.D == 1:
            arr = arr.reshape(-1)
            if arr.shape != (self.M,):
                raise ValueError(f"x must have shape ({self.M},)")
            _check_node_domain(arr, self.bases[0], 0)
        else:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape != (self.M, self.D):
                raise ValueError(f"x must have shape ({self.M}, {self.D})")
            for j, basis in enumerate(self.bases):
                _check_node_domain(arr[:, j], basis, j)

        self._x_raw = arr.copy()
        if self._plan is not None:
            self._plan.x = np.ascontiguousarray(self._transform_x(self._x_raw))

    @property
    def f(self) -> np.ndarray:
        if self._plan is None:
            return None
        return self._plan.f

    @f.setter
    def f(self, value: np.ndarray):
        if value is None:
            if self._plan is not None:
                self._plan.f = None
            return
        if self.finalized:
            raise RuntimeError("Plan already finalized")
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.shape[0] != self.M:
            raise ValueError(f"f must be a one-dimensional array of size {self.M}")
        self._ensure_backend(np.iscomplexobj(arr))
        if self._backend == "nfft":
            self._plan.f = np.ascontiguousarray(arr, dtype=np.complex128)
        else:
            self._plan.f = np.ascontiguousarray(arr, dtype=np.float64)

    @property
    def fhat(self) -> np.ndarray:
        if self._plan is None:
            return None
        raw = self._plan.fhat
        if raw is None:
            return None
        if self._backend == "nfft":
            return _reduce_full_to_reduced_fhat(raw, self.N, self.bases)
        return raw * _cos_alg_scale_factors(self.N, self.bases)

    @fhat.setter
    def fhat(self, value: np.ndarray):
        if value is None:
            if self._plan is not None:
                self._plan.fhat = None
            return
        if self.finalized:
            raise RuntimeError("Plan already finalized")
        arr = np.asarray(value)
        nf = _reduced_size(self.N, self.bases)
        if arr.ndim != 1 or arr.shape[0] != nf:
            raise ValueError(f"fhat must be a one-dimensional array of size {nf}")

        self._ensure_backend(np.iscomplexobj(arr))
        if self._backend == "nfft":
            full = _expand_reduced_to_full_fhat(
                arr.astype(np.complex128, copy=False), self.N, self.bases
            )
            self._plan.fhat = full
        else:
            scale = _cos_alg_scale_factors(self.N, self.bases)
            self._plan.fhat = np.ascontiguousarray(
                arr.astype(np.float64, copy=False) * scale
            )

    def nfmt_trafo(self):
        """
        Computes the NFMT using the fast approximate algorithm (NFFT or
        NFCT, depending on the backend) for the provided nodes in **x** and
        coefficients in **fhat**.
        """
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        if self._plan is None:
            raise ValueError("fhat has not been set.")
        self._plan.trafo()
        return self.f

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
        if self._plan is None:
            raise ValueError("fhat has not been set.")
        self._plan.trafo_direct()
        return self.f

    def trafo_direct(self):
        """
        Alternative call for **nfmt_trafo_direct()**
        """
        return self.nfmt_trafo_direct()

    def nfmt_adjoint(self):
        """
        Computes the adjoint NFMT using the fast approximate algorithm (NFFT
        or NFCT, depending on the backend) for the provided nodes in **x**
        and coefficients in **f**.
        """
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        if self._plan is None:
            raise ValueError("f has not been set.")
        self._plan.adjoint()
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
        if self._plan is None:
            raise ValueError("f has not been set.")
        self._plan.adjoint_direct()
        return self.fhat

    def adjoint_direct(self):
        """
        Alternative call for **nfmt_adjoint_direct()**
        """
        return self.nfmt_adjoint_direct()
