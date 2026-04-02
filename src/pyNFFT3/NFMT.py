import numpy as np

from .flags import BASES


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


def _check_nodes_for_bases(X, bases):
    for j, basis in enumerate(bases):
        col = X[:, j]
        if BASES[basis] == 0:
            if np.min(col) < -0.5 or np.max(col) >= 0.5:
                raise ValueError(
                    f"Nodes for exp basis in dimension {j} must be in [-0.5, 0.5)."
                )
        elif BASES[basis] in {1, 2}:
            if np.min(col) < 0.0 or np.max(col) > 1.0:
                raise ValueError(
                    f"Nodes for {basis} basis in dimension {j} must be in [0, 1]."
                )
        else:
            raise ValueError(f"Unknown basis id for {basis}")


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


def _expand_reduced_to_full_fhat(fhat_reduced, N, bases):
    d = len(N)
    p = int(np.prod(N))
    pv_exp = _prefix_products_full(N)
    pv_red = _prefix_products_reduced(N, bases)
    full = np.zeros(p, dtype=np.complex128)

    for i in range(p):
        idx_red = 0
        e = 0
        valid = True
        for j in range(d):
            basis = bases[j]
            n_j = int(N[j])
            k_exp = (i // pv_exp[j]) % n_j
            if BASES[basis] > 0:
                k = abs(k_exp - (n_j // 2))
                if k == n_j // 2:
                    valid = False
                    break
                if k != 0:
                    e -= 1
                idx_red += k * pv_red[j]
            else:
                idx_red += k_exp * pv_red[j]

        if valid:
            full[i] = (np.sqrt(2.0) ** e) * fhat_reduced[idx_red]

    return full


def nfmt_index_set_without_zeros(N, bases):
    N = _validate_bandwidths(N)
    bases = _validate_bases(bases, len(N))

    ranges = []
    for bw, basis in zip(N, bases):
        if BASES[basis] == 0:
            # Julia semantics: {-N/2, ..., N/2-1}
            half = int(bw // 2)
            ranges.append(list(range(-half, half)))
        else:
            # Julia semantics for cos/alg: {0, ..., N/2-1}
            ranges.append(list(range(0, int(bw // 2))))

    if len(N) == 0:
        return np.array([0], dtype=np.int64)
    if len(N) == 1:
        return np.array(ranges[0], dtype=np.int64)

    mesh = np.array(np.meshgrid(*ranges, indexing="ij"), dtype=np.int64)
    return mesh.reshape(len(N), -1)


def _mixed_matrix(N, X, bases):
    N = _validate_bandwidths(N)
    bases = _validate_bases(bases, len(N))

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] != len(N):
        raise ValueError("X must have shape (M, d) with d == len(N)")

    _check_nodes_for_bases(X, bases)

    freq = nfmt_index_set_without_zeros(N, bases)
    if freq.ndim == 1:
        freq = freq.reshape(1, -1)

    M, d = X.shape
    nf = freq.shape[1]
    F = np.ones((M, nf), dtype=np.complex128)

    for j in range(d):
        n_j = freq[j]
        x_j = X[:, j][:, None]
        if BASES[bases[j]] == 0:
            # exp
            F *= np.exp(-2.0j * np.pi * x_j * n_j)
        elif BASES[bases[j]] == 1:
            # cos basis on [0, 1]
            factor = np.where(n_j == 0, 1.0, np.sqrt(2.0))
            F *= factor * np.cos(np.pi * x_j * n_j)
        elif BASES[bases[j]] == 2:
            # alg basis on [0, 1]
            factor = np.where(n_j == 0, 1.0, np.sqrt(2.0))
            F *= factor * np.cos(n_j * np.arccos(2.0 * x_j - 1.0))
        else:
            raise ValueError(f"Unknown basis id for {bases[j]}")

    return F


class NFMT:
    def __init__(self, N, M, bases, n=None, m=None, f1=None, f2=None):
        self.N = _validate_bandwidths(N)
        self.M = int(M)
        self.D = len(self.N)
        self.bases = _validate_bases(bases, self.D)
        self.n = n
        self.m = m
        self.f1 = f1
        self.f2 = f2

        if self.M <= 0:
            raise ValueError("M must be positive")

        self._x = None
        self._f = None
        self._fhat = None
        self._fhat_full = None
        self._matrix = None
        self.finalized = False

    def __del__(self):
        self.finalize_plan()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if value is None:
            self._x = None
            self._matrix = None
            return
        arr = np.asarray(value, dtype=np.float64, order="C")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape != (self.M, self.D):
            raise ValueError(f"x must have shape ({self.M}, {self.D})")

        # Explicit basis-dependent node checks to mirror Julia's NFMT setter logic.
        for j, basis in enumerate(self.bases):
            col = arr[:, j]
            if BASES[basis] == 0:
                if np.min(col) < -0.5 or np.max(col) >= 0.5:
                    raise ValueError(
                        f"Nodes for exp basis in dimension {j} must be in [-0.5, 0.5)."
                    )
            elif BASES[basis] in {1, 2}:
                if np.min(col) < 0.0 or np.max(col) > 1.0:
                    raise ValueError(
                        f"Nodes for {basis} basis in dimension {j} must be in [0, 1]."
                    )
            else:
                raise ValueError(f"Unknown basis id for {basis}")

        self._x = arr
        self._matrix = _mixed_matrix(self.N, self._x, self.bases)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        if value is None:
            self._f = None
            return
        arr = np.asarray(value, dtype=np.complex128, order="C")
        if arr.ndim != 1 or arr.shape[0] != self.M:
            raise ValueError(f"f must be one-dimensional with length {self.M}")
        self._f = arr

    @property
    def fhat(self):
        return self._fhat

    @fhat.setter
    def fhat(self, value):
        a = sum(BASES[b] > 0 for b in self.bases)
        nf = int(np.prod(self.N) // (2**a))
        if value is None:
            self._fhat = None
            self._fhat_full = None
            return
        arr = np.asarray(value, dtype=np.complex128, order="C")
        if arr.ndim != 1 or arr.shape[0] != nf:
            raise ValueError(f"fhat must be one-dimensional with length {nf}")

        # Explicit basis-dependent coefficient handling like in Julia:
        # build the internally expanded coefficient representation.
        self._fhat_full = _expand_reduced_to_full_fhat(arr, self.N, self.bases)
        self._fhat = arr

    def nfmt_init(self):
        return None

    def init(self):
        return self.nfmt_init()

    def nfmt_finalize_plan(self):
        self.finalized = True
        return None

    def finalize_plan(self):
        return self.nfmt_finalize_plan()

    def nfmt_trafo(self):
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        if self._matrix is None:
            raise ValueError("x has not been set")
        if self._fhat is None:
            raise ValueError("fhat has not been set")
        self._f = self._matrix @ self._fhat
        return self._f

    def trafo(self):
        return self.nfmt_trafo()

    def nfmt_adjoint(self):
        if self.finalized:
            raise RuntimeError("NFMT already finalized")
        if self._matrix is None:
            raise ValueError("x has not been set")
        if self._f is None:
            raise ValueError("f has not been set")
        self._fhat = self._matrix.conj().T @ self._f
        return self._fhat

    def adjoint(self):
        return self.nfmt_adjoint()

    def nfmt_trafo_direct(self):
        return self.nfmt_trafo()

    def trafo_direct(self):
        return self.nfmt_trafo_direct()

    def nfmt_adjoint_direct(self):
        return self.nfmt_adjoint()

    def adjoint_direct(self):
        return self.nfmt_adjoint_direct()