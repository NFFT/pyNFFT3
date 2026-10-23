``pyNFFT3.NFMT`` - NFMT Class
=============================================

NFMT Methods
--------------

.. autoclass:: pyNFFT3.NFMT
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: f, fhat, x, num_threads, __init__

NFMT Attributes
----------------

   .. attribute:: N <numpy.ndarray>

      The multiband limit :math:`(N_1,N_2,...,N_D)` of the trigonometric polynomial. Must contain positive and even integers.

   .. attribute:: M <int>

      The number of nodes. Must be a positive integer.

   .. attribute:: bases <list of str>

      The basis (``"exp"``, ``"cos"`` or ``"alg"``) used in each dimension.

   .. attribute:: D <int>

      The number of dimensions, which is equal to the length of **N**.

   .. attribute:: x <numpy.ndarray>

      Float array for sampling nodes: ``[-0.5, 0.5)`` in an ``exp`` dimension, ``[0, 1]`` in a ``cos`` or ``alg`` dimension.

   .. attribute:: f <numpy.ndarray>

      Complex or (see Notes) real array for NFMT values or coefficients for the adjoint NFMT.

   .. attribute:: fhat <numpy.ndarray>

      Complex or (see Notes) real array of the one-sided (reduced) mixed-basis Fourier coefficients for the NFMT, or values for the adjoint NFMT.

   .. attribute:: finalized <boolean>

      Boolean to indicate if the NFMT plan (and the NFFT/NFCT plan it wraps) is finalized.

Notes
------

NFMT has no transform of its own in the underlying C library: it reduces the
mixed transform to a plain :doc:`NFFT <nfft>` or :doc:`NFCT <nfct>` by
remapping nodes per dimension (``cos``/``alg`` nodes via a scale/arccos
substitution) and expanding/scaling the one-sided ``fhat`` into the
backend's own coefficient array (see ``NFMT.jl`` in `NFFT3.jl
<https://github.com/NFFT/NFFT3.jl>`_ for the reference this mirrors).
:meth:`trafo`/:meth:`adjoint` therefore run the real, approximate fast
algorithm, and :meth:`trafo_direct`/:meth:`adjoint_direct` the real direct
transform - both from the underlying C library.

``fhat``/``f`` may be assigned as either a ``complex128`` or a real
``float64`` array; the backend is picked lazily from whichever is assigned
first. A real array picks the cheaper NFCT backend, but only when every
dimension uses a ``cos``/``alg`` basis (the only case where the result is
guaranteed real) - a complex array, or any ``exp`` dimension, always picks
the NFFT backend (upcasting real input to complex128 if needed). The NFCT
backend additionally requires ``N/2`` to be even in every dimension (NFCT's
own bandwidth requirement); otherwise use an ``N`` that is a multiple of 4,
or pass complex data to fall back to the NFFT backend. Once a plan has
picked the real NFCT backend, assigning it complex data raises an error -
create a new plan instead.
