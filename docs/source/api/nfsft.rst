``pyNFFT3.NFSFT`` - NFSFT Class
=============================================

NFSFT Methods
-------------

.. autoclass:: pyNFFT3.NFSFT
   :members: 
   :undoc-members:
   :member-order: bysource
   :exclude-members: f, fhat, x, __init__

NFSFT Attributes
----------------

   .. attribute:: plan <nfsft_plan>

      NFSFT plan (C pointer)

   .. attribute:: N <int>

      The bandwidth :math:`N \epsilon N_0`. Must be a positive integer.

   .. attribute:: M <int>

      The number of nodes. Must be a positive integer.

   .. attribute:: flags <ctypes.c_uint32>

      The NFSFT flags.

   .. attribute:: nfft_flags <ctypes.c_unit32>

      The NFFT flags.

   .. attribute:: nfft_cutoff <int>

      The NFFT cutoff.

   .. attribute:: x <numpy.ndarray>

      Float array for sampling nodes.

   .. attribute:: f <numpy.ndarray>

      Complex array for NFSFT values or coefficients for the adjoint NFSFT.

   .. attribute:: fhat <numpy.ndarray>

      Complex array of spherical Fourier coefficients for the NFSFT or values for the adjoint NFSFT.

   .. attribute:: init_done <boolean>

      Boolean to indicate if the NFSFT plan is initialized.

   .. attribute:: finalized <boolean>

      Boolean to indicate if the NFSFT plan is finalized.

NFSFT Flags
------------
.. autodata:: pyNFFT3.flags.NFSFT_NORMALIZED
.. autodata:: pyNFFT3.flags.NFSFT_USE_NDFT
.. autodata:: pyNFFT3.flags.NFSFT_USE_DPT
.. autodata:: pyNFFT3.flags.NFSFT_MALLOC_X
.. autodata:: pyNFFT3.flags.NFSFT_MALLOC_F_HAT
.. autodata:: pyNFFT3.flags.NFSFT_MALLOC_F
.. autodata:: pyNFFT3.flags.NFSFT_PRESERVE_F_HAT
.. autodata:: pyNFFT3.flags.NFSFT_PRESERVE_X
.. autodata:: pyNFFT3.flags.NFSFT_PRESERVE_F
.. autodata:: pyNFFT3.flags.NFSFT_DESTROY_F_HAT
.. autodata:: pyNFFT3.flags.NFSFT_DESTROY_X
.. autodata:: pyNFFT3.flags.NFSFT_DESTROY_F
.. autodata:: pyNFFT3.flags.NFSFT_NO_DIRECT_ALGORITHM 
.. autodata:: pyNFFT3.flags.NFSFT_NO_FAST_ALGORITHM
.. autodata:: pyNFFT3.flags.NFSFT_ZERO_F_HAT
.. autodata:: pyNFFT3.flags.NFSFT_EQUISPACED
.. autodata:: pyNFFT3.flags.nfsft_default
.. autodata:: pyNFFT3.flags.nfsft_nfft_default
.. autodata:: pyNFFT3.flags.nfsft_default_nfft_cut_off
.. autodata:: pyNFFT3.flags.nfsft_default_threshold