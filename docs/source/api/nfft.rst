``pyNFFT3.NFFT`` - NFFT Class
=============================================

NFFT Methods
--------------

.. autoclass:: pyNFFT3.NFFT
   :members: 
   :undoc-members:
   :member-order: bysource
   :exclude-members: f, fhat, x, num_threads, __init__

NFFT Attributes
----------------

   .. attribute:: plan <nfft_plan>

      NFFT plan (C pointer)

   .. attribute:: N <numpy.ndarray>

      The multiband limit :math:`(N_1,N_2,...,N_D)` of the trigonometric polynomial :math:`f`. Must contain positive and even integers.

   .. attribute:: M <int>

      The number of nodes. Must be a positive integer.

   .. attribute:: n <numpy.ndarray>

      The oversampling :math:`(n_1,n_2,...,n_D)` per dimension. 

   .. attribute:: m <int>

      The window size. A larger m results in more accuracy but at a higher computational cost.

   .. attribute:: f1 <c_uint32>

      The NFFT flags.

   .. attribute:: f2 <c_unit32>

      The FFTW flags.

   .. attribute:: x <numpy.ndarray>

      Float array for sampling nodes.

   .. attribute:: f <numpy.ndarray>

      Complex array for NFFT values or coefficients for the adjoint NFFT.

   .. attribute:: fhat <numpy.ndarray>

      Complex array of Fourier coefficients for the NFFT or values for the adjoint NFFT.

   .. attribute:: D <int>

      The number of dimensions, which is equal to the length of **N**.

   .. attribute:: init_done <boolean>

      Boolean to indicate if the NFFT plan is initialized.

   .. attribute:: finalized <boolean>

      Boolean to indicate if the NFFT plan is finalized.