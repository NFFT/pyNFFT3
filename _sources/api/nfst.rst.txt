``pyNFFT3.NFST`` - NFST Class
=============================================

NFST Methods
-------------

.. autoclass:: pyNFFT3.NFST
   :members: 
   :undoc-members:
   :member-order: bysource
   :exclude-members: f, fhat, x, num_threads, __init__

NFST Attributes
----------------

   .. attribute:: plan <nfst_plan>

      NFST plan (C pointer)

   .. attribute:: N <numpy.ndarray>

      The multiband limit :math:`(N_1,N_2,...,N_D)` of the trigonometric polynomial :math:`f^s`. Must contain positive integers.

   .. attribute:: M <int>

      The number of nodes. Must be a positive integer.

   .. attribute:: n <numpy.ndarray>

      The oversampling :math:`(n_1,n_2,...,n_D)` per dimension. 

   .. attribute:: m <int>

      The window size. A larger m results in more accuracy but at a higher computational cost.

   .. attribute:: f1 <c_uint32>

      The NFST flags.

   .. attribute:: f2 <c_unit32>

      The FFTW flags.

   .. attribute:: x <numpy.ndarray>

      Float array for sampling nodes.

   .. attribute:: f <numpy.ndarray>

      Float array for NFST values or coefficients for the adjoint NFST.

   .. attribute:: fhat <numpy.ndarray>

      Float array of Fourier coefficients for the NFST or values for the adjoint NFST.

   .. attribute:: D <int>

      The number of dimensions, which is equal to the length of **N**.

   .. attribute:: init_done <boolean>

      Boolean to indicate if the NFST plan is initialized.

   .. attribute:: finalized <boolean>

      Boolean to indicate if the NFST plan is finalized.