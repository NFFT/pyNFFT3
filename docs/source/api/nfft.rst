``pyNFFT3.NFFT`` - core NFFT functionalities
=============================================

.. automodule:: pyNFFT3

NFFT Class
----------

.. autoclass:: pyNFFT3.NFFT
   :members: 
   :undoc-members: 
   :exclude-members: f, fhat, x, num_threads

   .. attribute:: N

      The multiband limit of the trigonometric polynomial *f*.

   .. attribute:: M

      The number of nodes.

   .. attribute:: n

      The oversampling per dimension.

   .. attribute:: m

      The window size. A larger m results in more accuracy but at a higher computational cost.

   .. attribute:: f1

      The NFFT flags.

   .. attribute:: f2

      The FFTW flags.

   .. attribute:: x

      Array for sampling nodes.

   .. attribute:: f

      Array for NFFT values or coefficients for the adjoint NFFT.

   .. attribute:: fhat

      The Fourier coefficients for the NFFT or values for the adjoint NFFT.

   .. attribute:: D

      The number of dimensions, which is equal to the length of **N**.

   .. attribute:: init_done

      Boolean to indicate if the NFFT plan is initialized.

   .. attribute:: finalized

      Boolean to indicate if the NFFT plan is finalized.