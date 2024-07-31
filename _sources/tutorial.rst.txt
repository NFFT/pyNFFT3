Tutorials
==========

Here you will find example usage for:

(table of contents here)

Using NFFT
----------

Fields:
*******

- **N**: the multiband limit of the trigonometric polynomial *f*.
- **M**: the number of nodes.
- **n**: the oversampling per dimension.
- **m**: the window size. A larger m results in more accuracy but at a higher computational cost.
- **f1**: the NFFT flags.
- **f2**: the FFTW flags.
- **x**: array for sampling nodes.
- **f**: array for NFFT values or coefficients for the adjoint NFFT.
- **fhat**: the Fourier coefficients for the NFFT or values for the adjoint NFFT.
- **D**: the number of dimensions, is equal to the length of **N**.
- **init_done**: boolean to indicate if the NFFT plan is initialized.
- **finalized**: boolean to indicate if the NFFT plan is finalized.

Methods:
********

- **__init__**: constructor for the NFFT plan.
- **__del__**: destructor for the NFFT plan.
- **finalize_plan**: finalizes an NFFT plan. This function does not have to be called by the user.
- **init**: initializes the NFFT plan in C. This function does not have to be called by the user.
- **trafo**: computes the NDFT using the fast NFFT algorithm.
- **trafo_direct**: computes the NDFT via naive matrix-vector multiplication.
- **adjoint**: computes the adjoint NDFT using the fast adjoint NFFT algorithm.
- **adjoint_direct**: computes the adjoint NDFT using naive matrix-vector multiplication.

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/NFFT_test.py>`_
for a detailed example of all function uses.

To start using NFFT, first import the class:

.. code-block:: python

    from pynfft3 import NFFT

You can then generate a plan with your desired multiband limit values and nubmer of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = np.array([16, 8], dtype='int32')  # 2 dimensions
    M = 100
    plan = NFFT(N, M)

To compute the NDFT using **trafo** or **trafo_direct**, both **x** and **fhat** must be set:

.. code-block:: python

    X = np.array([[abs(np.sin(i + j)) for j in range(d)] for i in range(M)])
    fhat = np.array([np.cos(k) + 1.0j * np.sin(k) for k in range(Ns)])
    plan.trafo()
