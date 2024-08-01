Tutorials
==========

Using NFFT
----------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/NFFT_test.py>`_
for a detailed example of all function uses.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/nfft.rst>`_
for the class methods and attributes.

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
