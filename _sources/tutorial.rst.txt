Tutorials
==========

Using NFFT
----------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/NFFT_test.py>`_
for a detailed example of all function uses and error tests.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/nfft.rst>`_
for the class methods and attributes.

To start using NFFT, first import the class:

.. code-block:: python

    from pynfft3 import NFFT

You can then generate a plan with your desired multiband limit values and number of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = np.array([16, 8], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5
    plan = NFFT(N, M)

To compute the NDFT using **trafo()** or **trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    plan.fhat = np.array([0.1+0.1j, 0.2-0.2j, 0.3+0.3j, 0.4-0.4j, 0.5+0.5j, 0.6-0.6j, 0.7+0.7j, 0.8-0.8j])
    plan.trafo()
    # or
    plan.trafo_direct()

To compute the adjoint NDFT using **adjoint()** or **adjoint_direct()**, both **x** and **f** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    f = np.array([1.0+1.0j, 1.1-1.1j, 1.2+1.2j, 1.3-1.3j, 1.4+1.4j])
    plan.adjoint()
    # or
    plan.adjoint_direct()

