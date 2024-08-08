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

    from pyNFFT3 import NFFT

You can then generate a plan with your desired multiband limit values and number of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = np.array([4, 2], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5 # 5 nodes
    plan = NFFT(N, M) # generate plan

To compute the NDFT using **trafo()** or **trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.fhat = np.array([0.1+0.1j, 0.2-0.2j, 0.3+0.3j, 0.4-0.4j, 0.5+0.5j, 0.6-0.6j, 0.7+0.7j, 0.8-0.8j]) # Fourier coefficients (numpy.prod(N) entries)
    plan.trafo()
    # or
    plan.trafo_direct()

To compute the adjoint NDFT using **adjoint()** or **adjoint_direct()**, both **x** and **f** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.f = np.array([1.0+1.0j, 1.1-1.1j, 1.2+1.2j, 1.3-1.3j, 1.4+1.4j]) # values for adjoint NFFT (M entries)
    plan.adjoint()
    # or
    plan.adjoint_direct()

Using NFCT
----------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/NFCT_test.py>`_
for a detailed example of all function uses and error tests.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/NFCT.rst>`_
for the class methods and attributes.

To start using NFCT, first import the class:

.. code-block:: python

    from pyNFFT3 import NFCT

You can then generate a plan with your desired multiband limit values and number of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = np.array([4, 2], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5 # 5 nodes
    plan = NFCT(N, M) # generate plan

To compute the NDCT using **trafo()** or **trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.fhat = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]) # Fourier coefficients (numpy.prod(N) entries)
    plan.trafo()
    # or
    plan.trafo_direct()

To compute the transposed NDCT using **nfct_transposed()** or **nfct_transposed_direct()**, both **x** and **f** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.f = np.array([1.0, 1.1, 1.2, 1.3, 1.4]) # values for adjoint NFFT (M entries)
    plan.nfct_transposed()
    # or
    plan.nfct_transposed_direct()

Using NFST
----------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/NFST_test.py>`_
for a detailed example of all function uses and error tests.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/NFST.rst>`_
for the class methods and attributes.

To start using NFST, first import the class:

.. code-block:: python

    from pyNFFT3 import NFST

You can then generate a plan with your desired multiband limit values and number of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = np.array([4, 2], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5 # 5 nodes
    plan = NFST(N, M) # generate plan

To compute the NDST using **trafo()** or **trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.fhat = np.array([1.1, 2.2, 3.3]) # Fourier coefficients (numpy.prod(N - 1) entries)
    plan.trafo()
    # or
    plan.trafo_direct()

To compute the transposed NDST using **nfst_transposed()** or **nfst_transposed_direct()**, both **x** and **f** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.f = np.array([1.0, 1.1, 1.2, 1.3, 1.4]) # values for adjoint NFFT (M entries)
    plan.nfst_transposed()
    # or
    plan.nfst_transposed_direct()

Using fastsum
--------------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/fastsum_test.py>`_
for a detailed example of all function uses and error tests.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/fastsum.rst>`_
for the class methods and attributes.

To start using fastsum, first import the class:

.. code-block:: python

    from pyNFFT3 import fastsum

To generate a fastsum plan, you must define **d**, **N**, **M**, **kernel**, and **c**.

The possible kernel types are:
    - gaussian
    - multiquadric
    - inverse_multiquadric
    - logarithm
    - thinplate_spline
    - one_over_square
    - one_over_modulus
    - one_over_x
    - inverse_multiquadric3
    - sinc_kernel
    - cosc
    - cot
    - one_over_cube
    - log_sin
    - laplacian_rbf

The given **c** will be converted to an array with length depending on the chosen kernel:

.. code-block:: python

    d = 2 # 2 dimensions
    N = 3 # 3 source nodes
    M = 5 # 5 target nodes
    kernel = "multiquadric"
    c = 1 / numpy.sqrt(N) # set kernel parameter
    plan = FASTSUM(N, M) # generate plan

First, the values for **x**, **y**, and **alpha** must be set.
Note that the values in **x** and **y** must satisfy:

    .. math::
        \|\pmb{x}_k\| \leq 0.5 \left(0.5 - \epsilon_B\right)

        \|\pmb{y}_k\| \leq 0.5 \left(0.5 - \epsilon_B\right)

.. code-block:: python

    plan.x = np.array([[0.1, 0.15], [-0.1, 0.15], [0.05, 0.09]]) # source nodes (N entries)
    plan.y = np.array([[0.07, 0.08], [-0.013, 0.021], [0.11, 0.16], [0.12, -0.08], [0.10, -0.11]]) # target nodes (M entries)
    plan.alpha = np.array([1.0+1.0j, 1.1-1.1j, 1.2+1.2j]) # source coefficients (N entries)

You can then compute the fast NFFT-based summation using **fastsum_trafo()** or the direct computation of sums using **fastsum_trafo_exact()**:

.. code-block:: python

    plan.fastsum_trafo()
    # or
    plan.fastsum_trafo_exact()