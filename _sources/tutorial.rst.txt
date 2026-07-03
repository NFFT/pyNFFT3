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

    import numpy as np
    N = np.array([4, 2], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5 # 5 nodes
    plan = NFFT(N, M) # generate plan

To compute the NDFT using **nfft_trafo()** or **nfft_trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.fhat = np.array([0.1+0.1j, 0.2-0.2j, 0.3+0.3j, 0.4-0.4j, 0.5+0.5j, 0.6-0.6j, 0.7+0.7j, 0.8-0.8j]) # Fourier coefficients (numpy.prod(N) entries)
    plan.nfft_trafo()
    # or
    plan.nfft_trafo_direct()

To compute the adjoint NDFT using **nfft_adjoint()** or **nfft_adjoint_direct()**, both **x** and **f** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.f = np.array([1.0+1.0j, 1.1-1.1j, 1.2+1.2j, 1.3-1.3j, 1.4+1.4j]) # values for adjoint NFFT (M entries)
    plan.nfft_adjoint()
    # or
    plan.nfft_adjoint_direct()

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

    import numpy as np
    N = np.array([4, 2], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5 # 5 nodes
    plan = NFCT(N, M) # generate plan

To compute the NDCT using **nfct_trafo()** or **nfct_trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.fhat = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]) # Fourier coefficients (numpy.prod(N) entries)
    plan.nfct_trafo()
    # or
    plan.nfct_trafo_direct()

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

    import numpy as np
    N = np.array([4, 2], dtype='int32')  # 2 dimensions, ensure proper type
    M = 5 # 5 nodes
    plan = NFST(N, M) # generate plan

To compute the NDST using **nfst_trafo()** or **nfst_trafo_direct()**, both **x** and **fhat** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.fhat = np.array([1.1, 2.2, 3.3]) # Fourier coefficients (numpy.prod(N - 1) entries)
    plan.nfst_trafo()
    # or
    plan.nfst_trafo_direct()

To compute the transposed NDST using **nfst_transposed()** or **nfst_transposed_direct()**, both **x** and **f** must be set:

.. code-block:: python

    plan.x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]) # sampling nodes (M entries)
    plan.f = np.array([1.0, 1.1, 1.2, 1.3, 1.4]) # values for adjoint NFFT (M entries)
    plan.nfst_transposed()
    # or
    plan.nfst_transposed_direct()

Using NFSFT
------------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/NFSFT_test.py>`_
for a detailed example of all function uses and error tests.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/NFSFT.rst>`_
for the class methods and attributes.

To start using NFSFT, first import the class:

.. code-block:: python

    from pyNFFT3 import NFSFT

You can then generate a plan with your desired bandwidth and number of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = 6  # bandwidth
    M = 8 # 8 nodes
    plan = NFSFT(N, M) # generate plan

To compute the NFSFT using **nfsft_trafo()** or the NDFST using **nfsft_trafo_direct()**, both **x** and **fhat** must be set.
**fhat** must be created in a format using **nfsft_index()** to properly space the entries of **fhat**:

.. code-block:: python

    import numpy as np
    plan.x = np.array([
        [0.48, 0.45, -0.18, 0.03, -0.08, 0.49, -0.32, 0.13],
        [-0.37, 0.12, -0.13, 0.21, -0.15, 0.09, 0.4, 0.43]]) # sampling nodes (2xM entries )

    fhat = np.zeros((2*plan.N+2)**2, dtype=np.complex128) # set up structure for fhat ((2*N+2)*(2*N+2) entries)
    for k in range(N + 1):
        for n in range(-k, k + 1):
            index = plan.nfsft_index(k, n)
            fhat[index] = (np.random.rand() - 0.5) + 1j * (np.random.rand() - 0.5) # Fourier coefficients, spaced using nfsft_index()

    plan.nfsft_trafo()
    # or
    plan.nfsft_trafo_direct()

To compute the adjoint NFSFT using **nfsft_adjoint()** or **nfsft_adjoint_direct()**, both **x** and **f** must be set.
**nfsft_trafo()** and **nfsft_trafo_direct()** produce values for **f**, so it is possible to use these outputs:

.. code-block:: python

    plan.x = np.array([
        [0.48, 0.45, -0.18, 0.03, -0.08, 0.49, -0.32, 0.13],
        [-0.37, 0.12, -0.13, 0.21, -0.15, 0.09, 0.4, 0.43]]) # sampling nodes (2xM entries )

    plan.nfsft_trafo() # assuming fhat is set, this will populate plan.f
    # or
    plan.f = np.array([0.1+0.1j, 0.2-0.2j, 0.3+0.3j, 0.4-0.4j, 0.5+0.5j, 0.6-0.6j, 0.7+0.7j, 0.8-0.8j]) # values for adjoint NFSFT (M entries)

    plan.nfsft_transposed()
    # or
    plan.nfsft_transposed_direct()

Using FSFT
------------

View the `test file <https://github.com/NFFT/pyNFFT3/blob/main/tests/FSFT_test.py>`_
for a detailed example of all function uses and error tests.

Or view the `API reference <https://github.com/NFFT/pyNFFT3/blob/main/docs/source/api/FSFT.rst>`_
for the class methods and attributes.

To start using FSFT, first import the class:

.. code-block:: python

    from pyNFFT3 import FSFT

You can then generate a plan with your desired bandwidth and number of nodes (which will be checked for proper type/size):

.. code-block:: python

    N = 6  # bandwidth
    M = 8 # 8 nodes
    plan = FSFT(N, M) # generate plan

To compute the FSFT using **fsft_trafo()** or the DFST using **fsft_trafo_direct()**, just **fhat** must be set.
**fhat** must be created in a format using **fsft_index()** to properly space the entries of **fhat**:

.. code-block:: python

    import numpy as np

    fhat = np.zeros((2*plan.N+2)**2, dtype=np.complex128) # set up structure for fhat ((2*N+2)*(2*N+2) entries)
    for k in range(N + 1):
        for n in range(-k, k + 1):
            index = plan.fsft_index(k, n)
            fhat[index] = (np.random.rand() - 0.5) + 1j * (np.random.rand() - 0.5) # Fourier coefficients, spaced using fsft_index()

    plan.fsft_trafo()
    # or
    plan.fsft_trafo_direct()

To compute the adjoint FSFT using **fsft_adjoint()** or **fsft_adjoint_direct()**, just **f** must be set.
**fsft_trafo()** and **fsft_trafo_direct()** produce values for **f**, so it is possible to use these outputs:

.. code-block:: python

    plan.fsft_trafo() # assuming fhat is set, this will populate plan.f
    # or
    plan.f = np.array([0.1+0.1j, 0.2-0.2j, 0.3+0.3j, 0.4-0.4j, 0.5+0.5j, 0.6-0.6j, 0.7+0.7j, 0.8-0.8j]) # values for adjoint FSFT (M entries)

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

- `gaussian`
  :math:`K(x) = \exp\left(-\frac{x^2}{c^2}\right)`
- `multiquadratic`
  :math:`K(x) = \sqrt{x^2 + c^2}`
- `inverse_multiquadratic`
  :math:`K(x) = \sqrt{\frac{1}{x^2 + c^2}}`
- `logarithm`
  :math:`K(x) = \log(\vert x \vert)`
- `thinplate_spline`
  :math:`K(x) = x^2 \log(\vert x \vert)`
- `one_over_square`
  :math:`K(x) = \frac{1}{x^2}`
- `one_over_modulus`
  :math:`K(x) = \frac{1}{\vert x \vert}`
- `one_over_x`
  :math:`K(x) = \frac{1}{x}`
- `one_over_multiquadric3`
  :math:`K(x) = \left(\frac{1}{x^2 + c^2}\right)^\frac{3}{2}`
- `sinc_kernel`
  :math:`K(x) = \frac{\sin(cx)}{x}`
- `cosc`
  :math:`K(x) = \frac{\cos(cx)}{x}`
- `kcot`
  :math:`K(x) = \cot(cx)`
- `one_over_cube`
  :math:`K(x) = \frac{1}{x^3}`
- `log_sin`
  :math:`K(x) = \log(\vert\sin(cx)\vert)`
- `laplacian_rbf`
  :math:`K(x) = \exp\left(-\frac{\vert x \vert}{c}\right)`
- `der_laplacian_rbf`
  :math:`K(x) = \frac{\vert x \vert}{c} \exp\left(-\frac{\vert x \vert}{c}\right)`
- `xx_gaussian`
  :math:`K(x) = \frac{x^2}{c^2} \exp\left(-\frac{x^2}{c^2}\right)`
- `absx`
  :math:`K(x) = \vert x \vert`

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

    import numpy as np
    plan.x = np.array([[0.1, 0.15], [-0.1, 0.15], [0.05, 0.09]]) # source nodes (N entries)
    plan.y = np.array([[0.07, 0.08], [-0.013, 0.021], [0.11, 0.16], [0.12, -0.08], [0.10, -0.11]]) # target nodes (M entries)
    plan.alpha = np.array([1.0+1.0j, 1.1-1.1j, 1.2+1.2j]) # source coefficients (N entries)

You can then compute the fast NFFT-based summation using **fastsum_trafo()** or the direct computation of sums using **fastsum_trafo_exact()**:

.. code-block:: python

    plan.fastsum_trafo()
    # or
    plan.fastsum_trafo_exact()