# pyNFFT3

Python interface for the [NFFT C library](https://github.com/NFFT/nfft). Based on the existing [Julia interface](https://nfft.github.io/NFFT3.jl).

`pyNFFT3` provides the following fast algorithms and includes test scripts and dependencies for each:
- nonequispaced fast Fourier transform (NFFT) 
- nonequispaced fast cosine transform (NFCT) 
- nonequispaced fast sine transform (NFST)
- fast summation (fastsum) 

## Getting started

The pyNFFT3 package can be installed via pip:

```
pip install pynfft
```

Read the [documentation](https://nfft.github.io/NFFT3.jl/stable/) for specific usage information.

## Using NFFT
### Fields:
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
- **init_done**: boolean to indicate if the NFFT plan is initalized.
- **finalized:** boolean to indicate if the NFFT plan is finalized.

### Methods:
- **__init__**: constructor for the NFFT plan.
- **__del__**: destructor for the NFFT plan.
- **finalize_plan**: finalizes an NFFT plan. This function does not have to be called by the user.
- **init**: intialises the NFFT plan in C. This function does not have to be called by the user.
- **trafo**: computes the NDFT using the fast NFFT algorithm.
- **trafo_direct**: computes the NDFT via naive matrix-vector multiplication.
- **adjoint**: computes the adjoint NDFT using the fast adjoint NFFT algorithm.
- **adjoint_direct**: computes the adjoint NDFT using naive matrix-vector multiplication.

View the [test file](LINK HERE!!) for a detailed example of all function uses.

To start using NFFT, first import the class:

``` 
from pynfft3 import NFFT 
```

You can then generate a plan with your desired values (which will be checked for proper type/size):

```
N = np.array([16, 8], dtype='int32') # 2 dimensions
M = 100
plan = NFFT(N,M)
```

To compute the NDFT using **trafo** or **trafo_direct**, both **x** and **fhat** must be set:
```
X = np.array([[abs(np.sin(i + j)) for j in range(d)] for i in range(M)])
fhat = np.array([np.cos(k) + 1.0j * np.sin(k) for k in range(Ns)])
plan.trafo()
```

Requirements
------------

- Python 3.10 or greater
- Numpy 2.0.0 or greater
