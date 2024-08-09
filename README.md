# pyNFFT3

Python interface for the [NFFT C library](https://github.com/NFFT/nfft). Based on the existing [Julia interface](https://nfft.github.io/NFFT3.jl).

[![](https://github.com/NFFT/pyNFFT3/actions/workflows/ci.yml/badge.svg)](https://github.com/NFFT/pyNFFT3/actions/workflows/ci.yml)

`pyNFFT3` provides the following fast algorithms and includes test scripts and dependencies for each:
- nonequispaced fast Fourier transform (NFFT) 
- nonequispaced fast cosine transform (NFCT) 
- nonequispaced fast sine transform (NFST)
- fast summation (fastsum) 

## Getting started

The [pyNFFT3 package](https://pypi.org/project/pyNFFT3/) can be installed via pip:

```
pip install pyNFFT3
```

Read the [documentation](https://nfft.github.io/pyNFFT3/) for specific usage information.

Requirements
------------

- Python 3.8 or greater
- Numpy 2.0.0 or greater
- cpufeature 0.2.1 or greater
