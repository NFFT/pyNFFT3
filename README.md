# pyNFFT3

Python interface to the [NFFT C library](https://github.com/NFFT/nfft), based on the existing [Julia interface](https://nfft.github.io/NFFT3.jl). **The compiled NFFT (and FFTW) libraries are bundled in the pip wheel** — there is no separate C library to install, no FFTW to provide, and no compiler required.

[![](https://github.com/NFFT/pyNFFT3/actions/workflows/ci.yml/badge.svg)](https://github.com/NFFT/pyNFFT3/actions/workflows/ci.yml)

`pyNFFT3` provides the following fast algorithms and includes test scripts and dependencies for each:
- nonequispaced fast Fourier transform (NFFT) 
- nonequispaced fast cosine transform (NFCT) 
- nonequispaced fast sine transform (NFST)
- nonequispaced fast spherical Fourier transforms (NFSFT)
- fast spherical Fourier transforms (FSFT)
- fast summation (fastsum) 

## Getting started

The [pyNFFT3 package](https://pypi.org/project/pyNFFT3/) can be installed via pip:

```
pip install pyNFFT3
```

That is all: the prebuilt NFFT/FFTW shared libraries ship inside the wheel and are loaded
via `ctypes` at runtime, so nothing else needs to be built or installed (unlike the older
[`pyNFFT`](https://pypi.org/project/pyNFFT/), which required a system-installed NFFT + FFTW).
This is also why the wheel is comparatively large (~100 MB).

Read the [documentation](https://nfft.github.io/pyNFFT3/) for specific usage information.

Requirements
------------

No compiler and no system-installed NFFT or FFTW are needed — the wheel provides the
compiled libraries. The only Python-side requirements are:

- Python 3.8 or greater
- Numpy 2.0.0 or greater
- py-cpuinfo 9.0.0 or greater
- packaging 24.1 or greater

Note: `pip install pyNFFT3` will pull in NumPy 2.0+, which may upgrade NumPy in an existing
environment. The bundled binaries are platform-specific; if `import pyNFFT3` fails to load
its shared library on your platform, please open an issue with your OS/architecture.
