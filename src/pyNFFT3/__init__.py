import ctypes
import os

# Create dummy classes for plans
class nfft_plan(ctypes.Structure):
    pass
class nfst_plan(ctypes.Structure):
    pass
class nfct_plan(ctypes.Structure):
    pass
# class nfmt_plan(ctypes.Structure):
#     pass
class fastsum_plan(ctypes.Structure):
    pass

# Determine the file extension for shared libraries based on the operating system
if os.name == 'nt':  # Windows
    ending = ".dll"
elif os.uname().sysname == 'Darwin':  # macOS
    ending = ".dylib"
else:  # Linux
    ending = ".so"

# Define the library paths
package_dir = os.path.dirname(__file__)
lib_path_nfft = os.path.join(package_dir, "libnfftjulia" + ending)
lib_path_nfct = os.path.join(package_dir, "libnfctjulia" + ending)
lib_path_nfst = os.path.join(package_dir, "libnfstjulia" + ending)
# lib_path_nfmt = os.path.join(package_dir, "libnfmtjulia" + ending)
lib_path_fastsum = os.path.join(package_dir, "libfastsumjulia" + ending)

# Load the libraries
nfftlib = ctypes.CDLL(lib_path_nfft)
nfctlib = ctypes.CDLL(lib_path_nfct)
nfstlib = ctypes.CDLL(lib_path_nfst)
# nfstlib = ctypes.CDLL(lib_path_nfst)
fastsumlib = ctypes.CDLL(lib_path_fastsum)

# Import modules
from .NFFT import *
from .NFCT import *
from .NFST import *
# from .NFMT import *
from .fastsum import *
from .flags import *

# Export functions and flags
__all__ = [
    'NFFT', 'NFCT', 'NFST', 'FASTSUM', 'NFMT',
    'nfft_finalize_plan', 'nfft_init', 'nfft_trafo', 'nfft_adjoint',
    'nfft_trafo_direct', 'nfft_adjoint_direct', 'nfct_finalize_plan',
    'nfct_init', 'nfct_trafo', 'nfct_adjoint', 'nfct_transposed',
    'nfct_trafo_direct', 'nfct_adjoint_direct', 'nfct_transposed_direct',
    'nfst_finalize_plan', 'nfst_init', 'nfst_trafo', 'nfst_adjoint',
    'nfst_trafo_direct', 'nfst_adjoint_direct', 'fastsum_finalize_plan',
    'fastsum_init', 'fastsum_trafo', 'fastsum_trafo_exact',
    'finalize_plan', 'nffct_finalize_plan', 'nffct_init', 'nffct_trafo',
    'nffct_adjoint', 'init', 'trafo', 'adjoint', 'trafo_direct',
    'adjoint_direct', 'trafo_exact',
    'PRE_PHI_HUT', 'FG_PSI', 'PRE_LIN_PSI', 'PRE_FG_PSI', 'PRE_PSI',
    'PRE_FULL_PSI', 'MALLOC_X', 'MALLOC_F_HAT', 'MALLOC_F',
    'FFT_OUT_OF_PLACE', 'FFTW_INIT', 'NFFT_SORT_NODES',
    'NFFT_OMP_BLOCKWISE_ADJOINT', 'NFCT_SORT_NODES',
    'NFCT_OMP_BLOCKWISE_ADJOINT', 'NFST_SORT_NODES',
    'NFST_OMP_BLOCKWISE_ADJOINT', 'PRE_ONE_PSI', 'FFTW_MEASURE',
    'FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED', 'FFTW_CONSERVE_MEMORY',
    'FFTW_EXHAUSTIVE', 'FFTW_PRESERVE_INPUT', 'FFTW_PATIENT',
    'FFTW_ESTIMATE', 'FFTW_WISDOM_ONLY', 'f1_default_1d', 'f1_default',
    'f2_default', 'default_window_cut_off'
]