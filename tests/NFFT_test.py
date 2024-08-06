import numpy as np
from src.pyNFFT3.flags import *
from src.pyNFFT3.NFFT import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([16, 8, 4], dtype='int32') # 3d

M = 100
d = len(N)
Ns = np.prod(N)

X = (np.random.rand(3, M) - 0.5).astype(np.float64).T
fhat = np.random.rand(Ns) + 1.0j*np.random.rand(Ns)

# test init and setting
plan = NFFT(N,M)
plan.x = X
plan.fhat = fhat

# test trafo
plan.trafo() # value is in plan.f
f2 = plan.f

# compare with directly computed
if d == 1:
    I = [[k] for  k in range(int(-N[0]/2),int(N[0]/2))]
elif d == 2:
    I = [[k, i] for  k in range(int(-N[0]/2),int(N[0]/2)) for i in range(int(-N[1]/2),int(N[1]/2))]
elif d == 3:
    I = [[k, i, j] for  k in range(int(-N[0]/2),int(N[0]/2)) for i in range(int(-N[1]/2),int(N[1]/2)) for j in range(int(-N[2]/2),int(N[2]/2))]

F = np.array([[np.exp(-2 * np.pi * 1j * np.dot(X.T[:,j],I[l])) for l in range (0,Ns) ] for j in range(0,M)])
F_mat = np.asmatrix(F)

# get errors from trafo test
f1 = F @ fhat
error_vector = f1 - f2
norm_euclidean_traf = np.linalg.norm(error_vector) / np.linalg.norm(f1)
norm_infinity_traf = np.linalg.norm(error_vector, np.inf) / np.linalg.norm(fhat, 1)
print("Euclidean norm for trafo test:", norm_euclidean_traf)
print("Infinity norm: for trafo test", norm_infinity_traf)
assert norm_euclidean_traf < 1e-10, f"TEST FAILED: Euclidiean norm ({norm_euclidean_traf}) for trafo test is not less than 1e-10"
assert norm_infinity_traf < 1e-10, f"TEST FAILED: Infinity norm ({norm_infinity_traf}) for trafo test is not less than 1e-10"

# test transpose
plan.adjoint()
f2 = plan.fhat
f1 = F_mat.H @ plan.f

# get errors from transpose test
error_vector = f1 - f2
norm_euclidean_adj = np.linalg.norm(f1 - plan.fhat)
norm_infinity_adj = np.linalg.norm(f1 - plan.fhat, np.inf)
print("Euclidean norm for transpose test:", norm_euclidean_adj)
print("Infinity norm for transpose test:", norm_infinity_adj)
assert norm_euclidean_adj < 1e-10, f"TEST FAILED: Euclidiean norm ({norm_euclidean_adj}) for transpose test is not less than 1e-10"
assert norm_infinity_adj < 1e-10, f"TEST FAILED: Infinity norm ({norm_infinity_adj}) for transpose test is not less than 1e-10"