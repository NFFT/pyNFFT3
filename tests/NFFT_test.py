import numpy as np
from src.pyNFFT3.flags import *
from src.pyNFFT3.NFFT import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([4, 2], dtype='int32') # 3d

M = 5
d = len(N)
Ns = np.prod(N)

X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
fhat = np.array([0.1+0.1j, 0.2-0.2j, 0.3+0.3j, 0.4-0.4j, 0.5+0.5j, 0.6-0.6j, 0.7+0.7j, 0.8-0.8j])
f = np.array([1.0+1.0j, 1.1-1.1j, 1.2+1.2j, 1.3-1.3j, 1.4+1.4j])

print(X)
print(fhat)
print(f)

# test init and setting
plan_traf = NFFT(N,M)
plan_traf.x = X
plan_traf.f = f # this gets overwritten
plan_traf.fhat = fhat

plan_adj = NFFT(N,M)
plan_adj.x = X
plan_adj.f = f # this gets overwritten
plan_adj.fhat = fhat

# test trafo
plan_traf.trafo() # value is in plan.f

# test adjoint
plan_adj.adjoint()

# compare with directly computed
if d == 1:
    I = [[k] for  k in range(int(-N[0]/2),int(N[0]/2))]
elif d == 2:
    I = [[k, i] for  k in range(int(-N[0]/2),int(N[0]/2)) for i in range(int(-N[1]/2),int(N[1]/2))]
elif d == 3:
    I = [[k, i, j] for  k in range(int(-N[0]/2),int(N[0]/2)) for i in range(int(-N[1]/2),int(N[1]/2)) for j in range(int(-N[2]/2),int(N[2]/2))]

F = np.array([[np.exp(-2 * np.pi * 1j * np.dot(X.T[:,j],I[l])) for l in range (0,Ns) ] for j in range(0,M)])
F_mat = np.asmatrix(F)

## norm values should be ~e-12
# for testing trafo
f1 = F @ fhat
norm_euclidean_traf = np.linalg.norm(f1 - plan_traf.f)
norm_infinity_traf = np.linalg.norm(f1 - plan_traf.f, np.inf)
print("Euclidean norm for trafo test:", norm_euclidean_traf)
print("Infinity norm: for trafo test", norm_infinity_traf)

# for testing adjoint
f1 = F_mat.H @ f
norm_euclidean_adj = np.linalg.norm(f1 - plan_adj.fhat)
norm_infinity_adj = np.linalg.norm(f1 - plan_adj.fhat, np.inf)
print("Euclidean norm for adjoint test:", norm_euclidean_adj)
print("Infinity norm for adjoint test:", norm_infinity_adj)

#TODO: Add tests to check for large error values