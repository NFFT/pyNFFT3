import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFFT import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([16, 8, 4], dtype='int32') # 3d

M = 100
d = len(N)
Ns = np.prod(N)

X = np.array([[abs(np.sin(i + j)) for j in range(d)] for i in range(M)])
fhat = np.array([np.cos(k) + 1.0j * np.sin(k) for k in range(Ns)])
f = np.array([np.sin(m) + 1.0j * np.cos(m) for m in range(M)])

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

# test transpose
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

# for testing transpose
f1 = F_mat.H @ f
norm_euclidean_adj = np.linalg.norm(f1 - plan_adj.fhat)
norm_infinity_adj = np.linalg.norm(f1 - plan_adj.fhat, np.inf)
print("Euclidean norm for transpose test:", norm_euclidean_adj)
print("Infinity norm for transpose test:", norm_infinity_adj)

#TODO: Add tests to check for large error values