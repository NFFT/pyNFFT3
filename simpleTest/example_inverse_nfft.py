# pip install pyNFFT3

# Inverse NFFT (iNFFT) mittels iterativem Loeser
#
# NFFT.trafo() berechnet aus Fourier-Koeffizienten fhat Funktionswerte f an
# Streupunkten x. plan.adjoint() ist NUR die Transponierte dieser Abbildung -
# keine Umkehrabbildung (siehe example_nfft.py)! Um aus gegebenen (evtl.
# verrauschten) Messwerten f an den Knoten x die Koeffizienten fhat zu
# rekonstruieren, muss man das lineare Ausgleichsproblem
#     min_fhat  ||F * fhat - f||^2 + lambda * ||fhat||^2
# loesen, wobei F die (dichte, MxN) NFFT-Matrix ist. F wird nie explizit
# aufgebaut - stattdessen nutzt ein iterativer Loeser (hier: CG fuer die
# Normalengleichung F^H F fhat = F^H f) ausschliesslich die schnellen
# trafo()/adjoint()-Auswertungen von NFFT als Matrix-Vektor-Produkte.
# Das ist z.B. die Grundlage der Regression in pyANOVAapprox (dort mit lsqr).

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from pyNFFT3 import NFFT

rng = np.random.default_rng(0)

##################################
## Definition der Parameter     ##
##################################

N = np.array([64], dtype="int32")  # Bandbreite
Ns = N[0]
M = 400  # Anzahl der Messknoten (M > Ns: ueberbestimmtes System)
lam = 1e-6  # kleine Tikhonov-Regularisierung fuer numerische Stabilitaet

##############################################################
## "wahre" Koeffizienten, die spaeter rekonstruiert werden  ##
##############################################################

freqs_true = {-20: 1.0 + 0.0j, 0: 0.6 + 0.3j, 15: 0.4 - 0.4j}
fhat_true = np.zeros(Ns, dtype=np.complex128)
for k, val in freqs_true.items():
    fhat_true[k + Ns // 2] = val

###########################################################
## Messdaten erzeugen: NFFT auswerten und Rauschen zugeben ##
###########################################################

x = np.ascontiguousarray(rng.random(M) - 0.5)

plan = NFFT(N, M)
plan.x = x
plan.fhat = fhat_true.copy()
plan.trafo()
f_meas = plan.f.copy()
f_meas += 0.01 * (rng.standard_normal(M) + 1j * rng.standard_normal(M))  # Messrauschen

#####################################################################
## F und F^H als schnelle, matrixfreie lineare Abbildungen (NFFT!) ##
#####################################################################


def matvec(fhat_vec):
    plan.fhat = np.ascontiguousarray(fhat_vec, dtype=np.complex128).ravel()
    plan.trafo()
    return plan.f.copy()


def rmatvec(f_vec):
    plan.f = np.ascontiguousarray(f_vec, dtype=np.complex128).ravel()
    plan.adjoint()
    return plan.fhat.copy()


def normal_matvec(v):
    return rmatvec(matvec(v)) + lam * v  # (F^H F + lambda I) v


AhA = LinearOperator((Ns, Ns), matvec=normal_matvec, dtype=np.complex128)
rhs = rmatvec(f_meas)  # rechte Seite F^H f

##########################################################
## Inverse Transformation: CG auf der Normalengleichung ##
##########################################################

fhat_rec, info = cg(AhA, rhs, rtol=1e-8, maxiter=200)
print("CG info (0 = konvergiert):", info)

err = np.linalg.norm(fhat_rec - fhat_true) / np.linalg.norm(fhat_true)
print(f"relativer Fehler der rekonstruierten Koeffizienten: {err:.2e}")

################################################
## Visualisierung: wahre vs. rekonstruierte   ##
## Koeffizienten                              ##
################################################

k_axis = np.arange(-Ns // 2, Ns // 2)
plt.figure()
markerline1, _, _ = plt.stem(
    k_axis, np.abs(fhat_true), linefmt="C0-", markerfmt="C0o", basefmt=" "
)
markerline1.set_label("wahre Koeffizienten")
markerline2, _, _ = plt.stem(
    k_axis, np.abs(fhat_rec), linefmt="C1--", markerfmt="C1x", basefmt=" "
)
markerline2.set_label("rekonstruiert (CG + NFFT)")
plt.xlabel("Frequenz k")
plt.ylabel("|fhat_k|")
plt.title("Inverse NFFT: Rekonstruktion aus verrauschten Streupunkt-Messungen")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
