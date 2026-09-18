# pip install pyNFFT3

# Einfaches Beispiel fuer die nichtaequidistante schnelle Fourier-Transformation (NFFT)
#
# Die NFFT wertet ein trigonometrisches Polynom
#     f(x) = sum_{k=-N/2}^{N/2-1} fhat_k * exp(-2*pi*i*k*x)
# an beliebigen (nicht notwendig aequidistanten) Knoten x in [-0.5, 0.5) aus.
# Fuer aequidistante Knoten waere das die gewoehnliche (schnelle) Fourier-Transformation;
# die NFFT erlaubt dieselbe Geschwindigkeit auch fuer beliebig verteilte Knoten.

import matplotlib.pyplot as plt
import numpy as np

from pyNFFT3 import NFFT

rng = np.random.default_rng(1)

##################################
## Definition der Parameter     ##
##################################

N = np.array([32], dtype="int32")  # Bandbreite (Anzahl der Fourier-Koeffizienten)
Ns = N[0]
M = 40  # Anzahl der nicht-aequidistanten Auswertungsknoten

# Ein trigonometrisches Polynom mit nur drei aktiven Frequenzen -> leicht nachvollziehbar
freqs = {-8: 1.0 + 0.0j, 3: 0.5 - 0.2j, 10: 0.3 + 0.3j}
fhat = np.zeros(Ns, dtype=np.complex128)
for k, val in freqs.items():
    fhat[k + Ns // 2] = val  # Index l <-> Frequenz k = l - N/2

############################
## Erzeugung der Knoten   ##
############################

x = np.ascontiguousarray(rng.random(M) - 0.5)  # zufaellig verteilte Knoten in [-0.5, 0.5)

##########################
## NFFT auswerten       ##
##########################

plan = NFFT(N, M)
plan.x = x
plan.fhat = fhat.copy()

plan.trafo()  # schnelle NFFT
f_fast = plan.f.copy()

plan.trafo_direct()  # langsame, direkte Auswertung (NDFT) zum Vergleich
f_direct = plan.f.copy()

print(f"max |f_schnell - f_direkt| = {np.max(np.abs(f_fast - f_direct)):.2e}")
print("-> die schnelle NFFT liefert (bis auf Rundungsfehler) dasselbe wie die direkte Summe")

################################################
## Visualisierung: Polynom auf feinem Gitter  ##
################################################

x_grid = np.linspace(-0.5, 0.5, 2000, endpoint=False)
f_grid = np.zeros_like(x_grid, dtype=np.complex128)
for k, val in freqs.items():
    f_grid += val * np.exp(-2j * np.pi * k * x_grid)

plt.figure()
plt.plot(x_grid, f_grid.real, label="trigonometrisches Polynom (Realteil)")
plt.scatter(x, f_fast.real, color="C1", zorder=5, label="NFFT an Streupunkten")
plt.xlabel("x")
plt.ylabel("Re f(x)")
plt.title("NFFT: Auswertung eines trigonometrischen Polynoms an Streupunkten")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

###############################################
## adjungierte NFFT: nur die Transponierte!  ##
###############################################

# plan.adjoint() wendet die adjungierte (transponierte, konjugiert-komplexe) Abbildung an.
# Das ist NICHT die Umkehrabbildung der NFFT - fuer eine echte Rekonstruktion der
# Koeffizienten aus Messwerten siehe example_inverse_nfft.py (iterativer Loeser).
plan.f = f_fast.copy()
plan.adjoint()
print("adjungierte NFFT (erste 5 Werte):", np.round(plan.fhat[:5], 3))
