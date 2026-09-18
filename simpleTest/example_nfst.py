# pip install pyNFFT3

# Einfaches Beispiel fuer die nichtaequidistante schnelle Sinus-Transformation (NFST)
#
# Die NFST wertet ein reines Sinus-Polynom
#     f(x) = sum_{k=1}^{N-1} fhat_{k-1} * sin(2*pi*k*x)
# an beliebig verteilten Knoten x aus. Sie eignet sich fuer reellwertige, ungerade
# Funktionen (f(-x) = -f(x)), analog zur NFCT fuer gerade Funktionen.

import matplotlib.pyplot as plt
import numpy as np

from pyNFFT3 import NFST

rng = np.random.default_rng(3)

##################################
## Definition der Parameter     ##
##################################

N = np.array([16], dtype="int32")  # Bandbreite
Ns = N[0]
M = 30  # Anzahl der nicht-aequidistanten Auswertungsknoten

# Ein Sinus-Polynom mit nur zwei aktiven Frequenzen -> leicht nachvollziehbar
# fhat hat Ns - 1 Eintraege (k = 1, ..., N-1); k = 0 traegt bei sin() nichts bei
freqs = {2: 1.0, 6: 0.5}
fhat = np.zeros(Ns - 1, dtype=np.float64)
for k, val in freqs.items():
    fhat[k - 1] = val

############################
## Erzeugung der Knoten   ##
############################

x = np.ascontiguousarray(rng.random(M))  # zufaellig verteilte Knoten in [0, 1)

##########################
## NFST auswerten       ##
##########################

plan = NFST(N, M)
plan.x = x
plan.fhat = fhat.copy()

plan.trafo()  # schnelle NFST
f_fast = plan.f.copy()

plan.trafo_direct()  # langsame, direkte Auswertung zum Vergleich
f_direct = plan.f.copy()

print(f"max |f_schnell - f_direkt| = {np.max(np.abs(f_fast - f_direct)):.2e}")

################################################
## Visualisierung: Polynom auf feinem Gitter  ##
################################################

x_grid = np.linspace(0, 1, 2000, endpoint=False)
f_grid = np.zeros_like(x_grid)
for k, val in freqs.items():
    f_grid += val * np.sin(2 * np.pi * k * x_grid)

plt.figure()
plt.plot(x_grid, f_grid, label="Sinus-Polynom")
plt.scatter(x, f_fast, color="C1", zorder=5, label="NFST an Streupunkten")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("NFST: Auswertung eines reellen, ungeraden Signals an Streupunkten")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

################################################
## transponierte NFST: nur die Transponierte! ##
################################################

# plan.nfst_transposed() ist die Transponierte, keine Umkehrabbildung.
plan.f = f_fast.copy()
plan.nfst_transposed()
print("transponierte NFST (erste 5 Werte):", np.round(plan.fhat[:5], 3))
