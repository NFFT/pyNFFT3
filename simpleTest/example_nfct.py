# pip install pyNFFT3

# Einfaches Beispiel fuer die nichtaequidistante schnelle Kosinus-Transformation (NFCT)
#
# Die NFCT wertet ein reines Kosinus-Polynom
#     f(x) = sum_{k=0}^{N-1} fhat_k * cos(2*pi*k*x)
# an beliebig verteilten Knoten x aus. Sie eignet sich fuer reellwertige, gerade
# Funktionen (f(-x) = f(x)) - man kommt dabei komplett ohne komplexe Zahlen aus,
# im Gegensatz zur allgemeinen NFFT (siehe example_nfft.py).

import matplotlib.pyplot as plt
import numpy as np

from pyNFFT3 import NFCT

rng = np.random.default_rng(2)

##################################
## Definition der Parameter     ##
##################################

N = np.array([16], dtype="int32")  # Bandbreite (Anzahl der Kosinus-Koeffizienten)
Ns = N[0]
M = 30  # Anzahl der nicht-aequidistanten Auswertungsknoten

# Ein Kosinus-Polynom mit nur zwei aktiven Frequenzen -> leicht nachvollziehbar
freqs = {0: 0.5, 3: 1.0, 7: 0.4}
fhat = np.zeros(Ns, dtype=np.float64)
for k, val in freqs.items():
    fhat[k] = val

############################
## Erzeugung der Knoten   ##
############################

x = np.ascontiguousarray(rng.random(M))  # zufaellig verteilte Knoten in [0, 1)

##########################
## NFCT auswerten       ##
##########################

plan = NFCT(N, M)
plan.x = x
plan.fhat = fhat.copy()

plan.trafo()  # schnelle NFCT
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
    f_grid += val * np.cos(2 * np.pi * k * x_grid)

plt.figure()
plt.plot(x_grid, f_grid, label="Kosinus-Polynom")
plt.scatter(x, f_fast, color="C1", zorder=5, label="NFCT an Streupunkten")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("NFCT: Auswertung eines reellen, geraden Signals an Streupunkten")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

################################################
## transponierte NFCT: nur die Transponierte! ##
################################################

# plan.nfct_transposed() ist die Transponierte, keine Umkehrabbildung.
plan.f = f_fast.copy()
plan.nfct_transposed()
print("transponierte NFCT (erste 5 Werte):", np.round(plan.fhat[:5], 3))
