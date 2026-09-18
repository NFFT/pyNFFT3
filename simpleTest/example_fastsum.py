# pip install pyNFFT3

# Einfaches Beispiel fuer die schnelle Summation (fastsum)
#
# fastsum berechnet Summen der Form
#     f(y_m) = sum_{k=1}^{N} alpha_k * K(||y_m - x_k||)
# fuer viele Zielpunkte y_m gleichzeitig - z.B. die Ueberlagerung vieler
# Gauss-Kerne (=Quellpunkte x_k mit Gewicht alpha_k), ausgewertet an allen
# Zielpunkten. Die direkte Berechnung kostet O(N*M), fastsum schafft es in
# nahezu O(N + M).

import matplotlib.pyplot as plt
import numpy as np

from pyNFFT3.fastsum import *

rng = np.random.default_rng(6)

##################################
## Definition der Parameter     ##
##################################

d = 2  # Dimension
N = 15  # Anzahl der Quellpunkte (Kernzentren)
kernel = "gaussian"  # K(x) = exp(-x^2 / c^2)
c = 0.05  # Kernbreite
eps_B = 1 / 16  # aeusserer Sicherheitsabstand (siehe Dokumentation)
r_max = 0.5 * (0.5 - eps_B)  # max. erlaubter Radius fuer Knoten

############################################
## Quellpunkte mit zufaelligen Gewichten  ##
############################################

r = np.sqrt(rng.random(N)) * r_max
phi = rng.random(N) * 2 * np.pi
X = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
alpha = rng.uniform(0.5, 1.5, N).astype(np.complex128)  # positive "Massen"

#####################################################
## Zielpunkte: regelmaessiges Gitter in der Scheibe ##
#####################################################

gx, gy = np.meshgrid(np.linspace(-r_max, r_max, 40), np.linspace(-r_max, r_max, 40))
Y_all = np.column_stack((gx.ravel(), gy.ravel()))
mask = np.linalg.norm(Y_all, axis=1) <= r_max
Y = Y_all[mask]
M = Y.shape[0]

##########################
## fastsum auswerten    ##
##########################

plan = FASTSUM(d, N, M, kernel, c, eps_B=eps_B)
plan.x = X
plan.alpha = alpha
plan.y = Y

plan.fastsum_trafo()  # schnelle Summation
f_fast = plan.f.copy()

plan.fastsum_trafo_exact()  # direkte, langsame Summation zum Vergleich
f_exact = plan.f.copy()

print(f"max |f_schnell - f_direkt| = {np.max(np.abs(f_fast - f_exact)):.2e}")

################################################
## Visualisierung: ueberlagerte Gauss-Kerne   ##
################################################

plt.figure()
sc = plt.scatter(Y[:, 0], Y[:, 1], c=f_fast.real, cmap="viridis", s=25)
plt.colorbar(sc, label="f(y) (Summe der Gauss-Kerne)")
plt.scatter(X[:, 0], X[:, 1], c="red", marker="x", label="Quellpunkte x_k")
plt.gca().set_aspect("equal")
plt.xlabel("y_1")
plt.ylabel("y_2")
plt.title("fastsum: schnelle Auswertung ueberlagerter Gauss-Kerne")
plt.legend()
plt.show()
