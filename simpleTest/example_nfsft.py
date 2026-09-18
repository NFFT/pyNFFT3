# pip install pyNFFT3

# Einfaches Beispiel fuer die nichtaequidistante schnelle Kugelflaechen-
# funktionen-Transformation (NFSFT)
#
# Die NFSFT wertet eine Funktion auf der Einheitssphaere, die als Reihe von
# Kugelflaechenfunktionen Y_k^n gegeben ist,
#     f(x) = sum_{k=0}^{N} sum_{n=-k}^{k} fhat_{k,n} * Y_k^n(x)
# an beliebig verteilten Punkten x auf der Sphaere aus. Ein Knoten x = (x0, x1)
# codiert dabei die Kugelkoordinaten ueber x0 = phi/(2*pi) in [-0.5, 0.5)
# (Azimut) und x1 = theta/pi in [0, 1) (Polarwinkel).

import matplotlib.pyplot as plt
import numpy as np

from pyNFFT3.NFSFT import *

rng = np.random.default_rng(4)

##################################
## Definition der Parameter     ##
##################################

N = 8  # Bandbreite (maximaler Grad k der Kugelflaechenfunktionen)
M = 400  # Anzahl der Auswertungspunkte auf der Sphaere

############################
## Erzeugung der Knoten   ##
############################

X = np.zeros((2, M))
X[0, :] = rng.random(M) - 0.5  # Azimut phi/(2*pi) in [-0.5, 0.5)
X[1, :] = rng.random(M)  # Polarwinkel theta/pi in [0, 1)

plan = NFSFT(N, M)
plan.x = X

################################################################
## Nur eine einzige Kugelflaechenfunktion Y_2^1 ist aktiv     ##
################################################################

fhat = np.zeros((2 * plan.N + 2) ** 2, dtype=np.complex128)
index = plan.nfsft_index(2, 1)  # Index fuer (k, n) = (2, 1)
fhat[index] = 1.0 + 0.0j
plan.fhat = fhat

##########################
## NFSFT auswerten      ##
##########################

# Hinweis: trafo_direct() muss VOR trafo() aufgerufen werden, sonst liefert
# trafo_direct() auf demselben Plan falsche Werte (siehe NFSFT_test.py).
plan.trafo_direct()  # langsame, direkte Auswertung (NDSFT) zum Vergleich
f_direct = plan.f.copy()

plan.trafo()  # schnelle NFSFT
f_fast = plan.f.copy()

print(f"max |f_schnell - f_direkt| = {np.max(np.abs(f_fast - f_direct)):.2e}")

################################################
## Visualisierung: Funktionswerte auf der Sphaere ##
################################################

phi = 2 * np.pi * X[0]
theta = np.pi * X[1]
xs = np.sin(theta) * np.cos(phi)
ys = np.sin(theta) * np.sin(phi)
zs = np.cos(theta)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
p = ax.scatter(xs, ys, zs, c=f_fast.real, cmap="coolwarm")
fig.colorbar(p, label="Re f(x)", shrink=0.7)
ax.set_title("NFSFT: Kugelflaechenfunktion Y_2^1 auf Streupunkten der Sphaere")
ax.set_box_aspect((1, 1, 1))
plt.show()

###############################################
## adjungierte NFSFT: nur die Transponierte! ##
###############################################

plan.f = f_fast.copy()
plan.adjoint()
print("adjungierte NFSFT, Koeffizient bei (k,n)=(2,1):", np.round(plan.fhat[index], 3))
