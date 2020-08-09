"""
    Test script for mesh.py (Mesh class)
"""

from mesh import *


eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31


def V(x):
    return 0


print("Testing the Mesh class' features\n")
print("Infinite well\n")
L = nm
E_1 = hbar**2 * np.pi**2 / (2*m*L**2)
infiniteWellMesh = Mesh(boundaries=[0,  L], nb_element=500, potential=V)
eigvals = infiniteWellMesh.solve()
# for i, E in enumerate(eigvals, start=1):
#     print("%d : %2.3f eV = %2.3f E_1. Ratio par rapport à la valeur"
#               "théorique : %2.5f" % (i, E/eV, E/E_1, E/E_1/i**2))

print("Fixed number of elements works")

infiniteWellMesh = Mesh(boundaries=[0,  L], element_size=L/500.5, potential=V)
eigvals = infiniteWellMesh.solve()
# for i, E in enumerate(eigvals, start=1):
#     print("%d : %2.3f eV = %2.3f E_1. Ratio par rapport à la valeur"
#               "théorique : %2.5f" % (i, E/eV, E/E_1, E/E_1/i**2))

print("Fixed element size works")
