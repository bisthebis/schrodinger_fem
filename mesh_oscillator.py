# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:05:07 2020

@author: boris
"""

#  import numpy as np
from element import LinearElement
from abstract_mesh import AbstractMesh
from scipy.linalg import eigh

eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31

# Potentiel
E0 = eV/2
w = 2*E0/hbar

x_caract = (hbar/(m*w))**0.5
p_caract = hbar/x_caract


class MeshHO(AbstractMesh):
    
    def __init__(self, nb_elems, precision):
        super().__init__()
        self.construct(nb_elems, precision)

    def construct(self, N, precision):
        """Precision = Element lenth in multiples of x_caract"""
        # nb_elem = 2*N  # assignment never used
        self.dofs = 2*N+1
        self.elements = []

        for e in range(-N, N):
            x0 = e*x_caract*precision
            x1 = (e+1)*x_caract*precision

            elem = LinearElement(x0, x1, [self.current_DOF(), self.next_DOF()])
            elem.V0 = 0.5*m*w*w*((x1+x0)/2)**2
            self.elements.append(elem)

        self.construct_matrices()

            
            
if __name__ == "__main__":
    mesh = MeshHO(300, 0.02)
    eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False)
    for i in sorted(eigvals)[0:20]:
        print("%2.5f eV" % (i/eV))
