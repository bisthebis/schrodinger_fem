# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:37:30 2020

@author: boris
"""

import numpy as np
from element import LinearElement, LinearElementBoundary
from abstract_mesh import AbstractMesh
from scipy.linalg import eigh

eV = 1.602e-1 #kg*nm^2/s^2
hbar = 1.054e-16 #kg*mm^2/s
nm = 1
m = 9.109e-31


class MeshPotentialWell(AbstractMesh):
    
    def __init__(self, L, element_count):
        super().__init__()
        self.nb_elem = element_count
        self.L = L
        
        self.construct()
        
    def construct(self):
        """Precision = Element lenth in multiples of x_caract"""
        self.dofs = self.nb_elem + 1 # Not taking in account boundary conditions
        self.elements = []
        
        element_length = self.L / self.nb_elem
        
        for i in range (0, self.nb_elem):
            x0 = i * element_length
            x1 = (i + 1) * element_length
                        
            if i == 0 or i == self.nb_elem - 1:
                elem = LinearElementBoundary(x0, x1, [self.current_DOF()])
            else:
                elem = LinearElement(x0, x1, [self.current_DOF(), self.next_DOF()])

            
            elem.V0 = 0
            self.elements.append(elem)
            
        self.construct_matrices()

        


# Main
if __name__ == "__main__":
    L = nm
    E_1 = hbar**2 * np.pi**2 / (2*m*L**2)
    print("Energie théorique : %2.3f eV" % (E_1/eV))
    mesh = MeshPotentialWell(L, 500)
    eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False)
    for i, E in enumerate(sorted(eigvals)[0:20], start=1):
        print("%d : %2.3f eV = %2.3f E_1. Ratio par rapport à la valeur théorique : %2.5f" % (i, E/eV, E/E_1, E/E_1/i**2))