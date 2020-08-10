# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:09:02 2020

@author: boris
"""

#Parameters
from abstract_mesh import AbstractMesh
from element_2D import RectangleElement
import numpy as np


eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31



class MeshPotentialWell2D(AbstractMesh):

    def __init__(self, w, h, nx, ny):
        super().__init__()
        self.nb_elem = nx*ny
        self.construct(w, h, nx, ny)

    def construct(self, w, h, nx, ny):
        """Precision = Element lenth in multiples of x_caract"""
        self.dof_counter = (nx + 1) * (ny + 1) - 1 #Hardcoded :c
        self.elements = []

        

        for i in range(0, nx):
            for j in range(0, ny):
                x = i * w
                y = j * h
                
                elem = RectangleElement(x, y, w, h, [i + j * (nx + 1),\
                                                     i + j * (nx + 1) + 1,\
                                                     i + (j + 1) * (nx + 1),\
                                                     (i + 1) + (j + 1) * (nx + 1)])
                elem.V0 = 0
                self.elements.append(elem)


        # Remove boundary DOFs
        boundary_DOFs = [i for i in range(0, nx + 1)] + \
                        [i + (ny * (nx + 1)) for i in range(0, nx + 1)] + \
                        [j * (nx + 1) for j in range (0, ny + 1)] + \
                        [j * (nx + 1) + nx for j in range (0, ny + 1)]
        boundary_DOFs = list(dict.fromkeys(boundary_DOFs))
        boundary_DOFs = sorted(boundary_DOFs, reverse = True)
        print(boundary_DOFs)
        self.construct_matrices()
        
        for dof in boundary_DOFs:
            self.remove_DOF(dof)
        
        
if __name__ == "__main__":
    width = nm
    height = 1.5*nm
    nx = 40 #Number of rectangles in x-axis
    ny = 40
    
    E_1_1 = (hbar * np.pi) **2 / (2 * m) * (1 / width**2 + 1 / height**2)
    print("Energie min : %2.3f eV" % (E_1_1 / eV))
    
    # Element size
    w = width / nx
    h = height / ny
    mesh = MeshPotentialWell2D(w, h, nx, ny)
    eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False)
    for i, E in enumerate(sorted(eigvals)[0:20], start=1):
        print("%d : %2.3f eV = %2.3f E_1." % (i, E/eV, E/E_1_1))