# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:09:02 2020

@author: boris
"""

#Parameters
from abstract_mesh import AbstractMesh
from element_2D import RectangleElement
import numpy as np
from scipy.linalg import eigh

eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31

"""
TODO:
    improve removing of boundary DOFs by creating boundary element like in 1D
    improve computation time by preallocating?
"""

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
    ny = 60
    
    E0 = (hbar * np.pi) **2 / (2 * m)
    E_1_1 = E0 * (1 / width**2 + 1 / height**2)
    print("Energie min : %2.3f eV" % (E_1_1 / eV))
    expected_energies = sorted([E0 * (((x+1)/width)**2 + ((y+1)/height)**2) for x, y in np.ndindex(nx, ny)])
    
    # Element size
    w = width / nx
    h = height / ny
    mesh = MeshPotentialWell2D(w, h, nx, ny)
    eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False)
    for i, E in enumerate(sorted(eigvals)[0:20], start=1):
        print("%d : %2.3f eV. Expected : %2.3f eV. Excess : %2.3f %%" % (i, E/eV, expected_energies[i-1]/eV, 100 * (E / expected_energies[i-1] - 1)))

