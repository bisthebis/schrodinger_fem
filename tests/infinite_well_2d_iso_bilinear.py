# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:09:02 2020

@author: boris
"""

import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from abstract_mesh import AbstractMesh
from element_2D import RectangleElement, RectangleElementBoundary
from iso_element_2D import BilinearQuadrangleElement
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Parameters
eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31

"""
TODO:
    improve removing of boundary DOFs by creating boundary element like in 1D
    improve computation time by preallocating?
"""


class BilinearMeshPotentialWell2D(AbstractMesh):

    def __init__(self, w, h, nx, ny):
        super().__init__()
        self.nb_elem = nx*ny
        self.construct(w, h, nx, ny)

    def construct(self, w, h, nx, ny):
        
        self.elements = []
        self.dof_counter = (nx+1) * (ny+1) - 1
        
        for i in range(0, nx):
            for j in range(0, ny):
                # Bottom-left corner position
                x = i * w
                y = j * h
                
                elem = BilinearQuadrangleElement([(x,y), (x+w, y), (x+w, y+h), (x, y+h)],
                                                  [i + (nx + 1) * j, i + 1 + (nx + 1) * j, i + 1 + (nx + 1) * (j + 1), i + (nx + 1) * (j + 1)])

                
                elem.V0 = 0
                self.elements.append(elem)

        # Remove boundary DOFs
        boundary_DOFs = [i for i in range(0, nx + 1)] + \
                        [i + (ny * (nx + 1)) for i in range(0, nx + 1)] + \
                        [j * (nx + 1) for j in range (0, ny + 1)] + \
                        [j * (nx + 1) + nx for j in range (0, ny + 1)]
        boundary_DOFs = list(dict.fromkeys(boundary_DOFs))
        boundary_DOFs = sorted(boundary_DOFs, reverse = True)
        self.construct_matrices()
        
        for dof in boundary_DOFs:
            self.remove_DOF(dof)
            

class BiTriangularMeshPotentialWell2D(AbstractMesh):

    def __init__(self, w, h, nx, ny):
        super().__init__()
        self.nb_elem = nx*ny
        self.construct(w, h, nx, ny)

    def construct(self, w, h, nx, ny):
        """Precision = Element lenth in multiples of x_caract"""
        self.dof_counter = (nx - 1) * (ny - 1) - 1  # Hardcoded :c
        self.elements = []

        # bottom left corner
        elem = RectangleElementBoundary(0, 0, w, h, [0], 0)
        elem.V0 = 0
        self.elements.append(elem)
        # bottom right corner
        elem = RectangleElementBoundary((nx-2) * w, 0, w, h, [nx-2], 0)
        elem.V0 = 0
        self.elements.append(elem)
        # top left corner
        elem = RectangleElementBoundary(0, (ny-2) * h, w, h,
                                        [(ny-2)*(nx-1)], 0)
        elem.V0 = 0
        self.elements.append(elem)
        # top right corner
        elem = RectangleElementBoundary((nx-2) * w, (ny-2) * h, w, h,
                                        [(ny-1)*(nx-1)-1], 0)
        elem.V0 = 0
        self.elements.append(elem)

        # top and bottom borders
        for i in range(0, nx-2):
            x = i * w
            j = 0
            y = 0
            elem = RectangleElementBoundary(x, y, w, h,
                                            [i + j * (nx - 1),
                                             i + j * (nx - 1) + 1],
                                            2)
            elem.V0 = 0
            self.elements.append(elem)

            # x = i * w
            j = ny - 2
            y = j * h
            elem = RectangleElementBoundary(x, y, w, h,
                                            [i + j * (nx - 1),
                                             i + j * (nx - 1) + 1],
                                            2)
            elem.V0 = 0
            self.elements.append(elem)

        # left and right borders
        for j in range(0, ny-2):
            i = 0
            x = 0
            y = j * h
            elem = RectangleElementBoundary(x, y, w, h,
                                            [i + j * (nx - 1),
                                             i + (j + 1) * (nx - 1)],
                                            1)
            elem.V0 = 0
            self.elements.append(elem)

            i = nx - 2
            x = (nx - 2) * w
            # y = j * h
            elem = RectangleElementBoundary(x, y, w, h,
                                            [i + j * (nx - 1),
                                             i + (j + 1) * (nx - 1)],
                                            1)
            elem.V0 = 0
            self.elements.append(elem)

        # inner elements
        for i in range(0, nx-2):
            for j in range(0, ny-2):
                x = i * w
                y = j * h

                elem = RectangleElement(x, y, w, h,
                                        [i + j * (nx - 1),
                                         i + j * (nx - 1) + 1,
                                         i + (j + 1) * (nx - 1),
                                         (i + 1) + (j + 1) * (nx - 1)])
                elem.V0 = 0
                self.elements.append(elem)

        self.construct_matrices()

if __name__ == "__main__":
    
    print('--- Test in square well (1 nm wide) ---')
    width = nm
    height = nm
    
    n_min = 20
    n_max = 50
    n_trial = [n for n in range(n_min, n_max + 1)]
    
    # Expected results
    E0 = (hbar * np.pi) ** 2 / (2 * m)
    E_1_1 = E0 * (1 / width**2 + 1 / height**2)
    expected_energies = sorted([E0 * (((x+1)/width)**2 + ((y+1)/height)**2)
                               for x, y in np.ndindex(n_min, n_min)])
    
    # Data to collect
    # Error on the lowest energy level
    first_error_bilinear = []
    first_error_triangular = []
    # Error on th 5th energy level
    fifth_error_bilinear = []
    fifth_error_triangular = []
    # Error on the 20th energy level
    last_error_bilinear = []
    last_error_triangular = []
    
    # Time measurements
    time_in_eigh_bilinear = []
    time_in_eigh_triangular = []
    
    
    for n in n_trial:
        print(("Testing a %dx%d grid..." % (n, n)))
        w = width / n
        h = height / n
        mesh_bilinear = BilinearMeshPotentialWell2D(w, h, n, n)
        mesh_triangular = BiTriangularMeshPotentialWell2D(w, h, n, n)
        
        
        #Computation and timing
        start = timer()
        eigvals_bilinear = eigh(mesh_bilinear.H, mesh_bilinear.M, eigvals_only=True, eigvals=(0,19))
        end = timer()
        time_in_eigh_bilinear.append(end-start)
        
        start = timer()
        eigvals_triangular = eigh(mesh_triangular.H, mesh_triangular.M, eigvals_only=True, eigvals=(0,19))
        end = timer()
        time_in_eigh_triangular.append(end-start)
        
        first_error_bilinear.append(100 * (eigvals_bilinear[0] / expected_energies[0] - 1))
        first_error_triangular.append(100 * (eigvals_triangular[0] / expected_energies[0] - 1))
        fifth_error_bilinear.append(100 * (eigvals_bilinear[4] / expected_energies[4] - 1))
        fifth_error_triangular.append(100 * (eigvals_triangular[4] / expected_energies[4] - 1))
        last_error_bilinear.append(100 * (eigvals_bilinear[19] / expected_energies[19] - 1))
        last_error_triangular.append(100 * (eigvals_triangular[19] / expected_energies[19] - 1))


    ## Plotting
    # Plot of relative errors
    max_height = max(abs(np.array(last_error_triangular)))
    plt.figure()
    plt.ylim([-max_height, max_height])
    plt.plot(n_trial, first_error_bilinear, '-o', label='Bilinéaire, $E_1$')
    plt.plot(n_trial, first_error_triangular, '-o', label='Triangulaire, $E_1$')
    plt.plot(n_trial, fifth_error_bilinear, '-o', label='Bilinéaire, $E_5$')
    plt.plot(n_trial, fifth_error_triangular, '-o', label='Triangulaire, $E_5$')
    plt.plot(n_trial, last_error_bilinear, '-o', label='Bilinéaire, $E_{20}$')
    plt.plot(n_trial, last_error_triangular, '-o', label='Triangulaire, $E_{20}$')

    plt.ylabel('Erreur relative (%)')
    plt.xlabel('Largeur et hauteur du puits, en nombre d''éléments')
    plt.legend()
    plt.savefig('puits_infini.pdf')