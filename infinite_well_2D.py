# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:09:02 2020

@author: boris
"""


from abstract_mesh import AbstractMesh
from element_2D import RectangleElement, RectangleElementBoundary
import numpy as np
from scipy.linalg import eigh

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


class MeshPotentialWell2D(AbstractMesh):

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
    width = nm
    height = 1.5*nm
    nx = 100  # Number of rectangles in x-axis
    ny = 100

    E0 = (hbar * np.pi) ** 2 / (2 * m)
    E_1_1 = E0 * (1 / width**2 + 1 / height**2)
    print("Energie min : %2.3f eV" % (E_1_1 / eV))
    expected_energies = sorted([E0 * (((x+1)/width)**2 + ((y+1)/height)**2)
                               for x, y in np.ndindex(nx, ny)])

    # Element size
    w = width / nx
    h = height / ny
    mesh = MeshPotentialWell2D(w, h, nx, ny)
    # print(mesh.H)
    # print(mesh.M)
    eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False)
    for i, E in enumerate(sorted(eigvals)[0:20], start=1):
        print("%d : %2.3f eV. Expected : %2.3f eV. Excess : %2.3f %%" %
              (i, E/eV, expected_energies[i-1]/eV,
               100 * (E / expected_energies[i-1] - 1)))
