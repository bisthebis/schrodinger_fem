from math import ceil
# math.ceil returns an integer while np.ceil returns a float64
import numpy as np
from scipy.linalg import eigh
from abstract_mesh import AbstractMesh
from element import LinearElement, LinearElementBoundary


"""
TODO:
    write a proper description of the class and of its features
    implement a bias (varying element size)
    implement a choice of element type (linear, cubic)
    implement boundary condition (homogeneous Dirichlet for now)
    implement graphical visualization of the eigenvalues/functions
"""


class Mesh(AbstractMesh):
    def __init__(self, boundaries, potential, nb_element=100,
                 element_size=None):
        super().__init__()
        length_domain = abs(boundaries[1]-boundaries[0])
        # element_size overrides nb_element
        if element_size is None:
            element_size = length_domain/nb_element
        else:
            nb_element = ceil(length_domain/element_size)
            # redefines element_size in order to match the nb of element
            element_size = length_domain/nb_element
        # creation of the elements
        self.elements = []
        x0 = boundaries[0]
        x1 = x0 + element_size
        ddls = [self.current_DOF()]
        new_element = LinearElementBoundary(x0, x1, ddls)
        new_element.V0 = potential((x0+x1)/2)
        self.elements.append(new_element)

        for i in range(1, nb_element-1):
            x0 = x1
            x1 = x0 + element_size
            ddls = [self.current_DOF(), self.next_DOF()]
            new_element = LinearElement(x0, x1, ddls)
            new_element.V0 = potential((x0+x1)/2)
            self.elements.append(new_element)

        x0 = x1
        x1 = x0 + element_size
        ddls = [self.current_DOF()]
        new_element = LinearElementBoundary(x0, x1, ddls)
        new_element.V0 = potential((x0+x1)/2)
        self.elements.append(new_element)

    def solve(self, eigFuncs=False, nbOfSolutions=10):
        self.construct_matrices()
        if eigFuncs:
            eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False,
                                    subset_by_index=[0, nbOfSolutions-1])
            return sorted(eigvals), sorted(eigvecs)
        else:
            eigvals = eigh(self.H, self.M, eigvals_only=True,
                           subset_by_index=[0, nbOfSolutions-1])
            return sorted(eigvals)
