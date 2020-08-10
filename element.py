# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:50:24 2020

@author: boris
"""

import numpy as np


hbar = 1.054e-16  # kg*mm^2/s
m = 9.109e-31


class LinearElement:
    """Linear FEM element"""

    def __init__(self, x0, x1, ddls):
        self.ddls = ddls
        self.x0 = x0
        self.x1 = x1
        self.L = np.absolute(x1 - x0)

        self.K = None
        self.M = None

    def compute_stiffness(self):
        self.K = 1/self.L * np.matrix([[1, -1], [-1, 1]])

    def get_stiffness(self):
        if self.K is None:
            self.compute_stiffness()
        return self.K

    def compute_mass(self):
        self.M = self.L * np.matrix([[1/3, 1/6], [1/6, 1/3]])

    def get_mass(self):
        if self.M is None:
            self.compute_mass()
        return self.M

    def get_hamiltonian(self):
        """H for a piecewise constant V set as attribute of the element"""
        if self.V0 is None:
            raise ValueError("Element has unimplemented potential")
        return (hbar*hbar/(2*m) * self.get_stiffness()
                + self.V0 * self.get_mass())


class LinearElementBoundary(LinearElement):
    """Linear FEM element with a single DOF (the other being set to 0)"""

    def __init__(self, x0, x1, ddls):
        super().__init__(x0, x1, ddls)

    def compute_stiffness(self):
        self.K = 1/self.L * np.matrix([[1]])

    def compute_mass(self):
        self.M = self.L * np.matrix([[1/3]])
