# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:55:43 2020

@author: boris
"""
import numpy as np


# Mutualiser ?
hbar = 1.054e-16  # kg*mm^2/s
m = 9.109e-31


class LinearRectangleTriangle:
    """
    Triangle of height b and width a, with DOFs numbered as such :

        3
        | \
        |  \
        1---2
        the first DOF must be the one with the right angle
        DOF 1 is at (x0, y0)

        (Rotations should be OK)
    """

    def __init__(self, x0, y0, a, b, ddls):
        self.ddls = ddls
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b

        self.K = None
        self.M = None

    # TODO : mutualiser l'interface des éléments

    def get_stiffness(self):
        if self.K is None:
            self.compute_stiffness()
        return self.K

    def compute_mass(self):
        self.M = self.a * self.b / 12 * np.matrix([
            [1, 1/2, 1/2],
            [1/2, 1, 1/2],
            [1/2, 1/2, 1]
            ])

    def compute_stiffness(self):
        a = self.a
        b = self.b
        self.K = 0.5 * np.matrix([
                [a/b+b/a, -b/a, -a/b],
                [-b/a, b/a, 0],
                [-a/b, 0, a/b]
                ])

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


class RectangleElement:
    """
    Rectangle of width a and height b with origin (DOF 1) at x0, y0.
    Made of two CSTs paired together

    3---4
    |   |
    |   |
    1---2
    """

    def __init__(self, x0, y0, a, b, ddls):
        self.ddls = ddls
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b

        self.K = None
        self.M = None

    # TODO : mutualiser l'interface des éléments

    def get_stiffness(self):
        if self.K is None:
            self.compute_stiffness()
        return self.K

    def compute_mass(self):
        self.M = self.a * self.b / 12 * np.matrix([
                [1, 1/2, 1/2, 0],
                [1/2, 2, 1, 1/2],
                [1/2, 1, 2, 1/2],
                [0, 1/2, 1/2, 1]
                ])

    def compute_stiffness(self):
        a = self.a
        b = self.b
        self.K = 0.5 * np.matrix([
                [a/b+b/a, -b/a, -a/b, 0],
                [-b/a, a/b+b/a, 0, -b/a],
                [-a/b, 0, a/b+b/a, -a/b],
                [0, -b/a, -a/b, a/b+b/a]
                ])

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


class RectangleElementBoundary(RectangleElement):

    """
    Types:
        0: corner
        1: vertical border
        2: horizontal border
        3: inner corner
               3
               |
               |
            1--2
    """
    def __init__(self, x0, y0, a, b, ddls, elem_type):
        super().__init__(x0, y0, a, b, ddls)
        self.elem_type = elem_type

    def compute_mass(self):
        if self.elem_type == 0:
            self.M = self.a * self.b / 12 * np.matrix([1])
        elif self.elem_type in (1, 2):
            self.M = self.a * self.b / 12 * np.matrix([
                    [1, 1/2],
                    [1/2, 2],
                    ])
        elif self.elem_type == 3:
            self.M = self.a * self.b / 12 * np.matrix([
                    [1, 1/2, 0],
                    [1/2, 2, 1/2],
                    [0, 1/2, 1]
                    ])

    def compute_stiffness(self):
        a = self.a
        b = self.b
        if self.elem_type == 0:
            self.K = 0.5 * np.matrix([a/b])
        elif self.elem_type == 1:
            self.K = 0.5 * np.matrix([
                    [a/b+b/a, -a/b],
                    [-a/b, a/b+b/a],
                    ])
        elif self.elem_type == 2:
            self.K = 0.5 * np.matrix([
                    [a/b+b/a, -b/a],
                    [-b/a, a/b+b/a]
                    ])
        elif self.elem_type == 3:
            self.K = 0.5 * np.matrix([
                    [a/b+b/a, -b/a, 0],
                    [-b/a, a/b+b/a, -b/a],
                    [0, -b/a, a/b+b/a]
                    ])
    
class LinearTriangleElement():

    """
    Triangle with three arbitrary vertices (x_i, y_i). Orientation doesn't matter
    """
    def __init__(self, x1, y1, x2, y2, x3, y3, ddls):
        self.ddls = ddls
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        
        self.jacobian_matrix = np.matrix([
                [x2-x1, x3-x1],
                [y2-y1, y3-y1],
                ])
        
        self.jacobian = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        self.area = 0.5 * abs(self.jacobian)
        
        self.jacobian_inverse = 1/self.jacobian * np.matrix([
                [y3-y1, x1-x3],
                [y1-y2, x2-x1],
                ])

        self.K = None
        self.M = None

    def compute_mass(self):
        self.M = self.jacobian / 12 * np.matrix([
            [1, 1/2, 1/2],
            [1/2, 1, 1/2],
            [1/2, 1/2, 1]
            ])

    def compute_stiffness(self):
        """
        Ugly, but should work :)
        """
        
        self.K = np.empty((3,3))
        dxi_dx = self.jacobian_inverse[0, 0]
        dxi_dy = self.jacobian_inverse[0, 1]
        deta_dx = self.jacobian_inverse[1, 0]
        deta_dy = self.jacobian_inverse[1, 1]
        
        self.K[0, 0] = self.area * ((dxi_dx + deta_dx)**2 + (dxi_dy + deta_dy)**2)
        self.K[1, 1] = self.area * ((dxi_dx)**2 + (dxi_dy)**2)
        self.K[2, 2] = self.area * ((deta_dx)**2 + (deta_dy)**2)
        
        self.K[0, 1] = self.area * -1 * ((dxi_dx + deta_dx) * dxi_dx + (dxi_dy + deta_dy) * dxi_dy)
        self.K[0, 2] = self.area * -1 * ((dxi_dx + deta_dx) * deta_dx + (dxi_dy + deta_dy) * deta_dy)
        self.K[1, 2] = self.area * -1 * (dxi_dx * deta_dx + dxi_dy * deta_dy)
        
        self.K[1, 0] = self.K[0][1]
        self.K[2, 0] = self.K[0][2]
        self.K[2, 1] = self.K[1][2]

    
    def get_mass(self):
        if self.M is None:
            self.compute_mass()
        return self.M
    
    def get_stiffness(self):
        if self.K is None:
            self.compute_stiffness()
        return self.K

    def get_hamiltonian(self):
        """H for a piecewise constant V set as attribute of the element"""
        if self.V0 is None:
            raise ValueError("Element has unimplemented potential")
        return (hbar*hbar/(2*m) * self.get_stiffness()
                + self.V0 * self.get_mass())
