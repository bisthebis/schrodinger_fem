# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:34:12 2020

@author: boris
"""

import numpy as np


# Mutualiser ?
hbar = 1.054e-16  # kg*mm^2/s
m = 9.109e-31

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
        

class BilinearQuadrangleElement():

    """
    Quadrangle with arbitrary vertices (x_i, y_i). Orientation doesn't matter (does it ??)
    "points" is a list of 4 pairs
    Points are in xi-eta space (both in [-1,1])
    Nodes to local space : 
        - 1 : (-1, -1)
        - 2 : (1, -1)
        - 3 : (1, 1)
        - 4 : (-1, 1)
    """
    def __init__(self, points, ddls):
        self.ddls = ddls
        self.xs = np.array([point[0] for point in points])
        self.ys = np.array([point[1] for point in points])
        
        
        self.K = None
        self.M = None

    def eval_jacobian_matrix(self, xi, eta):
        dNi_dxi = 0.25 * np.array([eta - 1, 1 - eta, 1 + eta, -1 - eta])
        dNi_deta = 0.25 * np.array([xi - 1, -1 - xi, 1 + xi, 1 - xi])
        return np.array([
                [self.xs.dot(dNi_dxi), self.xs.dot(dNi_deta)],
                [self.ys.dot(dNi_dxi), self.ys.dot(dNi_deta)]
                ])

    def compute_mass(self, mode='standard'):
        gp = 3**(-0.5)
        Ni = [lambda xi, eta : 0.25*(1-xi)*(1-eta), lambda xi, eta : 0.25*(1+xi)*(1-eta), lambda xi, eta : 0.25*(1+xi)*(1+eta), lambda xi, eta : 0.25*(1-xi)*(1+eta)]
        self.M = np.empty((4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                if mode == 'reduced':
                   self.K[i, j] = 4 * np.linalg.det(self.eval_jacobian_matrix(0,0))
                elif mode == "standard":
                   
                   self.K[i, j] = Ni[i](-gp, -gp) * Ni[j](-gp, -gp) * np.linalg.det(self.eval_jacobian_matrix(-gp, -gp)) + \
                                   Ni[i](gp, gp) * Ni[j](gp, gp) * np.linalg.det(self.eval_jacobian_matrix(gp, gp)) + \
                                   Ni[i](-gp, gp) * Ni[j](-gp, gp) * np.linalg.det(self.eval_jacobian_matrix(-gp, gp)) + \
                                   Ni[i](gp, -gp) * Ni[j](gp, -gp) * np.linalg.det(self.eval_jacobian_matrix(gp, -gp))
                else:
                    raise ValueError("Invalid integration mode")
                


    def _compute_stiffness_integrand(self, i, j, xi, eta):
        """
        Integrand is grad(Ni) . grad(Nj) * J
        where grad(Ni) . grad(Nj) is :
            dNi/dx * dNj/dx + dNi/dy * dNj/dy
            and
            dN/dx_k = dN/dxi * dxi/dx_k + dN/deta * deta/d_xk
        """
        jacobian_matrix = self.eval_jacobian_matrix(xi, eta)
        J = np.linalg.det(jacobian_matrix)
        inverse_jacobian = np.linalg.inv(jacobian_matrix)
                
        dNi_dxi = 0.25 * np.array([eta - 1, 1 - eta, 1 + eta, -1 - eta])
        dNi_deta = 0.25 * np.array([xi - 1, -1 - xi, 1 + xi, 1 - xi])
        
        dNi_dx = dNi_dxi[i] * inverse_jacobian[0, 0] + dNi_deta[i] * inverse_jacobian[1, 0]
        dNj_dx = dNi_dxi[j] * inverse_jacobian[0, 0] + dNi_deta[j] * inverse_jacobian[1, 0]
        
        dNi_dy = dNi_dxi[i] * inverse_jacobian[0, 1] + dNi_deta[i] * inverse_jacobian[1, 1]
        dNj_dy = dNi_dxi[j] * inverse_jacobian[0, 1] + dNi_deta[j] * inverse_jacobian[1, 1]
        
        return (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * J
    
    def compute_stiffness(self, mode='standard'):
        gp = 3**(-0.5)
        self.K = np.empty((4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                if mode == 'reduced':
                   self.K[i, j] = 4 * self._compute_stiffness_integrand(i, j, 0, 0)
                elif mode == "standard":
                   self.K[i, j] =  self._compute_stiffness_integrand(i, j, gp, gp) + \
                                   self._compute_stiffness_integrand(i, j, -gp, gp) + \
                                   self._compute_stiffness_integrand(i, j, gp, -gp) + \
                                   self._compute_stiffness_integrand(i, j, -gp, -gp)
                else:
                    raise ValueError("Invalid integration mode")
        
        

    
    def get_mass(self, mode='standard'):
        if self.M is None:
            self.compute_mass(mode)
        return self.M
    
    def get_stiffness(self, mode='standard'):
        if self.K is None:
            self.compute_stiffness()
        return self.K

    def get_hamiltonian(self):
        """H for a piecewise constant V set as attribute of the element"""
        if self.V0 is None:
            raise ValueError("Element has unimplemented potential")
        return (hbar*hbar/(2*m) * self.get_stiffness()
                + self.V0 * self.get_mass())
