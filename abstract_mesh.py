# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:35:42 2020

@author: boris
"""

import numpy as np

class AbstractMesh:
    """
    Abstract interface for a mesh consisting of a list of elements (self.elements), all having DOFs mapping with DOFS < self.do_counter. 
    (Typically using current_DOF and next_DOF)
    This creates the relevant matrices, of dimension self.dofs.
    "Element" must :
        - Implement get_hamiltonian() and get_mass()
        - Have an attribute "ddls" containing the list of global DOFs to influence
    
    """
    def __init__(self):
        self.dof_counter = 0
    
    def construct_matrices(self):
        nb_dofs = self.dof_counter + 1
        
        self.H = np.zeros((nb_dofs, nb_dofs))
        self.M = np.zeros((nb_dofs, nb_dofs))
        for elem in self.elements:
            H = elem.get_hamiltonian()
            M = elem.get_mass()
            elem_dof_counter = len(elem.ddls)
            for i in range(0, elem_dof_counter):
                for j in range(0, elem_dof_counter):
                    self.H[elem.ddls[i], elem.ddls[j]] = self.H[elem.ddls[i], elem.ddls[j]] + H[i,j]
                    self.M[elem.ddls[i], elem.ddls[j]] = self.M[elem.ddls[i], elem.ddls[j]] + M[i,j]
                
    def current_DOF(self):
        """
        Returns the current DOF identifier
        """
        return self.dof_counter
                    
    def next_DOF(self):
        """
        Returns the next unused DOF identifier, and advance the counter
        """
        self.dof_counter = self.dof_counter + 1
        return self.dof_counter
    
    def remove_DOF(self, i):
        """
        Removes the row and column related to a given DOF. Be careful : this changes the ordering of DOFs ! 
        Reconstructing the matrices will cancel the changes.
        """
        self.H = np.delete(np.delete(self.H, i, 0), i, 1)
        self.M = np.delete(np.delete(self.M, i, 0), i, 1)