# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:35:42 2020

@author: boris
"""

import numpy as np

class AbstractMesh:
    """
    Abstract interface for a mesh consisting of a list of elements (self.elements), all having DOFs mapping with DOFS < self.dofs.
    This creates the relevant matrices, of dimension self.dofs.
    "Element" must :
        - Implement get_hamiltonian() and get_mass()
        - Have an attribute "ddls" containing the list of global DOFs to influence
    
    """
    
    def construct_matrices(self):
        nb_dofs = self.dofs
        
        self.H = np.zeros((nb_dofs, nb_dofs))
        self.M = np.zeros((nb_dofs, nb_dofs))
        for elem in self.elements:
            H = elem.get_hamiltonian()
            M = elem.get_mass()
            for i in range(0,2):
                for j in range(0,2):
                    self.H[elem.ddls[i], elem.ddls[j]] = self.H[elem.ddls[i], elem.ddls[j]] + H[i,j]
                    self.M[elem.ddls[i], elem.ddls[j]] = self.M[elem.ddls[i], elem.ddls[j]] + M[i,j]