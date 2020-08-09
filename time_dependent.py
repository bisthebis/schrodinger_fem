# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:52:33 2020

@author: boris
"""

from infinite_well import MeshPotentialWell
from numpy.linalg import solve
from numpy import vdot  # , dot
from scipy.linalg import eigh

eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31


N = 5
mesh = MeshPotentialWell(nm, N)
H = mesh.H
M = mesh.M

# States are M-normalized : for all x in eigvecs, x*M*x = 1.
# Thus, x*H*x = the eigenvalue
eigvals, eigvecs = eigh(mesh.H, mesh.M, eigvals_only=False)

initial_state = (eigvecs[:, 0] + eigvecs[:, 1]) * 2**-0.5


def schrodinger_derivative(q, H, M):
    # i hbar M dq/dt = Hq
    Mdq_dt = -1j/hbar * H.dot(q)
    dq_dt = solve(M, Mdq_dt)
    return dq_dt


dy = schrodinger_derivative(initial_state, H, M)
dt = 1e-17  # secondes

next_state = initial_state + dy*dt

new_norm = abs(vdot(next_state, M.dot(next_state)))
expected_new_norm = 1 + dt**2 * abs(vdot(dy, M.dot(dy)))
print("Comparaison des normes : %2.5f vs %2.5f, soit %2.3f %% de trop"
      % (new_norm, expected_new_norm, 100*(expected_new_norm-1)))
