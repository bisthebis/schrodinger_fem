# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:52:33 2020

@author: boris
"""

from infinite_well import MeshPotentialWell
from mesh_oscillator import MeshHO
from numpy.linalg import solve
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

eV = 1.602e-1  # kg*nm^2/s^2
hbar = 1.054e-16  # kg*mm^2/s
nm = 1
m = 9.109e-31


N = 100
#mesh = MeshOH(N, 0.02)
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

def state_norm(q, M):
    return np.vdot(q, M.dot(q))**0.5

dy = schrodinger_derivative(initial_state, H, M)
dt = 1e-17  # secondes

def normalized(q, M):
    return q / state_norm(q, M)

def energy_expectation(q, H, M):
    return np.vdot(q, H.dot(q)) / np.vdot(q, M.dot(q))

def semi_implicit_matrices(H, M, dt, theta):
    """
    (i*hbar*M - theta*dt*H) q_(n+1) = (i*hbar*M + dt*(1-theta)H) q_(n).
    Returns the first and second matrices, i.e. the tuple (A,B) such that Aq_(n+1) = Bq_n
    theta = 0 => explicit
    theta = 1 => implicit
    theta = 1/2 => Crank-Nicholson
    """
    return (1j * hbar*M - theta * dt * H, 1j*hbar*M + (1-theta) * dt * H)

def crank_nicholson_matrices(H, M, dt):
    """
    (i*hbar*M - dt/2*H) q_(n+1) = (i*hbar*M + dt/2*H) q_(n).
    Returns the first and second matrices, i.e. the tuple (A,B) such that Aq_(n+1) = Bq_n
    """
    #return (1j*hbar*M - dt/2*H, 1j*hbar*M + dt/2*H)
    return semi_implicit_matrices(H, M, dt, 0.5)

def implicit_matrices(H, M, dt):
    """
    (i*hbar*M - dt*H) q_(n+1) = (i*hbar*M) q_(n).
    Returns the first and second matrices, i.e. the tuple (A,B) such that Aq_(n+1) = Bq_n
    """
    return (1j*hbar*M - dt*H, 1j*hbar*M)



#Time loop
nb_iterations = 150
dt = 1e-5
def run_explicit():
    print("EXPLICIT SCHEME")
    states = [initial_state]
    print("Energies initiales : %2.4f et %2.4f (eV). Moyenne = %2.4f eV" % (eigvals[0] / eV, eigvals[1] / eV, 0.5 * (eigvals[0] + eigvals[1]) / eV))
    
    for i in range (0, nb_iterations):
        dy = dt * schrodinger_derivative(states[-1], H, M)
        states.append(normalized(states[-1] + dy, M))
        print("%d : énergie = %2.4f eV, norme = %2.6f" % (i, energy_expectation(states[-1], H, M) / eV, state_norm(states[-1], M)))
        
    energies = [energy_expectation(states[i], H, M) / eV for i in range(0, nb_iterations)]
    plt.plot(energies)
    
def run_semi_implicit(theta):
    print("SEMI-IMPLICIT SCHEME. THETA = %1.5f" % (theta))
    states = [initial_state]
    print("Energies initiales : %2.4f et %2.4f (eV). Moyenne = %2.4f eV" % (eigvals[0] / eV, eigvals[1] / eV, 0.5 * (eigvals[0] + eigvals[1]) / eV))
    (A, B) = semi_implicit_matrices(H, M, dt, theta)
    
    for i in range (0, nb_iterations):
        new_state = solve(A, B.dot(states[-1]))
        states.append(new_state)
        print("%d : énergie = %2.4f eV, norme = %2.6f" % (i, energy_expectation(states[-1], H, M) / eV, state_norm(states[-1], M)))
        
    energies = [energy_expectation(states[i], H, M) / eV for i in range(0, nb_iterations)]
    print("E MAX, MIN : %2.5f eV et %2.5f eV" % (max(energies).real, min(energies).real))
    plt.plot(energies)


#run_explicit()
for theta in [0.4999, 0.5, 0.5001]:
    run_semi_implicit(theta)

