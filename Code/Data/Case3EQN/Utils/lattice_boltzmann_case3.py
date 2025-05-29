#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Gemini, Lattice Boltzmann Method for Case 3 equation (linear advection-reaction)
"""
import numpy as np

def LBM_CASE3(f_minus_1, f_0, f_plus_1, advection_velocity, source_term, omega, dt):
    #############################
    # Collision terms (omega)
    #############################

    # Density (macroscopic variable)
    rho_t = f_minus_1 + f_0 + f_plus_1

    # Lattice speed and speed of sound (standard D1Q3 with c=1)
    c = 1.0
    cs_sq = c**2 / 3.0

    # Equilibrium distribution functions for linear advection with constant velocity
    # Based on f_i_eq = w_i * rho * (1 + u * ci / cs^2)
    w = np.array([1/6, 2/3, 1/6]) # Weights for velocities [-1, 0, 1]
    ci = np.array([-c, 0, c])

    # Ensure dimensions match for element-wise multiplication
    # rho_t is a vector of size N, advection_velocity is a scalar
    # Need to handle broadcasting or repeat advection_velocity if it were spatially varying.
    # Here, advection_velocity is constant = 0.1

    # Equilibrium distributions
    f_minus_1_eq = w[0] * rho_t * (1 + advection_velocity * ci[0] / cs_sq)
    f_0_eq = w[1] * rho_t * (1 + advection_velocity * ci[1] / cs_sq)
    f_plus_1_eq = w[2] * rho_t * (1 + advection_velocity * ci[2] / cs_sq)

    # Collision step (BGK) with source term added to the equilibrium
    # Source term distribution: S_i = w_i * S
    S_minus_1 = w[0] * source_term
    S_0 = w[1] * source_term
    S_plus_1 = w[2] * source_term

    # fi_new = fi - omega * (fi - fi_eq) + S_i * dt
    f_minus_1_col = f_minus_1 - omega * (f_minus_1 - f_minus_1_eq) + S_minus_1 * dt
    f_0_col = f_0 - omega * (f_0 - f_0_eq) + S_0 * dt
    f_plus_1_col = f_plus_1 - omega * (f_plus_1 - f_plus_1_eq) + S_plus_1 * dt


    #############################
    # Streaming terms
    #############################

    # Particles moving in the positive direction (index + 1)
    f_plus_1_stream = np.roll(f_plus_1_col, 1) # Roll positively shifts elements (periodic boundary)
    # Handle boundary condition at the right boundary (a_max) - assuming outflow or other BC
    # For df/dt + u df/da = S with u > 0, boundary condition is needed at a=0.
    # At a_max, it's an outflow boundary (values stream out and are not replaced from outside).
    # Simple approach: leave the rolled value at the boundary, effectively allowing outflow.

    # Particles moving in the negative direction (index - 1)
    f_minus_1_stream = np.roll(f_minus_1_col, -1) # Roll negatively shifts elements (periodic boundary)
    # Handle boundary condition at the left boundary (a=0)
    # With advection_velocity > 0, f-1 at a=0 comes from outside the domain. Inflow BC for f-1[0].
    # Let's try a simple zero-gradient (Neumann) boundary condition for f-1 at a=0 for now.
    f_minus_1_stream[0] = f_minus_1_col[1] # Simple zero-gradient boundary condition for f-1 at a=0

    # Particles with zero velocity stay
    f_0_stream = f_0_col

    return f_minus_1_stream, f_0_stream, f_plus_1_stream

def run_lb_case3(f0, a, advection_velocity, source_term, t_max, dt, N, a_max):

    # Parameters
    omega = 1.5 # Relaxation parameter (adjust for stability and accuracy)
    # For stability with advection and source, omega typically needs to be between 0 and 2.

    da = a_max / N
    timesteps = int(t_max / dt)

    # Initialize distribution functions at t=0 (Assuming equilibrium based on initial f0 and advection_velocity)
    c = 1.0 # Lattice speed
    cs_sq = c**2 / 3.0 # Speed of sound squared
    w = np.array([1/6, 2/3, 1/6])
    ci = np.array([-c, 0, c])

    # Calculate initial equilibrium distributions based on f0 and advection_velocity
    f_minus_1 = w[0] * f0 * (1 + advection_velocity * ci[0] / cs_sq)
    f_0 = w[1] * f0 * (1 + advection_velocity * ci[1] / cs_sq)
    f_plus_1 = w[2] * f0 * (1 + advection_velocity * ci[2] / cs_sq)

    # Ensure initial distributions are non-negative
    f_minus_1[f_minus_1 < 0] = 0
    f_0[f_0 < 0] = 0
    f_plus_1[f_plus_1 < 0] = 0


    f_all = np.zeros((timesteps, N))
    f_all[0] = f0

    t_vec = np.arange(timesteps) * dt

    for n in range(1, timesteps):
        # Collision and streaming
        f_minus_1, f_0, f_plus_1 = LBM_CASE3(f_minus_1, f_0, f_plus_1, advection_velocity, source_term, omega, dt)

        # Calculate macroscopic density f
        f_t = f_minus_1 + f_0 + f_plus_1
        f_all[n] = f_t

        # print(f"Timestep {n}/{timesteps}, Time: {t_vec[n]:.4f}")

    return f_all, t_vec 