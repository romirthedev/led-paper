#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Gemini, Lattice Boltzmann Method for Case 4 equation (linear advection-reaction)
"""
import numpy as np

def LBM_CASE4(f_minus_1, f_0, f_plus_1, advection_velocity, reaction_term, omega, dt):
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

    # Equilibrium distributions
    f_minus_1_eq = w[0] * rho_t * (1 + advection_velocity * ci[0] / cs_sq)
    f_0_eq = w[1] * rho_t * (1 + advection_velocity * ci[1] / cs_sq)
    f_plus_1_eq = w[2] * rho_t * (1 + advection_velocity * ci[2] / cs_sq)

    # Collision step (BGK) with reaction term
    # The reaction term is -af. Adding it as S_i = w_i * S where S = -af.

    reaction_term_value = reaction_term * rho_t
    reaction_minus_1 = w[0] * reaction_term_value
    reaction_0 = w[1] * reaction_term_value
    reaction_plus_1 = w[2] * reaction_term_value

    # fi_coll = fi - omega * (fi - fi_eq)
    # fi_new = fi_coll + dt * Reaction_i

    f_minus_1_col = f_minus_1 - omega * (f_minus_1 - f_minus_1_eq)
    f_0_col = f_0 - omega * (f_0 - f_0_eq)
    f_plus_1_col = f_plus_1 - omega * (f_plus_1 - f_plus_1_eq)

    # Add reaction term after collision
    f_minus_1_reacted = f_minus_1_col + dt * reaction_minus_1
    f_0_reacted = f_0_col + dt * reaction_0
    f_plus_1_reacted = f_plus_1_col + dt * reaction_plus_1


    #############################
    # Streaming terms and Boundary Conditions
    #############################

    N = len(f_0_reacted) # Get the size of the domain

    # Streaming for interior nodes
    f_plus_1_stream = np.zeros_like(f_plus_1_reacted)
    f_plus_1_stream[1:] = f_plus_1_reacted[:-1] # Particles moving right

    f_minus_1_stream = np.zeros_like(f_minus_1_reacted)
    f_minus_1_stream[:-1] = f_minus_1_reacted[1:] # Particles moving left

    f_0_stream = f_0_reacted # Particles with zero velocity stay

    # Apply Boundary Condition at a=0 (left boundary, index 0): f(t, a=0) = 0
    # This is a Dirichlet BC on the macroscopic density. We need to set the incoming population (f_plus_1_stream[0]).
    # The outgoing population f_minus_1_reacted[0] streams to the left and is lost.
    # The particle f_0_reacted[0] stays at the boundary.
    # Macroscopic density at boundary: rho_t[0] = f_minus_1_stream[0] + f_0_stream[0] + f_plus_1_stream[0]
    # Here f_minus_1_stream[0] is the value from f_minus_1_reacted[1] after streaming.
    # We want rho_t[0] = 0.
    # f_plus_1_stream[0] = 0 - f_minus_1_stream[0] - f_0_stream[0]
    f_plus_1_stream[0] = -f_minus_1_reacted[1] - f_0_reacted[0]
    # Ensure non-negativity for stability, though this might affect accuracy near the boundary
    f_plus_1_stream[0] = max(0, f_plus_1_stream[0])


    # Apply Boundary Condition at a_max (right boundary, index N-1): df/da = 0 (Neumann)
    # This means the flux is zero at the boundary: J = f_plus_1 - f_minus_1 = 0 => f_plus_1 = f_minus_1
    # We need to set the outgoing population f_minus_1_stream[N-1].
    # The incoming population f_plus_1_reacted[N-1] streams from the left (index N-2).
    # The particle f_0_reacted[N-1] stays at the boundary.
    # A simple Neumann BC implementation in LBM is to set the unknown outgoing population equal to the known incoming one.
    # f_minus_1_stream[N-1] = f_plus_1_reacted[N-1]
    # Or, using non-equilibrium extrapolation:
    # f_minus_1_stream[N-1] = f_minus_1_col_reacted[N-2] + (f_minus_1_eq[N-2] - f_minus_1_col_reacted[N-2]) # Simple extrapolation
    # A common and often stable Neumann BC: set the outgoing population based on the density and known populations to satisfy zero flux.
    # J[N-1] = f_plus_1_stream[N-1] - f_minus_1_stream[N-1] = 0 => f_minus_1_stream[N-1] = f_plus_1_stream[N-1]
    # f_plus_1_stream[N-1] comes from f_plus_1_reacted[N-2] after streaming.
    # Let's use a simple copy of the distribution function from the interior node (zero gradient on f-1).
    f_minus_1_stream[N-1] = f_minus_1_reacted[N-2] # This is a simple extrapolation for f-1


    return f_minus_1_stream, f_0_stream, f_plus_1_stream

def run_lb_case4(f0, a, advection_velocity, reaction_term, t_max, dt, N, a_max):

    # Parameters
    omega = 1.5 # Relaxation parameter (adjust for stability and accuracy)
    # Try adjusting omega if unstable. Closer to 1 is more diffusive, closer to 2 is less diffusive but can be unstable.

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

    # Ensure initial distributions are non-negative (important for stability)
    f_minus_1[f_minus_1 < 0] = 0
    f_0[f_0 < 0] = 0
    f_plus_1[f_plus_1 < 0] = 0

    f_all = np.zeros((timesteps, N))
    f_all[0] = f0

    t_vec = np.arange(timesteps) * dt

    for n in range(1, timesteps):
        # Collision and streaming
        f_minus_1, f_0, f_plus_1 = LBM_CASE4(f_minus_1, f_0, f_plus_1, advection_velocity, reaction_term, omega, dt)

        # Calculate macroscopic density f
        f_t = f_minus_1 + f_0 + f_plus_1
        f_all[n] = f_t

        # print(f"Timestep {n}/{timesteps}, Time: {t_vec[n]:.4f}")

    return f_all, t_vec 