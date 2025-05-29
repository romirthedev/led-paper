#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Gemini, Lattice Boltzmann Method for Case 4 equation (linear advection-reaction)
"""
import numpy as np

def LBM_CASE4(f_minus_1, f_0, f_plus_1, advection_velocity, reaction_term, omega, dt, da):
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

    # Collision step (BGK)
    f_minus_1_col = f_minus_1 - omega * (f_minus_1 - f_minus_1_eq)
    f_0_col = f_0 - omega * (f_0 - f_0_eq)
    f_plus_1_col = f_plus_1 - omega * (f_plus_1 - f_plus_1_eq)

    # Add reaction term after collision (Source term S = -af)
    S = reaction_term * rho_t
    # Distribute source term based on weights (simplest approach)
    f_minus_1_reacted = f_minus_1_col + dt * w[0] * S
    f_0_reacted = f_0_col + dt * w[1] * S
    f_plus_1_reacted = f_plus_1_col + dt * w[2] * S


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

    # Apply Boundary Condition at a=0 (left boundary, index 0): f(t, a=0) = 0 (Dirichlet)
    # Use non-equilibrium extrapolation method for incoming population f_plus_1_stream[0]
    # f_plus_1_stream[0] = f_plus_1_eq[0] + (f_plus_1_reacted[0] - f_plus_1_eq[0]) # This is simple extrapolation, not for Dirichlet
    # For f(t, 0) = 0, and advection_velocity > 0 (meaning f_plus_1 streams *into* the domain at a=0)
    # The density at the boundary rho_t[0] should be 0.
    # rho_t[0] = f_minus_1_stream[0] + f_0_stream[0] + f_plus_1_stream[0] = 0
    # f_minus_1_stream[0] is known from f_minus_1_reacted[1] after streaming.
    # f_0_stream[0] is known from f_0_reacted[0] (stays at boundary).
    # So, f_plus_1_stream[0] = -f_minus_1_stream[0] - f_0_stream[0]
    f_plus_1_stream[0] = -f_minus_1_reacted[1] - f_0_reacted[0]
    # Ensure non-negativity, as distribution functions should be >= 0
    f_plus_1_stream[0] = max(0, f_plus_1_stream[0])

    # Apply Boundary Condition at a_max (right boundary, index N-1): df/da = 0 (Neumann)
    # Use non-equilibrium extrapolation method for outgoing population f_minus_1_stream[N-1]
    # f_minus_1_stream[N-1] = f_minus_1_eq[N-1] + (f_minus_1_reacted[N-1] - f_minus_1_eq[N-1]) # Simple extrapolation, not Neumann
    # For df/da = 0, f[N-1] = f[N-2]. This means rho_t[N-1] = rho_t[N-2].
    # A common Neumann BC implementation is to set the unknown outgoing population based on the known incoming one to satisfy zero flux (f_plus_1 = f_minus_1).
    # The incoming population f_plus_1_reacted[N-1] streams from index N-2.
    # We want f_minus_1_stream[N-1] = f_plus_1_stream[N-1].
    # f_plus_1_stream[N-1] is the value streamed from f_plus_1_reacted[N-2].
    f_minus_1_stream[N-1] = f_plus_1_reacted[N-2] # Set outgoing equal to incoming for f-1 and f+1
    # Ensure non-negativity
    f_minus_1_stream[N-1] = max(0, f_minus_1_stream[N-1])


    return f_minus_1_stream, f_0_stream, f_plus_1_stream

def run_lb_case4(f0, a, advection_velocity, reaction_term, t_max, dt, N, a_max):

    # Parameters
    omega = 1.0 # Start with omega = 1.0 (BGK) for simplicity
    # Adjust omega if needed for stability and accuracy (between 0 and 2).

    da = a_max / N
    timesteps = int(t_max / dt)

    # Initialize distribution functions at t=0 (Assuming equilibrium based on initial f0 and advection_velocity)
    c = 1.0 # Lattice speed
    cs_sq = c**2 / 3.0 # Speed of sound squared
    w = np.array([1/6, 2/3, 1/6])
    ci = np.array([-c, 0, c])

    # Calculate initial equilibrium distributions based on f0 and advection_velocity
    # Ensure f0 has the correct shape (N,)
    if f0.shape != a.shape:
        raise ValueError("Shape of initial condition f0 and spatial grid a must match.")

    # Calculate initial density
    rho_initial = f0 # Since f is the density

    # Calculate initial equilibrium distributions based on initial density and advection velocity
    f_minus_1 = w[0] * rho_initial * (1 + advection_velocity * ci[0] / cs_sq)
    f_0 = w[1] * rho_initial * (1 + advection_velocity * ci[1] / cs_sq)
    f_plus_1 = w[2] * rho_initial * (1 + advection_velocity * ci[2] / cs_sq)


    # Ensure initial distributions are non-negative (important for stability)
    f_minus_1[f_minus_1 < 0] = 0
    f_0[f_0 < 0] = 0
    f_plus_1[f_plus_1 < 0] = 0

    f_all = np.zeros((timesteps, N))
    f_all[0] = f0

    t_vec = np.arange(timesteps) * dt

    for n in range(1, timesteps):
        # Collision and streaming
        f_minus_1, f_0, f_plus_1 = LBM_CASE4(f_minus_1, f_0, f_plus_1, advection_velocity, reaction_term, omega, dt, da)

        # Calculate macroscopic density f
        f_t = f_minus_1 + f_0 + f_plus_1
        f_all[n] = f_t

        # print(f"Timestep {n}/{timesteps}, Time: {t_vec[n]:.4f}")

    return f_all, t_vec 