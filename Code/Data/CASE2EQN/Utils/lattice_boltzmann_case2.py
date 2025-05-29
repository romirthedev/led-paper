#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Gemini, based on FHN LBM for CASE2 equation
"""
import numpy as np

def LBM_CASE2(f_minus_1, f_0, f_plus_1, G, omega, dt, da):
    #############################
    # Collision terms (omega)
    #############################

    # Density
    rho_t = f_minus_1 + f_0 + f_plus_1

    # Equilibrium distribution functions for a drift term G(a)
    # For a 1D advection equation df/dt + d/da(Gf) = 0, the equilibrium distribution
    # for a DdQ3 (D1Q3) model can be related to the density rho and the drift G*rho
    # f0_eq = rho_t * (1 - G*dt/(2*da))
    # f_plus_1_eq = rho_t * G*dt/(4*da) + rho_t/2
    # f_minus_1_eq = -rho_t * G*dt/(4*da) + rho_t/2
    # This assumes a collision time tau=dt, leading to omega=2. Let's keep omega general for now.

    # A simpler approach for pure advection with velocity G(a):
    # f_eq[i] = wi * rho * (1 + 3 * u * ci / c^2), where u is fluid velocity (G), c is lattice speed (da/dt)
    # wi = [1/3, 1/3, 1/3] for D1Q3
    # ci = [-1, 0, 1]
    c = da/dt # lattice speed

    # Calculate equilibrium distributions based on density (rho_t) and drift velocity (G)
    f_0_eq = rho_t * (1 - G**2 / (2 * c**2)) # This needs verification for non-constant G
    f_plus_1_eq = rho_t/2 + rho_t * G / (2*c) # This assumes c=1 and needs adaptation
    f_minus_1_eq = rho_t/2 - rho_t * G / (2*c) # This assumes c=1 and needs adaptation

    # Let's try a simpler equilibrium based on pure advection:
    # f_eq for D1Q3: f_0_eq = rho(1 - |u|/c), f_plus_1_eq = rho * max(0, u/(2c) + 1/2), f_minus_1_eq = rho * max(0, -u/(2c) + 1/2)
    # Here u is the drift velocity G(a). Let's assume c=1 for simplicity in this adaptation.
    # This might require careful handling of the boundary conditions and the spatial dependence of G.

    # Let's reconsider based on standard advection LBM with velocity u:
    # f0_eq = rho * (1 - 3u^2 / c^2) # For D1Q3 with ci={-1, 0, 1}
    # f_plus_1_eq = rho * (1/6 + u / (2c) + u^2 / (2c^2)) # Simplified equilibrium for D1Q3
    # f_minus_1_eq = rho * (1/6 - u / (2c) + u^2 / (2c^2)) # Simplified equilibrium for D1Q3
    # Using c=1 and u=G:
    # f0_eq = rho * (1 - 3*G**2) # This doesn't look right for a drift term d/da(Gf)
    # f_plus_1_eq = rho * (1/6 + G/2 + G**2/2) # This is also not directly applicable.

    # Let's go back to the interpretation as a pure advection equation with velocity G(a).
    # The collision step should relax towards an equilibrium that drives the flow with velocity G.
    # For a simple advection (df/dt + G * df/da = 0), the equilibrium is f_eq = rho.
    # But the equation is df/dt + d/da(G*f) = 0. This is a conservative form.

    # Let's try a different approach based on the Bhatnagar-Gross-Krook (BGK) collision operator:
    # fi_new = fi - omega * (fi - fi_eq)
    # For D1Q3, the equilibrium distribution functions are typically defined as:
    # f_0_eq = rho - j^2 / rho # where j is momentum/current
    # f_plus_1_eq = 0.5 * (rho + j)
    # f_minus_1_eq = 0.5 * (rho - j)
    # In our case, the density is f, and the current j is related to G*f. The macroscopic velocity is G.
    # So, using f as density (rho) and G as velocity (u):
    # f_0_eq = f_t * (1 - G**2) # Assuming sound speed cs=1/sqrt(3) and |ci|=1
    # f_plus_1_eq = 0.5 * (f_t + f_t * G) # This is for c=1, cs=1/sqrt(2). For c=1, cs=1/sqrt(3), need to adjust.
    # f_minus_1_eq = 0.5 * (f_t - f_t * G) #

    # Let's assume a simple D1Q3 with speeds {-1, 0, 1} and weights {w-1, w0, w1}.
    # Macroscopic density: rho = f-1 + f0 + f1
    # Macroscopic momentum: j = -f-1 + 0*f0 + 1*f1
    # Equilibrium distributions: f_i^eq = w_i * rho * (1 + u * c_i / cs^2)  where u is the macroscopic velocity.
    # In our case, rho is 'f', the macroscopic velocity is 'G'.
    # For D1Q3 with c=1 and cs^2=1/3 (standard LBM):
    # w-1 = 1/6, w0 = 2/3, w1 = 1/6 --> This is for diffusion.
    # For advection, often simpler weights are used, or the equilibrium is defined differently.

    # Let's try a simpler equilibrium definition for advection:
    # f_plus_1_eq = f_t * (G + 1)/2 # This looks like a simple interpolation/upwind scheme
    # f_minus_1_eq = f_t * (1 - G)/2
    # f_0_eq = 0 # This doesn't conserve mass.

    # Let's go back to the BGK collision idea:
    # fi_new = fi - omega * (fi - fi_eq)
    # streaming: f_plus_1_streamed[i] = f_plus_1_new[i-1], f_minus_1_streamed[i] = f_minus_1_new[i+1]
    # The equation is df/dt + d/da(Gf) = 0. This is a continuity equation with flux J = Gf.
    # In LBM, the flux is related to the first moment of the distribution function.
    # Let's assume a D1Q3 model. The flux is f1 - f-1. So f1 - f-1 should be related to G*f_t.

    # Let's try setting the equilibrium based on the macroscopic variables (f_t and G):
    # f_plus_1_eq = 0.5 * f_t * (1 + G) # Simple advection equilibrium
    # f_minus_1_eq = 0.5 * f_t * (1 - G)
    # f_0_eq = 0 # This still doesn't feel right for D1Q3.

    # Standard D1Q3 equilibrium:
    # w = [1/3, 1/3, 1/3], c = [-1, 0, 1]
    # f_eq[i] = w[i] * rho * (1 + u * c[i] / cs^2)
    # rho = f_t, u = G, cs^2 = da^2 / (3*dt^2)  # This is the speed of sound squared for the lattice

    # Let's assume c=1 (lattice speed)
    # f_minus_1_eq = (1/6) * rho_t * (1 + G * (-1) / cs^2)
    # f_0_eq = (2/3) * rho_t * (1 + G * 0 / cs^2) = (2/3) * rho_t
    # f_plus_1_eq = (1/6) * rho_t * (1 + G * (1) / cs^2)
    # We need to relate cs^2 to the physical parameters da and dt.
    # In standard LBM, cs^2 = c^2 / 3. If we take c=da/dt, then cs^2 = (da/dt)^2 / 3.
    # This looks complicated.

    # Let's try a simpler approach focusing on the flux G*f.
    # The momentum in LBM is j = f1 - f-1.
    # We want this momentum to be G * f_t at equilibrium.
    # Also, f_0 + f_plus_1 + f_minus_1 = f_t.

    # Consider a simple advection scheme using LBM approach:
    # f_plus_1_eq = alpha * f_t * G # Something proportional to G for the positive direction
    # f_minus_1_eq = beta * f_t * (-G) # Something proportional to -G for the negative direction
    # f_0_eq = ...

    # Let's go back to the equilibrium definition that relates to rho and j:
    # f_0_eq = rho - j^2 / (2*rho)  # This form might be for higher order moments
    # f_plus_1_eq = 0.5 * (rho + j)
    # f_minus_1_eq = 0.5 * (rho - j)
    # Here rho = f_t, and j = G * f_t.
    # f_plus_1_eq = 0.5 * (f_t + G * f_t) = 0.5 * f_t * (1 + G)
    # f_minus_1_eq = 0.5 * (f_t - G * f_t) = 0.5 * f_t * (1 - G)
    # f_0_eq = f_t - f_plus_1_eq - f_minus_1_eq = f_t - 0.5*f_t*(1+G) - 0.5*f_t*(1-G) = f_t - 0.5*f_t - 0.5*f_t - 0.5*f_t*G + 0.5*f_t*G = 0
    # This implies f_0_eq = 0 for pure advection.

    # So, let's try this equilibrium for collision:
    # f_plus_1_eq = 0.5 * rho_t * (1 + G)
    # f_minus_1_eq = 0.5 * rho_t * (1 - G)
    # f_0_eq = np.zeros_like(rho_t) # Since we expect no particles staying at the same location due to drift.

    f_plus_1_eq = 0.5 * rho_t * (1 + G)
    f_minus_1_eq = 0.5 * rho_t * (1 - G)
    f_0_eq = np.zeros_like(rho_t) # Assume f0_eq is zero for pure advection with these velocities

    # Ensure equilibrium distributions are non-negative (important for stability)
    f_plus_1_eq[f_plus_1_eq < 0] = 0
    f_minus_1_eq[f_minus_1_eq < 0] = 0
    # f_0_eq[f_0_eq < 0] = 0 # Should be zero anyway

    # Collision step (BGK)
    f_plus_1_col = f_plus_1 - omega * (f_plus_1 - f_plus_1_eq)
    f_minus_1_col = f_minus_1 - omega * (f_minus_1 - f_minus_1_eq)
    f_0_col = f_0 - omega * (f_0 - f_0_eq)

    #############################
    # Streaming terms
    #############################

    f_plus_1_stream = np.zeros_like(f_plus_1_col)
    # Particles moving in the positive direction (index + 1)
    f_plus_1_stream[:-1] = f_plus_1_col[1:]
    # Handle boundary condition at the right boundary (a_max) - assuming outflow/zero gradient for f+1
    f_plus_1_stream[-1] = f_plus_1_col[-1] # Simple extrapolation or boundary condition

    f_minus_1_stream = np.zeros_like(f_minus_1_col)
    # Particles moving in the negative direction (index - 1)
    f_minus_1_stream[1:] = f_minus_1_col[:-1]
    # Handle boundary condition at the left boundary (a=0) - assuming inflow/given boundary condition
    # For d/da(Gf), the boundary condition is often on Gf at a=0.
    # If f is defined on [0, a_max], Gf at a=0 determines incoming flux.
    # For f-1 at a=0, it comes from outside the domain.
    # Let's try a simple zero-flux or inflow boundary condition at a=0 for now.
    # If G > 0 at a=0, f-1 at a=0 comes from outside. If G < 0, f+1 at a=0 comes from i=1.
    # Given the equation df/dt + d/da(Gf) = 0, if G(0) > 0, we need a boundary condition for f at a=0.
    # If G(0) < 0, no boundary condition is needed at a=0 for f.
    # G(a) = 0.434 + 0.2604 * a. G(0) = 0.434 > 0. So we need a boundary condition for f at a=0.
    # The initial condition is non-zero around a=0.2. Let's assume a zero-inflow boundary condition for simplicity for now.
    f_minus_1_stream[0] = f_minus_1_col[0] # Simple boundary condition

    f_0_stream = f_0_col # Particles with zero velocity stay

    return f_minus_1_stream, f_0_stream, f_plus_1_stream

def run_lb_case2(f0, a, G, t_max, dt, N, a_max):

    # Parameters
    # omega = 1.0 # Relaxation parameter. omega = 1/tau. tau = 1 for BGK.
    omega = 1.9 # A value close to 2 for faster relaxation (closer to explicit methods)
    # For stability, omega should be between 0 and 2.

    da = a_max / N
    timesteps = int(t_max / dt)

    # Initialize distribution functions (Assuming equilibrium at t=0)
    # Based on f_t = f_0 at t=0 and u = G
    f_plus_1 = 0.5 * f0 * (1 + G)
    f_minus_1 = 0.5 * f0 * (1 - G)
    f_0 = np.zeros_like(f0) # For pure advection equilibrium

    # Ensure initial distributions are non-negative
    f_plus_1[f_plus_1 < 0] = 0
    f_minus_1[f_minus_1 < 0] = 0

    f_all = np.zeros((timesteps, N))
    f_all[0] = f0

    t_vec = np.arange(timesteps) * dt

    for n in range(1, timesteps):
        # Collision and streaming
        f_minus_1, f_0, f_plus_1 = LBM_CASE2(f_minus_1, f_0, f_plus_1, G, omega, dt, da)

        # Calculate macroscopic density f
        f_t = f_minus_1 + f_0 + f_plus_1
        f_all[n] = f_t

        # print(f"Timestep {n}/{timesteps}, Time: {t_vec[n]:.4f}")

    return f_all, t_vec 