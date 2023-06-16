#!/usr/bin/env python3

# Quantum Examples, some examples for Quantum Information Processing.
# Copyright (C) 2022-2023  Dirk "YouDirk" Lehmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import examplelib as el

import numpy as np

import qutip as qt
import qutip.qip.operations as qo

# ********************************************************************

# To simulate (time ordering is left to right)
# ============================================
#
# |0> ---|H|---*---- psi_out_0  \
#              |                 } bell_state_00 = 1/sqrt(2)*(|00> + |11>)
# |0> --------|+|--- psi_out_1  /
#
#
# Angles alpha/phi relative to sigmax for measurement
# ===================================================
#
#   a|0> := alpha_a|0>               = -pi/4
#   b|0> := alpha_b|0>               = -1/12*pi
#   c|0> := alpha_c|0>               =  1/12*pi
#   c|1> := alpha_c|1> = c|0> - pi/2 = -5/12*pi
#
#   => alpha_i is element of {a|0>, b|0>, c|0>, c|1>}
#
#
# Measurement (time ordering is left to right)
# ============================================
#
#             .-----------------.
#             |                 |
#             |  RNG {0, 1, 2}  |
#             |                 |
#             |  0: a|0>  b|0>  |
#             |  1: a|0>  c|0>  |
#             |  2: b|0>  c|1>  |
#             |     /\   /\     |
#             '-----**---**-----'
#                   ||   ||
#           alpha_0 /|   |\ alpha_1
#            phi_0  \|   |/  phi_1
#                   ||   ||
#                   **||||||||||||||||||||||||||||||||||||||||||||||
#                   ||   ||                                       ||
#                   ||   **||||||||||||||||||||||||||||||||||||   ||
#                   ||   ||                                  ||   ||
#             .-----**---**------.         .-----------.   .-**---**-.
# |0> -|H|-*--|                  |  +1/-1  |           |   | ~~~~~~~ |
#          |  |      MEAS_O      |||||||||||  Filter   ||||| Counter |
# |0> ----|+|-| (rel. to sigmax) |         | energy=+1 |   | ~~~~~~~ |
#             '------------------'         '-----------'   '---------'
#
#
# Hilbert space
# =============
#
# psi_out_1 (i.e. Alice)              psi_out_0 (i.e. Bob)
# ''''''''''''''''''''''              ''''''''''''''''''''
# ^ |1>                               ^ |1>
# |/                                  |
# *****.         c|0>                 *****.         c|0>
# |   '******.   /    +delta_alpha    |   '******.   /    +delta_alpha
# |         '***M**. /                |         '***M**. /
# |            /  '*/.                |            /  '*/.
# |           /    /'**               |           /    /'**
# |          /    /   **              |          /    /   **
# |         /         ~MM <-- sigma_x |         /         ~MM <-- sigma_x
# |        /        ~~  **            |        /        ~~  **
# |       /       ~~  -- -delta_alpha |       /       ~~  -- -delta_alpha
# |      /      ~~   /   **           |      /      ~~   /   **
# |     /     ~~         **           |     /     ~~         **
# |    /    ~~          ++MM <-- b|0> |    /    ~~          ++MM <-- b|0>
# |   /   ~~       +++++  **          |   /   ~~       +++++  **
# |  /  ~~    +++++       ** a|0>     |  /  ~~    +++++       ** a|0>
# | / ~~ +++++            ** /        | / ~~ +++++            ** /
# |/~~\++                 **/         |/~~\++                 **/
# '----\------------------**->        '----\------------------**->
#       \                  |                \                  |
#       alpha_1            |0>              alpha_0            |0>
#
# In classical predicate logic we would assume that if we measure
# psi_out_1 (commonly called 'Alice') statistically, i.e. P(psi_out_1
# | alpha_1), in a first experiment followed by a second experiment in
# which we are measuring statistically psi_out_0 (commonly called
# 'Bob'), i.e. P(psi_out_0 | alpha_0), then if Alice and Bobs quantum
# bits act in the same manner by entangle them with the Bell State
# b_00
#
#   Bell State
#   ----------
#
#     b_00 = 1/sqrt(2)(|00> + |11>)
#
# we would expect that if Alice and Bob measure them together in one
# single experiment
#
#   Probabilistic Markov logic
#   --------------------------
#
#     P(psi_out_1 | alpha_1      AND  psi_out_0 | alpha_0)
#       = P(psi_out_1 | alpha_1)  *   P(psi_out_0 | alpha_0)
#       = cos^2(alpha_1)          *   cos^2(alpha_0)
#
# We would expect that the logical conjunction AND will be
# mathematically realized by multiply the probabilities, which we get
# by implement the experiments successive in sequence.  In that case
#
#   Bell's inequality
#   -----------------
#
#     P(a_|0>, b_|0>) <= P(a_|0>, c_|0>) + P(b_|0>, c_|1>)
#
# will be satisfied.  But in quantum logic (and in practice) this is
# not true.
#
# In quantum logic instead, it seems that an logical AND predicate is
# a subtraction of the angles in Hilbert space.  It seems that
#
#   Quantum logic
#   -------------
#
#     P(psi_out_1 | alpha_1      AND  psi_out_0 | alpha_0)
#       = cos^2(alpha_1           -   alpha_0)
#
# In that case of quantum logic, Bell's inequality will not be
# satisfied, and the sum of it's right-hand side P(a_|0>, c_|0>) +
# P(b_|0>, c_|1>) is lower than the left-hand side P(a_|0>, b_|0>)
# for every delta_alpha in the picture above.
#
# Therefore the quantum logic contradicts Bell's inequality.
#
#
# Result of simulation
# ====================
#   Expectations are part of the simulation output.

# ********************************************************************
# Output result

class MySim (el.DefaultSim):

    # alpha = Theta/2 = phi/4
    # alpha: Hilbert space, Theta: Bloch sphere, phi: rotation operator
    DELTA_ALPHA               = 1/12

    ALPHA_A0                  = -1/4
    ALPHA_B0                  = -DELTA_ALPHA
    ALPHA_C0                  = +DELTA_ALPHA
    ALPHA_C1                  = +DELTA_ALPHA - 1/2

    # Rotation in Hilbert space (alpha) around Y axis in Bloch sphere
    # coordinates (Theta) is for rotation operator (phi):
    #
    #   * phi = 2*Theta = 4*alpha
    #
    # Therefore, measuring to sigmax (means to |+> basis) is if
    #   phi = 2 * Theta      = 2 * (2 * alpha)
    #       = 2 * (1/2 * pi) = 2 * (2 * 1/4 * pi)
    #       = pi

    # DELTA_PHI = 1/3 pi            <=> delta_alpha = 1/12 pi
    DELTA_PHI                 = 4*DELTA_ALPHA
    PHI_A0                    = 4*ALPHA_A0
    PHI_B0                    = 4*ALPHA_B0
    PHI_C0                    = 4*ALPHA_C0
    PHI_C1                    = 4*ALPHA_C1

    def pre_measurement(self, custom_args: dict, sim_output: qt.Qobj,
                        measurement_ops: object) -> (qt.Qobj, object):
        custom_args['measurement_case'] = np.random.choice(range(3))

        match custom_args['measurement_case']:
            case 0:
                # O_sigmax(a_|0>, b_|0>) = O_sigmax(-1/4*pi, -1/12*pi)
                #
                # Probability in predicate/first-order logic (satisfy
                # Bell's inequality):
                #   P(a_|0> AND b_|0>)
                #     = cos( -1/4*pi )**2 * cos( -1/12*pi )**2
                #     = (2+sqrt(3))/8
                #     = 0.46650635094610965
                # Probability in quantum logic (contradicts Bell's
                # inequality):
                #   P(a_|0>, b_|0>)
                #     = cos( (-1/4*pi) - (-1/12*pi) )**2
                #     = 3/4 = 0.75
                #
                # !!! Consider: P(a_|0> AND b_|0>) == P(a_|0> AND c_|0>)
                #          but: P(a_|0>,    b_|0>) != P(a_|0>   , c_|0>)
                phi    = [self.PHI_A0*np.pi, self.PHI_B0*np.pi]
            case 1:
                # O_sigmax(a_|0>, c_|0>) = O_sigmax(-1/4*pi, 1/12*pi)
                #
                # Probability in predicate/first-order logic (satisfy
                # Bell's inequality):
                #   P(a_|0> AND c_|0>)
                #     = cos( -1/4*pi )**2 * cos( 1/12*pi )**2
                #     = (2+sqrt(3))/8
                #     = 0.46650635094610965
                # Probability in quantum logic (contradicts Bell's
                # inequality):
                #   P(a_|0>, c_|0>)
                #     = cos( (-1/4*pi) - (1/12*pi) )**2
                #     = 1/4 = 0.25
                #
                # !!! Consider: P(a_|0> AND b_|0>) == P(a_|0> AND c_|0>)
                #          but: P(a_|0>,    b_|0>) != P(a_|0>   , c_|0>)
                phi    = [self.PHI_A0*np.pi, self.PHI_C0*np.pi]
            case 2:
                # O_sigmax(b_|0>, c_|1>) = O_sigmax(-1/4*pi, 1/4*pi)
                #
                # Probability in predicate/first-order logic (satisfy
                # Bell's inequality):
                #   P(b_|0> AND c_|1>)
                #     = cos( -1/12*pi )**2 * sin(    1/12*pi     )**2
                #     = cos( -1/12*pi )**2 * cos( 1/12*pi - pi/2 )**2
                #     = cos( -1/12*pi )**2 * cos(    5/12*pi     )**2
                #     = 1/16
                #     = 0.0625
                # Probability in quantum logic (contradicts Bell's
                # inequality):
                #   P(b_|0>, c_|1>)
                #     = cos( (-1/12*pi) - (1/12*pi - pi/2) )**2
                #     = 1/4 = 0.25
                #
                # Consider: phi = 2*Theta = 4*alpha
                #               = 4*(pi/2) + DELTA_PHI
                phi    = [self.PHI_B0*np.pi, self.PHI_C1*np.pi]
            case _:
                raise AssertionError("Just 3 combinations are possible!")

        # Measuring first bit (bit 0)
        #o_kx = qt.tensor(qt.qeye(2), qo.ry(phi[0])*qt.sigmax())
        #
        # Measuring second bit (bit 1)
        #o_xk = qt.tensor(qo.ry(phi[1])*qt.sigmax(), qt.qeye(2))
        #
        # Measuring both bits at the !THE SAME TIME!
        #o_both = o_kx * o_xk

        # Same as O_BOTH, but shorter code.
        #
        # The relation is, for N in interval [0, n-1]:
        #
        #   O = tensor{N}( [R_y(phi)*sigmaz] )
        #     = product{N}( O_N )
        #     = product{N}( I_2^{n - N - 1} (tensor) R_y(phi)*sigmaz
        #                   (tensor) I_2^N )
        #     =   I_2^{2 - 1 - 1} (tensor) R_y(phi)*sigmaz (tensor) I_2^1
        #       * I_2^{2 - 0 - 1} (tensor) R_y(phi)*sigmaz (tensor) I_2^0
        #
        #     =   R_y(phi)*sigmaz (tensor) I_2^1
        #       * I_2^1 (tensor) R_y(phi)*sigmaz
        #     =   O_KX * O_XK
        #
        # For phi=pi it results in
        #   O = tensor{N}( [sigmax] )
        #     = sigmax (tensor) I_2^1 * I_2^1 (tensor) sigmax

        meas_o = qt.tensor(qo.ry(phi[1]) * qt.sigmax(),
                           qo.ry(phi[0]) * qt.sigmax())

        return sim_output, meas_o

    DELTA_ENERGY_THRESHOLD = 1e-6
    def analyse_sim_result(self, sim_results: dict, sim_runs: int):
        # Means that we are measuring |00> or |11> with a probability
        # of 1.0 if both measurement angles alpha/phi are 0.0*pi.
        energy_00 = 1

        cases_runs   = np.array([0, 0, 0])
        cases_counts = np.array([0, 0, 0])
        for count, energy, collapsed, custom_args in sim_results.values():
            cur_case = custom_args['measurement_case']

            cases_runs[cur_case] += count

            delta_energy = abs(energy - energy_00)
            if delta_energy > self.DELTA_ENERGY_THRESHOLD: continue

            cases_counts[cur_case] += count

        # ---

        print(
          ("\n**** For delta_alpha=%.4f*pi in Hilbert space"
           + " (delta_phi=%.4f*pi) measuring is"
         + "\n**** a|0> := alpha_a|0>               = % 2.4f*pi"
         + "\n**** b|0> := alpha_b|0>               = % 2.4f*pi"
         + "\n**** c|0> := alpha_c|0>               = % 2.4f*pi"
         + "\n**** c|1> := alpha_c|1> = c|0> - pi/2 = % 2.4f*pi"
         + "\n****")
            % (self.DELTA_ALPHA, self.DELTA_PHI,
               self.ALPHA_A0, self.ALPHA_B0, self.ALPHA_C0,
               self.ALPHA_C1))

        lhs_fol = np.cos( self.ALPHA_A0*np.pi )**2 \
                * np.cos( self.ALPHA_B0*np.pi )**2
        rhs_fol = [
            np.cos( self.ALPHA_A0*np.pi )**2
          * np.cos( self.ALPHA_C0*np.pi )**2,
            np.cos( self.ALPHA_B0*np.pi )**2
          * np.cos( self.ALPHA_C1*np.pi )**2 ]
        print(
          ("**** Expected: Bell's inequality in   Predicate/First-Order"
           + " Logic:"
         + "\n****   P(a|0>       AND b|0>)            <= P(a|0>       "
           + "AND c|0>)            + P(b|0>         AND c|1>)"
         + "\n****   cos^2(a|0>)    *cos^2(b|0>)       <= cos^2(a|0>)    "
           + "*cos^2(c|0>)       + cos^2(b|0>)      *cos^2(c|1>)"
         + "\n****   cos^2(% 2.2f*pi)*cos^2(% 2.4f*pi) <= cos^2(% 2.2f*pi)"
           + "*cos^2(% 2.4f*pi) + cos^2(% 2.4f*pi)*cos^2(% 2.4f*pi)"
         + "\n****   <=>  %.4f <= %.4f + %.4f"
         + "\n****   <=>  %.4f <= %.4f"
         + "\n****        = %s"
         + "\n****    =>  %s"
         + "\n****")
            % (self.ALPHA_A0, self.ALPHA_B0, self.ALPHA_A0,
               self.ALPHA_C0, self.ALPHA_B0, self.ALPHA_C1,
               lhs_fol, rhs_fol[0], rhs_fol[1],
               lhs_fol, sum(rhs_fol), lhs_fol <= sum(rhs_fol),
               "Success!" if lhs_fol <= sum(rhs_fol) else "FAILURE!"))

        lhs_qul =  np.cos(
            (self.ALPHA_A0 - self.ALPHA_B0)*np.pi )**2
        rhs_qul = [
          np.cos( (self.ALPHA_A0 - self.ALPHA_C0)*np.pi )**2,
          np.cos( (self.ALPHA_B0 - self.ALPHA_C1)*np.pi )**2]
        print(
          ("**** Expected: Bell's inequality in   Quantum Logic:"
         + "\n****   P(a|0>     AND b|0>)        <= P(a|0>     AND "
           + "c|0>)        + P(b|0>       AND c|1>)"
         + "\n****   cos^2(a|0>   - b|0>)        <= cos^2(a|0>   - "
           + "c|0>)        + cos^2(b|0>     - c|1>)"
         + "\n****   cos^2((% 2.2f - % 2.4f)*pi) <= cos^2((% 2.2f - "
           + "% 2.4f)*pi) + cos^2((% 2.4f - % 2.4f)*pi)"
         + "\n****   <=>  %.4f <= %.4f + %.4f"
         + "\n****   <=>  %.4f <= %.4f"
         + "\n****        = %s"
         + "\n****    =>  %s"
         + "\n****")
            % (self.ALPHA_A0, self.ALPHA_B0, self.ALPHA_A0,
               self.ALPHA_C0, self.ALPHA_B0, self.ALPHA_C1,
               lhs_qul, rhs_qul[0], rhs_qul[1],
               lhs_qul, sum(rhs_qul), lhs_qul <= sum(rhs_qul),
               "Success!" if not lhs_qul <= sum(rhs_qul) else "FAILURE!"))

        # ---

        p_result = cases_counts/cases_runs

        lhs = p_result[0]
        rhs = [p_result[1], p_result[2]]
        print(
          ("**** Result: Bell's inequality in   Quantum Logic:"
         + "\n****        %4d/%4d <= %4d/%4d + %4d/%4d"
         + "\n****   <=>     %.4f <=    %.4f + %.4f"
         + "\n****   <=>     %.4f <= %.4f"
         + "\n****        = %s"
         + "\n****    =>  %s\n")
            % (cases_counts[0], cases_runs[0], cases_counts[1],
               cases_runs[1], cases_counts[2], cases_runs[2],
               lhs, rhs[0], rhs[1],
               lhs  , sum(rhs), lhs <= sum(rhs),
               "Success!" if not lhs <= sum(rhs) else "FAILURE!"))

# ********************************************************************

sim = MySim(N=2)

# ********************************************************************
# The quantum circuit to simulate.

circ = sim.init_new_circ()

circ.add_gate("SNOT", targets=sim.qindex(0))
circ.add_gate("CNOT", controls=sim.qindex(0), targets=sim.qindex(1))

sim.circalloced_load(circ)

# ********************************************************************
# Defining input of quantum circuit.

# Order of qubits for tensor product:
#
#   q_1 (cross) q_0 = [0 1 2 3]
#
sim.circloaded_set_input(
    qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
)

# ********************************************************************
# Run all file outputs, statistics and simulations.

sim.inputset_run_all(ol_runs=20000, pl_runs=500)

# ********************************************************************
