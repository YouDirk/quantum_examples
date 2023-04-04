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

# To simulate
#
# TODO: Comment visual ASCII art of measurement angles in Hilbert
#       space.
#
# Result:
#   Expectations are part of the simulation output.

# ********************************************************************
# Output result

class MySim (el.DefaultSim):
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
    DELTA_PHI   = 4/12

    def pre_measurement(self, custom_args: dict, sim_output: qt.Qobj,
                        measurement_ops: object) -> (qt.Qobj, object):
        custom_args['measurement_case'] = np.random.choice(range(3))

        match custom_args['measurement_case']:
            case 0:
                # O_sigmax(a_|0>, b_|0>) = O_sigmax(-1/4*pi, -1/12*pi)
                #
                # Probability in first-order logic (satisfy Bell's
                # inequality):
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
                phi    = [-np.pi,
                                   -self.DELTA_PHI * np.pi]
            case 1:
                # O_sigmax(a_|0>, c_|0>) = O_sigmax(-1/4*pi, 1/12*pi)
                #
                # Probability in first-order logic (satisfy Bell's
                # inequality):
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
                phi    = [-np.pi,
                                   self.DELTA_PHI * np.pi]
            case 2:
                # O_sigmax(b_|0>, c_|1>) = O_sigmax(-1/4*pi, 1/4*pi)
                #
                # Probability in first-order logic (satisfy Bell's
                # inequality):
                #   P(b_|0> AND c_|1>)
                #     = cos( -1/12*pi )**2 * sin(    1/12*pi     )**2
                #     = cos( -1/12*pi )**2 * cos( pi/2 - 1/12*pi )**2
                #     = cos( -1/12*pi )**2 * cos(    5/12*pi     )**2
                #     = 1/16
                #     = 0.0625
                # Probability in quantum logic (contradicts Bell's
                # inequality):
                #   P(b_|0>, c_|1>)
                #     = cos( (-1/12*pi) - (pi/2 + 1/12*pi) )**2
                #     = 1/4 = 0.25
                #
                # Consider: phi = 2*Theta = 4*alpha
                #               = 4*(pi/2) + DELTA_PHI
                phi    = [-self.DELTA_PHI * np.pi,
                                   4*(np.pi/2) + self.DELTA_PHI * np.pi]
            case _:
                raise AssertionError("Just 3 combinations are possible!")

        # Measuring first bit (bit 0)
        #o_kx = qt.tensor(qt.qeye(2), qo.ry(phi)*qt.sigmaz())
        #
        # Measuring second bit (bit 1)
        #o_xk = qt.tensor(qo.ry(phi)*qt.sigmaz(), qt.qeye(2))
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

        meas_o = qt.tensor(qt.sigmax() * qo.ry(phi[1]),
                           qt.sigmax() * qo.ry(phi[0]))

        return sim_output, meas_o

    # alpha = Theta/2 = phi/4
    # alpha: Hilbert space, Theta: Bloch sphere, phi: rotation operator
    DELTA_ALPHA            = DELTA_PHI/4
    DELTA_ENERGY_THRESHOLD = 1e-6
    def analyse_sim_result(self, sim_results: dict, sim_runs: int):
        energy_00 = 1

        cases_runs   = np.array([0, 0, 0])
        cases_counts = np.array([0, 0, 0])
        for count, energy, collapsed, custom_args in sim_results.values():
            cur_case = custom_args['measurement_case']

            cases_runs[cur_case] += count

            delta_energy = abs(energy - energy_00)
            if delta_energy > self.DELTA_ENERGY_THRESHOLD: continue

            cases_counts[cur_case] += count

        p_result = cases_counts/cases_runs

        print(
          ("\n**** For delta_alpha=%.4f*pi in Hilbert space"
           + " (delta_phi=%.4f*pi) measuring is"
         + "\n**** a|0> := alpha_a|0>                = -pi/4"
         + "\n**** b|0> := alpha_b|0>                = -%.4f*pi"
         + "\n**** c|0> := alpha_c|0>                =  %.4f*pi"
         + "\n**** c|1> := alpha_c|1> =  pi/2 + c|0> =  %.4f*pi"
         + "\n****")
            % (self.DELTA_ALPHA, self.DELTA_PHI,
               self.DELTA_ALPHA, self.DELTA_ALPHA,
               (1/2 + self.DELTA_ALPHA)))

        lhs_fol = \
          np.cos( -1/4*np.pi )**2 * np.cos( -self.DELTA_ALPHA*np.pi )**2
        rhs_fol = [
          np.cos( -1/4*np.pi )**2 * np.cos(  self.DELTA_ALPHA*np.pi )**2,
          np.cos( -self.DELTA_ALPHA*np.pi )**2
            * np.cos( np.pi/2 - self.DELTA_ALPHA*np.pi )**2 ]
        print(
          ("**** Expected: Bell's inequality in   First-Order Logic:"
         + "\n****   P(a|0> AND b|0>)               <= P(a|0> AND c|0>)"
           + "              + P(b|0> AND c|1>)"
         + "\n****   cos^2(a|0>) *cos^2(b|0>)       <= cos^2(a|0>)"
           + " *cos^2(c|0>)      + cos^2(b|0>)      *cos^2(c|1>)"
         + "\n****   cos^2(-pi/4)*cos^2(-%.4f*pi) <= cos^2(-pi/4)*"
           + "cos^2(%.4f*pi) + cos^2(-%.4f*pi)*cos^2(pi/2 + %.4f*pi)"
         + "\n****        %.4f <= %.4f + %.4f"
         + "\n****   <=>  %.4f <= %.4f"
         + "\n****        = %s"
         + "\n****    =>  %s"
         + "\n****")
            % (self.DELTA_ALPHA, self.DELTA_ALPHA, self.DELTA_ALPHA,
               self.DELTA_ALPHA,
               lhs_fol, rhs_fol[0], rhs_fol[1],
               lhs_fol, sum(rhs_fol), lhs_fol <= sum(rhs_fol),
               "Success!" if lhs_fol <= sum(rhs_fol) else "FAILURE!"))

        lhs_qul =  np.cos(
            (-1/4*np.pi) - (-self.DELTA_ALPHA*np.pi) )**2
        rhs_qul = [
          np.cos(   (-1/4*np.pi) - (self.DELTA_ALPHA*np.pi) )**2,
          np.cos(   (-self.DELTA_ALPHA*np.pi)
                  - (np.pi/2 + self.DELTA_ALPHA*np.pi)      )**2]
        print(
          ("**** Expected: Bell's inequality in   Quantum Logic:"
         + "\n****   P(a_|0>, b_|0>)             <= P(a_|0>, c_|0>)"
           + "          + P(b_|0>, c_|1>)"
         + "\n****   cos^2(a|0>  - b|0>)         <= cos^2(a|0>  - c|0>"
           + ")      + cos^2(b|0>       - c|1>)"
         + "\n****   cos^2(-pi/4 - (-%.4f*pi)) <= cos^2(-pi/4 - %.4f*pi)"
           + " + cos^2(-%.4f*pi - (pi/2 + %.4f*pi))"
         + "\n****   <=>  %.4f <= %.4f + %.4f"
         + "\n****   <=>  %.4f <= %.4f"
         + "\n****        = %s"
         + "\n****    =>  %s"
         + "\n****")
            % (self.DELTA_ALPHA, self.DELTA_ALPHA, self.DELTA_ALPHA,
               self.DELTA_ALPHA,
               lhs_qul, rhs_qul[0], rhs_qul[1],
               lhs_qul, sum(rhs_qul), lhs_qul <= sum(rhs_qul),
               "Success!" if not lhs_qul <= sum(rhs_qul) else "FAILURE!"))

        lhs = p_result[0]
        rhs = [p_result[1], p_result[2]]
        print(
          ("**** Result: Bell's inequality in   Quantum Logic:"
         + "\n****        %.4f <= %.4f + %.4f"
         + "\n****   <=>  %.4f <= %.4f"
         + "\n****        = %s"
         + "\n****    =>  %s\n")
            % (lhs, rhs[0], rhs[1],
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
