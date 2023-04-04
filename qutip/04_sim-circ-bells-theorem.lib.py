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
# Output result

class MySim (el.DefaultSim):
    # Rotation in Hilbert space (alpha) around Y axis in Bloch sphere
    # coordinates (Theta) is:
    #
    #   * phi = 2*Theta = 4*alpha
    #
    # Therefore, measuring to sigmax (means to |+> basis) is if
    #   phi = 2 * Theta      = 2 * (2 * alpha)
    #       = 2 * (1/2 * pi) = 2 * (2 * 1/4 * pi)
    #       = pi

    # DELTA_PHI = 2/3 pi            <=> delta_alpha = 1/6 pi = 2/12 pi
    DELTA_PHI   = 8/12

    def pre_measurement(self, custom_args: dict, sim_output: qt.Qobj,
                        measurement_ops: object) -> (qt.Qobj, object):
        custom_args['measurement_case'] = np.random.choice(range(3))

        match custom_args['measurement_case']:
            case 0:
                # O(0, 1/6*pi)      <=> P(a_|0> and b_|0>)
                #   = cos(-3/12*pi)**2 * cos(-1/12*pi)**2
                #   = (2+sqrt(2))/8
                #   = 0.4267766952966
                phi    = [0             , self.DELTA_PHI * np.pi]
                o_1bit = [qt.sigmaz()   , qt.sigmaz()]
            case 1:
                # O(0, 3/8*pi)      <=> P(a_|0> and c_|0>)
                #   = cos(-2/8*pi)**2 * cos(1/8*pi)**2
                #   = (2+sqrt(2))/8
                #   = 0.4267766952966
                phi    = [0             , 2*self.DELTA_PHI * np.pi]
                o_1bit = [qt.sigmaz()   , qt.sigmaz()]
            case 2:
                # O(1/8*pi, 3/8*pi) <=> P(b_|0> and c_|1>)
                phi    = [self.DELTA_PHI * np.pi,
                                          2*self.DELTA_PHI * np.pi]
                o_1bit = [qt.sigmaz()   , -qt.sigmaz()]
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

        meas_o = qt.tensor(o_1bit[1] * qo.ry(phi[1]),
                           o_1bit[0] * qo.ry(phi[0]))

        return sim_output, meas_o

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

        lhs = p_result[0]
        rhs = [p_result[1], p_result[2]]
        print(("\n**** Bell's Inequality %.4f <= %.4f + %.4f"
               + "\n****              <=>  %.4f <= %.4f"
               + "\n****                   = %s\n")
              % (lhs  , rhs[0], rhs[1],
                 lhs  , sum(rhs),
                 lhs <= sum(rhs)))

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
