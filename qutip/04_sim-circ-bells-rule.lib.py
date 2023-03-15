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
    def analyse_sim_result(self, sim_results: dict, sim_runs: int):
        print("\n**** TODO !\n")

        for state, count in sim_results.values(): pass

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

# TODO ...
#
# Here measuring to |+> basis, means a rotation of sigma_z around
# y-axis with Theta=pi/2
theta = 1/2 * 2*np.pi
o = qt.tensor([qo.ry(theta) * qt.sigmaz()]*sim.N)
sim.init_set_measurement_ops(o)

sim.inputset_run_all(ol_runs=2000, pl_runs=250)

# ********************************************************************
