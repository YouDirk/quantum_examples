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
import qutip.qip.device as dv

sim = el.SimState(N=2)

# ********************************************************************
# The quantum circuit to simulate.

circ = sim.init_new_circ()

circ.add_gate("SNOT", targets=sim.qindex(0))
circ.add_gate("CNOT", controls=sim.qindex(0), targets=sim.qindex(1))

sim.circalloced_load(circ)

# ********************************************************************
# Save a visual representation of the quantum circuit as SVG.

sim.circloaded_save_svg()

# ********************************************************************
# Save the quantum circuit as Open Quantum Assembly Language (QASM).

sim.circloaded_save_qasm()

# ********************************************************************
# Defining input of quantum circuit.

# Order of qubits for tensor product:
#
#   q_1 (cross) q_0 = [0 1 2 3]
#
sim.circloaded_set_input(
    qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
)

# ********************************************************************
# Run statisitics for quantum circuit.

sim.inputset_statistics()

# ********************************************************************
# Run an 'operator-level' circuit simulation.

sim.inputset_run_ol(2000)

# ********************************************************************
# Setup a processor for 'pulse-level' circuit simulation.

# Set True to use a (realistic) ModelProcessor, otherwise an 'Optimal
# Control' will be used to predict optimal control pulses for the user
# defined Hamiltonians.
if True:
    processor = dv.LinearSpinChain(sim.N)
    #processor = dv.CircularSpinChain(sim.N) # Noise not working
    #processor = dv.DispersiveCavityQED(sim.N, num_levels=2) # ???
else:
    processor = dv.OptPulseProcessor(sim.N,
                                     drift=qt.tensor([qt.sigmaz()]*sim.N))
    processor.add_control(qt.sigmax(), cyclic_permutation=True)
    processor.add_control(qt.sigmay(), cyclic_permutation=True)
    processor.add_control(qt.tensor([qt.sigmay()]*sim.N),
                          cyclic_permutation=True)

noise = qt.qip.noise.RandomNoise(
        dt=0.01, rand_gen=np.random.normal, loc=0.00, scale=0.02)

sim.inputset_set_processor(processor, noise)

# ********************************************************************
# Plot pulses to SVG file.

sim.processorset_plot_pulses()

# ********************************************************************
# Run a 'pulse-level' circuit simulation.

sim.processorset_run_pl(250)

# ********************************************************************
