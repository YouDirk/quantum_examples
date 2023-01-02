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

import sys, os, re

import numpy as np
#from matplotlib import pyplot as mp

import qutip as qt
import qutip.qip.circuit as cc
import qutip.qip.device as dv

sim = el.SimState(N=2)

# ********************************************************************
# Some utility functions.

# Number of qubits in quantum circuit.
N = 2

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
    processor = dv.LinearSpinChain(N)
    #processor = dv.CircularSpinChain(N) # Noise not working
    #processor = dv.DispersiveCavityQED(N, num_levels=2) # ???
else:
    processor = dv.OptPulseProcessor(N,
                                     drift=qt.tensor([qt.sigmaz()]*N))
    processor.add_control(qt.sigmax(), cyclic_permutation=True)
    processor.add_control(qt.sigmay(), cyclic_permutation=True)
    processor.add_control(qt.tensor([qt.sigmay()]*N),
                          cyclic_permutation=True)

noise = qt.qip.noise.RandomNoise(
        dt=0.01, rand_gen=np.random.normal, loc=0.00, scale=0.02)

sim.inputset_set_processor(dv.LinearSpinChain(sim.N), noise)

# ********************************************************************
# Plot pulses to SVG file.

sim.processorset_plot_pulses()

##
## TODO
##

## ********************************************************************
## Run a 'pulse-level' circuit simulation.
#
## --- pulse plot ---
#sim_pl_fig, sim_pl_axis = sim_pl_processor.plot_pulses()
#
## Add noise to plot
#sim_pl_noisy_qobjevo, _ = sim_pl_processor.get_qobjevo(noisy=True)
#sim_pl_noisy_pulse = sim_pl_noisy_qobjevo.to_list()
#for i in range(1, len(sim_pl_noisy_pulse), 2):
#    noisy_coeff = sim_pl_noisy_pulse[i][1] + sim_pl_noisy_pulse[i+1][1]
#    sim_pl_axis[i//2].step(sim_pl_noisy_qobjevo.tlist, noisy_coeff)
#
#sim_pl_filename = "%s-pulse-%s.svg" % (
#                  os.path.splitext(sys.argv[0])[0], sim_pl_procname)
#try:
#    sim_pl_fig.savefig(sim_pl_filename, format='svg', transparent=True)
#
#    print("SVG       : pulses plotted to '%s'" % (sim_pl_filename))
#except Exception as e:
#    print("SVG : Could not write '%s'! %s" % (sim_pl_filename, str(e)))
## --- end of pulse plot ---
#
#def sim_pl_map(i: int):
#    result = sim_pl_processor.run_state(input_sim)
#
#    _, measurement = qt.measurement.measure(result.states[-1],
#                                            qt.tensor([qt.sigmaz()]*N))
#    hash_key = str(measurement)
#    return hash_key, measurement
#
#sim_pl_map_result = qt.parallel.parallel_map(sim_pl_map, range(sim_pl_N),
#                                             progress_bar=True)
#
#print_sim_map_result(sim_pl_map_result)

# ********************************************************************
