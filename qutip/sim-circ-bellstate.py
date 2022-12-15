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


import sys, os, re

import numpy as np
#from matplotlib import pyplot as mp

import qutip as qt
import qutip.qip.circuit as cc
import qutip.qip.device as dv

# ********************************************************************
# Some utility functions.

# Number of qubits in quantum circuit.
N = 2

# Just an for assisitng printing output
def state2str(state: qt.Qobj) -> str:
    val = sum([i*int(state[i][0][0].real)
               if state[i][0][0] != 0.0 else 0
               for i in range(state.shape[0])])

    val_bin = ''; v = abs(val)
    for i in range(N):
        val_bin = ('1' if v & 0x1 else '0') + val_bin
        v >>= 1
    val_bin = ('-' if val < 0 else '') + '0b' + val_bin

    return "%s\n  = circuit data %s (binary %s)" \
             % (state.trans(), val, val_bin)

# The index of the qubits in QubitCircuit in the QuTiP library is in
# reversed order to it's logical meaning.  For example q_0 is indexed
# with targets=2 if N=3.
def qindex(i: int) -> int: return N - i - 1

def print_sim_map_result(map_result: list):
    results = {}
    for hash_key, state in map_result:
        if not hash_key in results.keys():
            results[hash_key] = [state, 1]
        else: results[hash_key][1] += 1

    for state, count in results.values():
        print("Periodicity %s for %s" % (count/len(map_result),
                                         state2str(state)))

# ********************************************************************
# User defined quantum gates.

def bellsate_gate():
    # TODO
    pass

# ********************************************************************
# The quantum circuit to simulate.

print("\nQuantum-Circuit:")

circ = cc.QubitCircuit(N, num_cbits=N)

circ.add_gate("SNOT", targets=qindex(0))
circ.add_gate("CNOT", controls=qindex(0), targets=qindex(1))

circ_measure_label = "M_all"
for i in range(N):
    circ.add_measurement(circ_measure_label, targets=qindex(i),
                         classical_store=qindex(i))

print("%s\n%s" % (circ.gates, circ.propagators(expand=False)))

# ********************************************************************
# Save a visual representation of the quantum circuit as SVG.

svg_filename = os.path.splitext(sys.argv[0])[0] + '.svg'
svg_xml = circ._raw_svg()

# just a simple rescaling by injecting better units pt -> mm
svg_xml = re.sub( r'(width="[0-9]+)[a-z]+(")', r'\1mm\2', svg_xml)
svg_xml = re.sub(r'(height="[0-9]+)[a-z]+(")', r'\1mm\2', svg_xml)

try:
    svg_file = open(svg_filename, mode='w')
    svg_file.write(svg_xml)
    svg_file.close()

    print("\nSVG : circuit written to '%s'" % (svg_filename))
except Exception as e:
    print("\nSVG : Could not write '%s'! %s" % (svg_filename, str(e)))

# ********************************************************************
# Save the quantum circuit as Open Quantum Assembly Language (QASM).

import qutip.qip.qasm as qs

qasm_filename = os.path.splitext(sys.argv[0])[0] + '.qasm'

try:
    qs.save_qasm(circ, qasm_filename)

    print("QASM: circuit written to '%s'" % (qasm_filename))
except Exception as e:
    print("QASM: Could not write '%s'! %s" % (qasm_filename, str(e)))

# ********************************************************************
# Defining input of quantum circuit.

# Order of qubits for tensor product:
#
#   q_1 (cross) q_0 = [0 1 2 3]
#
input_sim = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))

print("\nInput:\n%s" % state2str(input_sim))

# ********************************************************************
# Run statisitics for quantum circuit.

print("\nStatistics:")

sim_ol = cc.CircuitSimulator(circ, precompute_unitary=True)

sim_ol_stat_result = sim_ol.run_statistics(input_sim)

sim_ol_stat_probs  = sim_ol_stat_result.get_probabilities()
sim_ol_stat_states = sim_ol_stat_result.get_final_states()

for i in range(len(sim_ol_stat_probs)):
      print("Probability %s for %s"
            % (sim_ol_stat_probs[i], state2str(sim_ol_stat_states[i])))

# ********************************************************************
# Run an 'operator-level' circuit simulation.

print("\nSimulation: Operator-Level (applying unitaries)")

sim_ol_N = 2000

def sim_ol_map(i: int):
    result = sim_ol.run(input_sim)

    measurement = result.get_final_states(0)
    hash_key = str(measurement)

    return hash_key, measurement

sim_ol_map_result = qt.parallel.parallel_map(sim_ol_map, range(sim_ol_N),
                                             progress_bar=True)

print_sim_map_result(sim_ol_map_result)

# ********************************************************************
# Run a 'pulse-level' circuit simulation.

# Measurements in circuit seems not to be supported for pulse-level
# simulation.  We are measuring manually at the end of every
# simulation.
circ.remove_gate_or_measurement(name=circ_measure_label, remove='all')

# Set True to use a (realistic) ModelProcessor, otherwise an 'Optimal
# Control' will be used to predict optimal control pulses for the user
# defined Hamiltonians.
if True:
    load_circuit_args = {}

    sim_pl_processor = dv.LinearSpinChain(N)
    #sim_pl_processor = dv.CircularSpinChain(N) # Noise not working
    #sim_pl_processor = dv.DispersiveCavityQED(N, num_levels=2) # ???
else:
    tslots = 10
    load_circuit_args = {'num_tslots': tslots, 'evo_time': tslots}

    sim_pl_processor = dv.OptPulseProcessor(N,
                                 drift=qt.tensor([qt.sigmaz()]*N))
    sim_pl_processor.add_control(qt.sigmax(), cyclic_permutation=True)
    sim_pl_processor.add_control(qt.sigmay(), cyclic_permutation=True)
    sim_pl_processor.add_control(qt.tensor([qt.sigmay()]*N),
                                 cyclic_permutation=True)

sim_pl_procname = sim_pl_processor.__class__.__name__
print("\nSimulation: Pulse-Level (using %s processor with noise)"
      % (sim_pl_procname))

sim_pl_N = 250

sim_pl_processor.pulse_mode = "discrete"
sim_pl_processor.load_circuit(circ, **load_circuit_args)

noise = qt.qip.noise.RandomNoise(
        dt=0.01, rand_gen=np.random.normal, loc=0.00, scale=0.02)
sim_pl_processor.add_noise(noise)

# --- pulse plot ---
sim_pl_fig, sim_pl_axis = sim_pl_processor.plot_pulses()

# Add noise to plot
sim_pl_noisy_qobjevo, _ = sim_pl_processor.get_qobjevo(noisy=True)
sim_pl_noisy_pulse = sim_pl_noisy_qobjevo.to_list()
for i in range(1, len(sim_pl_noisy_pulse), 2):
    noisy_coeff = sim_pl_noisy_pulse[i][1] + sim_pl_noisy_pulse[i+1][1]
    sim_pl_axis[i//2].step(sim_pl_noisy_qobjevo.tlist, noisy_coeff)

sim_pl_filename = "%s-pulse-%s.svg" % (
                  os.path.splitext(sys.argv[0])[0], sim_pl_procname)
try:
    sim_pl_fig.savefig(sim_pl_filename, format='svg', transparent=True)

    print("SVG       : pulses plotted to '%s'" % (sim_pl_filename))
except Exception as e:
    print("SVG : Could not write '%s'! %s" % (sim_pl_filename, str(e)))
# --- end of pulse plot ---

def sim_pl_map(i: int):
    result = sim_pl_processor.run_state(input_sim)

    _, measurement = qt.measurement.measure(result.states[-1],
                                            qt.tensor([qt.sigmaz()]*N))
    hash_key = str(measurement)
    return hash_key, measurement

sim_pl_map_result = qt.parallel.parallel_map(sim_pl_map, range(sim_pl_N),
                                             progress_bar=True)

print_sim_map_result(sim_pl_map_result)

# ********************************************************************
