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

import qutip as qt
import qutip.qip.circuit as cc

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

# ********************************************************************
# User defined quantum gates.

def bellsate_gate():
    # TODO
    pass

# ********************************************************************
# The quantum circuit to simulate.

circ = cc.QubitCircuit(N, num_cbits=N)

circ.add_gate("SNOT", targets=qindex(0))
circ.add_gate("CNOT", controls=qindex(0), targets=qindex(1))

for i in range(N):
    circ.add_measurement("M" + str(i), targets=[qindex(i)],
                         classical_store=qindex(i))

print("\nQuantum-Circuit:\n%s\n%s"
      % (circ.gates, circ.propagators(expand=False)))

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

    print("\nSVG : Written to '%s'" % (svg_filename))
except Exception as e:
    print("\nSVG : Could not write '%s'! %s" % (svg_filename, str(e)))

# ********************************************************************
# Save the quantum circuit as Open Quantum Assembly Language (QASM).

import qutip.qip.qasm as qs

qasm_filename = os.path.splitext(sys.argv[0])[0] + '.qasm'

try:
    qs.save_qasm(circ, qasm_filename)

    print("QASM: Written to '%s'" % (qasm_filename))
except Exception as e:
    print("QASM: Could not write '%s'! %s" % (qasm_filename, str(e)))

# ********************************************************************
# Defining input of quantum circuit.

# Order of qubits for tensor product:
#
#   q_1 (cross) q_0 = [0 1 2 3]
#
input_sim = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))

print("\nInput: %s" % (input_sim.trans()))

# ********************************************************************
# Run statisitics for quantum circuit.

sim_ol = cc.CircuitSimulator(circ, precompute_unitary=True)

sim_ol_stat_result = sim_ol.run_statistics(input_sim)

sim_ol_stat_probs  = sim_ol_stat_result.get_probabilities()
sim_ol_stat_states = sim_ol_stat_result.get_final_states()

print("\nStatistics:")
for i in range(len(sim_ol_stat_probs)):
      print("Probability %s for %s"
            % (sim_ol_stat_probs[i], state2str(sim_ol_stat_states[i])))

# ********************************************************************
# Run an 'operator-level' circuit simulation.

sim_ol_N = 2000

def sim_ol_map(i: int):
    result = sim_ol.run(input_sim)

    measurment = result.get_final_states(0)
    hash_key = str(measurment)

    return hash_key, measurment

print("\nSimulation: Operator-Level (applying unitaries)")
map_result = qt.parallel.parallel_map(sim_ol_map, range(sim_ol_N),
                                      progress_bar=True)

results = {}
for hash_key, measurment in map_result:
    if not hash_key in results.keys():
        results[hash_key] = [measurment, 1]
    else: results[hash_key][1] += 1

for v in results.values():
    print("Periodicity %s for %s" % (v[1]/sim_ol_N, state2str(v[0])))

# ********************************************************************
# Run a 'pulse-level' circuit simulation.

sim_pl_N = 2000

print("\nSimulation: Pulse-Level"
      + " (open time evolution Hamiltonian solvers with noisiness)")
