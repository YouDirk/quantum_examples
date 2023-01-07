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
import qutip.qip.circuit as cc

# ********************************************************************

# '00': f(x) = 0, '11': f(x) = 1, '01': f(x) = x, '10': f(x) = NOT(x)
F_DEF = '10'

# To simulate
#
#              .------------------------.
# |1> ---|H|---|x_0       x_0 XOR f(x_1)|------------- x_0
#              |     U_f                |
# |0> ---|H|---|x_1                  x_1|---|H|---*--- x_1
#              '------------------------'         |
#                                              .-----.
#                                              | Mes |
#                                              '-----'
#
# Result:
#   x_1 = |0> (prob. 100%) if f(x) is constant (f(x)=0 or f(x)=1)
#   x_1 = |1> (prob. 100%) if f(x) is balanced (f(x)=x or f(x)=NOT(x))

# ********************************************************************
# Gates

def Uf(f_def: str) -> qt.Qobj:
    # f(x) = 0
    if   f_def == '00':
        # same as tensor(I, I)
        result = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

    # f(x) = 1
    elif f_def == '11':
        # same as tensor(I, sigma_X)
        result = [[0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]]

    # f(x) = x
    elif f_def == '01':
        # same as CNOT
        result = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]]

    # f(x) = NOT(x)
    elif f_def == '10':
        # same as CNOT * tensor(I, sigma_X)
        result = [[0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

    else:
        raise AssertionError("F_DEF has an invalid string value.")

    return qt.Qobj(result, dims=[[2, 2], [2, 2]])

# ********************************************************************
# Output result

def _bit_n_toint(n_shift: int, state: qt.Qobj) -> int:
    number = sum([i*int(state[i][0][0].real)
                  if state[i][0][0] != 0.0 else 0
                  for i in range(state.shape[0])])

    # ABS() required to prevent masking negative 2-complement numbers.
    return 1 if abs(number) & (1 << n_shift) else 0

def analyse_result(sim_results: dict):
    f_balanced = -1

    for state, count in sim_results.values():
        cur = _bit_n_toint(1, state)

        if (f_balanced < 0): f_balanced = cur

        if f_balanced != cur:
            raise AssertionError(
              "Error in Deutschs Algorithm: MSB x_1 not constant"
              + " with a periodicity of 100%!")

    if f_balanced == 1:
        print("\n**** Result: f(x) is balanced!")
    elif f_balanced == 0:
        print("\n**** Result: f(x) is constant!")
    else:
        raise AssertionError(
          "_BIT_N_TOINT() does not return a bit value!")

# ********************************************************************
# Output chosen f(x) via F_DEF

uf = Uf(F_DEF)

print("\nChosen: f(x) via F_DEF\n")
print("  f(|0>) = |%d>\n  f(|1>) = |%d>"
  % (_bit_n_toint(0, uf * qt.tensor(qt.basis(2, 0), qt.basis(2, 0))),
     _bit_n_toint(0, uf * qt.tensor(qt.basis(2, 1), qt.basis(2, 0)))))

# ********************************************************************
# The quantum circuit to simulate.

sim = el.DefaultSim(N=2, analyse_sim_result=analyse_result)

circ = sim.init_new_circ()

circ.user_gates = {'U_f': Uf}

circ.add_gate("SNOT", targets=sim.qindex(0))
circ.add_gate("SNOT", targets=sim.qindex(1))

circ.add_gate("U_f", targets=[sim.qindex(1), sim.qindex(0)],
              arg_value=F_DEF)

circ.add_gate("SNOT", targets=sim.qindex(1))

sim.circalloced_load(circ)

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
# Run all file outputs, statistics and simulations.

sim.circloaded_run_all(ol_runs=2000, pl_runs=250)

# ********************************************************************
