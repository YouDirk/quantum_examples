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

import qutip as qt

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

class MySim (el.DefaultSim):
    def analyse_sim_result(self, sim_results: dict, sim_runs: int):
        print(
          "\n**** The MSB x_1 is to be interpreted as 'f(|0>) XOR"
          + " f(|1>)'.  That means, it is |0> if f(x)"
          + "\n**** is constant; otherwise it is |1> if f(x) is depending"
          + " on x (called 'balanced')"
          + "\n****")

        f_balanced = -1

        for state, count in sim_results.values():
            cur = MySim.state_tobit(state, 1)

            if f_balanced < 0: f_balanced = cur

            if f_balanced != cur:
                raise AssertionError(
                  "Error in Deutschs Algorithm: MSB x_1 not constant"
                  + " with a periodicity of 100%!")

        if f_balanced == 1:
            print("**** Result: f(x) is balanced!")
        elif f_balanced == 0:
            print("**** Result: f(x) is constant!")
        else:
            raise AssertionError(
              "STATE_TOBIT() does not return a bit value!")

# ********************************************************************

sim = MySim(N=2)

# ********************************************************************
# Output chosen f(x) via F_DEF

uf = Uf(F_DEF)

print("\nChosen: f(x) via F_DEF\n")
print("  f(|0>) = |%d>\n  f(|1>) = |%d>"
  % (MySim.state_tobit(uf * MySim.basis_fromint(2, 0x0), 0),
     MySim.state_tobit(uf * MySim.basis_fromint(2, 0x2), 0)
    ))

# ********************************************************************
# The quantum circuit to simulate.

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
sim.circloaded_set_input(MySim.basis_fromint(2, 0x1))

# ********************************************************************
# Run all file outputs, statistics and simulations.

sim.inputset_run_all(ol_runs=2000, pl_runs=250)

# ********************************************************************
