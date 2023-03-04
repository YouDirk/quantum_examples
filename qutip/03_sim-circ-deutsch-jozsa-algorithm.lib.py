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

# ********************************************************************

# Example: '10011010':
#   f(|0>) = |1>, f(|1>) = |0>, f(|2>) = |0>, f(|3>) = |1>,
#   f(|4>) = |1>, f(|5>) = |0>, f(|6>) = |1>, f(|7>) = |0>
#
# LEN(F_DEF) must be a power of 2 (2, 4, 8, 16)!
F_DEF = '0110100110010110'

# To simulate
#
#                .------------------------.
# |1> ---|H|-----|y             y XOR f(x)|--------------- y_out
#                |     U_f                |
# |0> ---|H|^n---|x^n                  x^n|---|H|^n---*--- x_out^n
#                '------------------------'           |
#                                                  .-----.
#                                                  | Mes |
#                                                  '-----'
#
# Result:
#   x_out = |0>    (prob. 100%)
#     if f(x) is constant (f(x)=0 or f(x)=1)
#   x_out = const  (prob. 100%)
#     if f(x) is balanced (means every split in half of bits has the
#                          same number of |1>'s on each side)
#   x_out = toggles
#     if f(x) is neither constant nor balanced

# ********************************************************************

# Number of bits in |x>
N = np.log2(len(F_DEF))
if N != np.ceil(N):
    raise AssertionError("F_DEF: Length is not a power of 2!  Means"
      + " domain of f(x) is not a power of 2 (2, 4, 8, 16)!")
else:
    N = int(N)

# ********************************************************************
# Gates

def Uf(f_def: str) -> qt.Qobj:
    n_domain = len(f_def)
    n = int(np.log2(n_domain))

    result = np.zeros([2*n_domain, 2*n_domain])
    for i in range(n_domain):
        bit = f_def[i]

        if bit == '0':
            result[2*i    ][2*i    ] = 1
            result[2*i + 1][2*i + 1] = 1
        elif bit == '1':
            result[2*i    ][2*i + 1] = 1
            result[2*i + 1][2*i    ] = 1
        else:
            raise AssertionError(
              "F_DEF: '%s' is not a valid bit value!" % (bit))

    return qt.Qobj(result,
                   dims=[[2]*(n+1), [2]*(n+1)])

# ********************************************************************
# Output result

class MySim (el.NoNoiseSim):
    def analyse_sim_result(self, sim_results: dict, sim_runs: int):
        print(
          "\n**** The output x_out is to be interpreted as:"
          + "\n****   - f(x) is balanced if x_out != |0000>"
          + "\n****   - f(x) is constant if x_out == |0000>"
          + "\n****   -             else if x_out is toggling."
          + "\n****")

        if len(sim_results) < 2:
            raise AssertionError(
              "Error in Deutsch-Jozsa algorithm: At least the LSB bit"
              + " y_out should toggle.  Otherwise it is possible that"
              + " OL_RUNS or PL_RUNS is set too low!")

        if len(sim_results) > 2:
            print("**** Result: f(x) is neither balanced nor constant!"
              + "  Number of measured results: %d" % len(sim_results))
            return

        f_balanced = -1
        for state, count in sim_results.values():
            number = MySim.state_toint(state)
            x = abs(number) >> 1

            if f_balanced < 0: f_balanced = x

            if f_balanced != x:
                raise AssertionError(
                  "Error in Deutsch-Jozsa algorithm: The output x_out"
                  + " toggles and is not constant, but just 2 states"
                  + " where measured!")

        if f_balanced > 0:
            print("**** Result: f(x) is balanced with x_out = |%d>!"
                  % (x))
        elif f_balanced == 0:
            print("**** Result: f(x) is constant!")
        else:
            raise AssertionError("Negative value!")

# ********************************************************************

sim = MySim(N=N+1)

# ********************************************************************
# Output chosen f(x) via F_DEF

uf = Uf(F_DEF)

print("\nChosen: f(x) via F_DEF\n")
for i in range(2**N):
    x = MySim.basis_fromint(N, i)
    f = uf * qt.tensor(x, qt.basis(2, 0))

    print("  f(|%d>) = |%s>" % (i, MySim.state_tobit(f, 0)))

# ********************************************************************
# The quantum circuit to simulate.

circ = sim.init_new_circ()

circ.user_gates = {'U_f': Uf}

for i in range(N+1):
    circ.add_gate("SNOT", targets=sim.qindex(i))

circ.add_gate("U_f", arg_value=F_DEF)

for i in range(1, N+1):
    circ.add_gate("SNOT", targets=sim.qindex(i))

sim.circalloced_load(circ)

# ********************************************************************
# Defining input of quantum circuit.

# Order of qubits for tensor product:
#
#   q_1 (cross) q_0 = [0 1 2 3]
#
sim.circloaded_set_input(MySim.basis_fromint(N+1, 0x1))

# ********************************************************************
# Run all file outputs, statistics and simulations.

sim.inputset_run_all(ol_runs=2000, pl_runs=250)

# ********************************************************************
