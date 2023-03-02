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

N = 2

# ********************************************************************
# Input

circ_input = el.SimState.basis_fromint(N, 0x3)

# ********************************************************************
# Output result

class MySim (el.NoNoiseSim):
    CIRC_AS_OP = qt.tensor([*[qt.qeye(2)]*(N-1), qo.snot()])

    # ----------------------------------------------------------------

    def meas_proj_projector(self, bases: list) -> qt.Qobj:
        for v in bases:
            if not v.isket:
                raise AssertionError("BASIS is not a ket vector!")
            if v.shape[0] != 2**self.N:
                raise AssertionError(
                    "LEN(BASIS) needs to be %d, but is %d!"
                    % (2**self.N, v.shape[0]))

        # P_m = sum ( |bases><bases| )
        return sum([bases[i].proj() for i in range(len(bases))])

    # ----------------------------------------------------------------

    def meas_povm_completeness_check(self, m: list):
        completeness = sum([_m.dag() * _m for _m in m])

        # sum(M_m.dag() * M_m) = I
        if completeness != qt.qeye([2]*self.N):
            raise AssertionError(
                "M_m are not satisfying the completeness equation!")

    def meas_povm_e(self, m: list, eigenv: list) -> list:
        if len(m) != len(eigenv):
            raise AssertionError(
                "LEN(M) and LEN(EIGENV) are not the same!")

        self.meas_povm_completeness_check(m)

        # E_m = eigenv_m * M_m.dagger * M_m
        return [eigenv[i] * m[i].dag() * m[i]
                for i in range(len(eigenv))]

    # ----------------------------------------------------------------

    def meas_obs_o(self, e: list) -> qt.Qobj:
        # O = sum( E_m )
        return sum(e)

    # ----------------------------------------------------------------

    def measure_probs(self, state: qt.Qobj, o: qt.Qobj) -> \
                                                   (list, list, list):
        if not state.isket:
            raise AssertionError("STATE is not a ket vector!")

        # Because
        #   E_m = eigenv_m * M_m.dagger * M_m
        #   O = sum( E_m )
        #
        # => The Eigendecomposition of O is
        #   O = sum( eigenv_m * Mdag_M ) = sum( energy_m * Mdag_M )
        l, v = o.eigenstates()
        povm_vals = {}
        for i in range(len(l)):
            if l[i] == .0: continue

            hash_key = str(l[i])

            if not hash_key in povm_vals.keys():
                povm_vals[hash_key] = [l[i], v[i].proj()]
            else: povm_vals[hash_key][1] += v[i].proj()

        mdag_m = [m for _, m in povm_vals.values()]
        energies = [v for v, _ in povm_vals.values()]

        # sum(M_m.dag() * M_m) = I
        self.meas_povm_completeness_check(mdag_m)

        result_probs = []
        result_state_collapsed = []
        result_energies = []
        for i in range(len(energies)):
            # p(m) = <state| M_m.dag()*M_m |state>
            prob = abs((state.dag() * mdag_m[i] * state)[0][0][0])

            # ignore measurement operators with p=0.0
            if prob == .0: continue

            # state_collapsed = M_m |state> / sqrt(p(m))
            state_collapsed = mdag_m[i] * state / np.sqrt(prob)

            result_probs.append(prob)
            result_state_collapsed.append(state_collapsed)
            result_energies.append(energies[i])

        return result_probs, result_state_collapsed, result_energies

    def measure(self, state: qt.Qobj, o: qt.Qobj) -> (float, qt.Qobj):
        if not state.isket:
            raise AssertionError("STATE is not a ket vector!")

        probs, states_collapsed, energies = self.measure_probs(state, o)

        # select one of the possible states with probability PROBS
        i = np.random.choice(range(len(probs)), p=probs)

        return energies[i], states_collapsed[i]

    # ----------------------------------------------------------------

    def analyse_sim_result(self, sim_results: dict):
        psi_out = self.CIRC_AS_OP * self.input

        # ---
        # Projective measurement of bit 0.  Measuring
        #   * energy -1 if first bit is |0>
        #   * energy  1 if first bit is |1>

        P0_x0 = self.meas_proj_projector(
            [MySim.basis_fromint(self.N, 0),
             MySim.basis_fromint(self.N, 2)])
        P0_x1 = self.meas_proj_projector(
            [MySim.basis_fromint(self.N, 1),
             MySim.basis_fromint(self.N, 3)])
        P0 = [P0_x0, P0_x1]

        energies0 = [-1., 1.]

        # Projective measurement is just a special case of POVM
        # measurement.
        E0 = self.meas_povm_e(P0, energies0)
        # POVM measurement is just a special case of Obs measurement.
        O0 = self.meas_obs_o(E0)

        # ---
        # Projective measurement of bit 1.  Measuring
        #   * energy -1 if second bit is |0>
        #   * energy  1 if second bit is |1>

        P1_0x = self.meas_proj_projector(
            [MySim.basis_fromint(self.N, 0),
             MySim.basis_fromint(self.N, 1)])
        P1_1x = self.meas_proj_projector(
            [MySim.basis_fromint(self.N, 2),
             MySim.basis_fromint(self.N, 3)])
        P1 = [P1_0x, P1_1x]

        energies1 = [-1., 1.]

        # Projective measurement is just a special case of POVM
        # measurement.
        E1 = self.meas_povm_e(P1, energies1)
        # POVM measurement is just a special case of Obs measurement.
        O1 = self.meas_obs_o(E1)

        # ---

        probs, coll, energ = [None]*2, [None]*2, [None]*2
        probs[0], coll[0], energ[0] = self.measure_probs(psi_out, O0)
        probs[1], coll[1], energ[1] = self.measure_probs(psi_out, O1)

        energy, psi_coll = [None]*4, [None]*4
        energy[0], psi_coll[0] = self.measure(psi_out,     O0)
        energy[1], psi_coll[1] = self.measure(psi_coll[0], O1)
        # The first measurement lets collapse PSI.  The following
        # measurements have the same results.
        energy[2], psi_coll[2] = self.measure(psi_coll[1], O0)
        energy[3], psi_coll[3] = self.measure(psi_coll[2], O1)

        print("")
        for j in range(len(probs)):
            print("**** O%d: measurement statistics, self implemented"
                  % (j))
            for i in range(len(probs[j])):
                print("****   energy % 2f with p=%f, collapsed=%s"
                      % (energ[j][i], probs[j][i],
                         list(coll[j][i].trans()[0][0])))

        print("**** measurement: ordered O0 -> O1 -> O0 -> O1")
        print("****   |psi_out> = %s" % (list(psi_out.trans()[0][0])))
        for j in range(len(energy)):
            print("****   energy % 2f, collapsed=%s"
                  % (energy[j], list(psi_coll[j].trans()[0][0])))
        print("****   |psi_measured> = %s = |%d>\n"
              % (list(psi_coll[-1].trans()[0][0]),
                 MySim.state_toint(psi_coll[-1])))

        # nothing to do here ...
        for state, count in sim_results.values(): pass

# ********************************************************************

sim = MySim(N=N)

# ********************************************************************
# The quantum circuit to simulate.

circ = sim.init_new_circ()

circ.add_gate("SNOT", targets=sim.qindex(0))

sim.circalloced_load(circ)

# ********************************************************************
# Defining input of quantum circuit.

# Order of qubits for tensor product:
#
#   q_1 (cross) q_0 = [0 1 2 3]
#
sim.circloaded_set_input(circ_input)

# ********************************************************************
# Run all file outputs, statistics and simulations.

sim.inputset_run_all(ol_runs=2000, pl_runs=250)

# ********************************************************************
