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


__all__ = ['SimState']

import sys, os, re, copy

import enum

import qutip as qt
import qutip.qip.circuit as cc
import qutip.qip.qasm as qs
import qutip.parallel as pa
import qutip.qip.device as dv
import qutip.qip.noise as ns

# ********************************************************************
# private stuff

class _state (enum.Flag):
    INITIALIZED = enum.auto()
    CIRC_ALLOCED = enum.auto()
    CIRC_LOADED = enum.auto()
    INPUT_SET = enum.auto()
    PROCESSOR_SET = enum.auto()

    def __str__(self) -> str:
        return "%s(%d)" % (super().__str__().split('.')[-1], self.value)

# ********************************************************************

class SimState:
    # ----------------------------------------------------------------
    # constants and some utility functions

    CIRC_MEASURE_LABEL = "M_all"

    SVG_FILENAME  = os.path.splitext(sys.argv[0])[0] + '.svg'
    QASM_FILENAME = os.path.splitext(sys.argv[0])[0] + '.qasm'

    def qindex(self, i: int) -> int:
        return self.N - i - 1

    def cindex(self, i: int) -> int:
        return self.cbits_N - i - 1

    # ----------------------------------------------------------------

    def __init__(self, N: int, cbits_N: int = -1):
        self.N = N
        self.cbits_N = cbits_N if cbits_N >= 0 else N

        self.state = _state.INITIALIZED

    def __repr__(self) -> str:
        return "%s: N=%d, cbits_N=%d" \
          % (self.state, self.N, self.cbits_N)

    def _assert(self, state):
        if self.state not in state:
            raise AssertionError(
              "State %s expected, but in state %s!\n  %s"
              % (state, self.state, self))

    def _assert_gr_equal(self, state):
        if self.state.value < state.value:
            raise AssertionError(
              "At least state %s expected, but current %s is lower!\n  %s"
              % (state, self.state, self))

    # Just an for assisitng printing output
    def _state2str(self, state: qt.Qobj) -> str:
        val = sum([i*int(state[i][0][0].real)
                   if state[i][0][0] != 0.0 else 0
                   for i in range(state.shape[0])])

        val_bin = ''; v = abs(val)
        for i in range(self.N):
            val_bin = ('1' if v & 0x1 else '0') + val_bin
            v >>= 1
        val_bin = ('-' if val < 0 else '') + '0b' + val_bin

        return "%s\n  = circuit data %s (binary %s)" \
               % (state.trans(), val, val_bin)

    def _print_map_result(self, map_result: list):
        results = {}
        for hash_key, state in map_result:
            if not hash_key in results.keys():
                results[hash_key] = [state, 1]
            else: results[hash_key][1] += 1

        for state, count in results.values():
            print("Periodicity %s for %s" % (count/len(map_result),
                                             self._state2str(state)))

    # ----------------------------------------------------------------
    # for state: INITIALIZED

    def init_new_circ(self) -> cc.QubitCircuit:
        print("\nQuantum-Circuit:")
        self._assert(_state.INITIALIZED)

        result = cc.QubitCircuit(self.N, num_cbits=self.cbits_N)

        self.state = _state.CIRC_ALLOCED
        return result

    # ----------------------------------------------------------------
    # for state: CIRC_ALLOCED

    def circalloced_load(self, circ: cc.QubitCircuit):
        self._assert(_state.CIRC_ALLOCED)

        self.circ = circ

        for i in range(self.cbits_N):
            circ.add_measurement(
              self.CIRC_MEASURE_LABEL, targets=self.qindex(i),
                classical_store=self.cindex(i))

        print("%s\n%s"
              % (self.circ.gates, self.circ.propagators(expand=False)))

        self.state = _state.CIRC_LOADED

    # ----------------------------------------------------------------
    # for state: CIRC_LOADED

    def circloaded_save_svg(self):
        self._assert_gr_equal(_state.CIRC_LOADED)

        xml = self.circ._raw_svg()

        # just a simple rescaling by injecting better units pt -> mm
        xml = re.sub( r'(width="[0-9]+)[a-z]+(")', r'\1mm\2', xml)
        xml = re.sub(r'(height="[0-9]+)[a-z]+(")', r'\1mm\2', xml)

        try:
            svg_file = open(self.SVG_FILENAME, mode='w')
            svg_file.write(xml)
            svg_file.close()

            print("\nSVG : circuit written to '%s'"
                  % (self.SVG_FILENAME))
        except Exception as e:
            print("\nSVG : Could not write '%s'! %s"
                  % (svg_filename, str(e)))

    def circloaded_save_qasm(self):
        self._assert_gr_equal(_state.CIRC_LOADED)

        try:
            qs.save_qasm(self.circ, self.QASM_FILENAME)

            print("QASM: circuit written to '%s'"
                  % (self.QASM_FILENAME))
        except Exception as e:
            print("QASM: Could not write '%s'! %s"
                  % (self.QASM_FILENAME, str(e)))

    def circloaded_set_input(self, input: qt.Qobj):
        print("\nInput:")
        self._assert(_state.CIRC_LOADED)

        self.input = input

        print(self._state2str(input))
        self.state = _state.INPUT_SET

    # ----------------------------------------------------------------
    # for state: INPUT_SET

    def inputset_statistics(self, **sim_args):
        sim_args.setdefault('precompute_unitary', True)

        print("\nStatistics:")
        self._assert_gr_equal(_state.INPUT_SET)

        sim = cc.CircuitSimulator(self.circ, **sim_args)

        stat_result = sim.run_statistics(self.input)

        stat_probs  = stat_result.get_probabilities()
        stat_states = stat_result.get_final_states()

        for i in range(len(stat_probs)):
            print("Probability %s for %s"
                  % (stat_probs[i], self._state2str(stat_states[i])))

    # ***

    def _inputset_run_ol_map(self, i: int, sim: cc.CircuitSimulator):
        result = sim.run(self.input)

        measurement = result.get_final_states(0)
        hash_key = str(measurement)

        return hash_key, measurement

    def inputset_run_ol(self, N: int, **sim_args):
        sim_args.setdefault('precompute_unitary', True)

        print("\nSimulation: Operator-Level (applying unitaries)")
        self._assert_gr_equal(_state.INPUT_SET)

        sim = cc.CircuitSimulator(self.circ, **sim_args)

        map_result = pa.parallel_map(self._inputset_run_ol_map,
          range(N), task_args=(sim,), progress_bar=True)

        self._print_map_result(map_result)

    # ***

    def inputset_set_processor(self, processor: dv.Processor,
                               noise: ns.Noise = None):
        procname = processor.__class__.__name__
        noisename = noise.__class__.__name__ if noise else 'no-noise'
        print("\nProcessor: %s, using %s" % (procname, noisename))

        self._assert_gr_equal(_state.INPUT_SET)

        if isinstance(processor, dv.ModelProcessor):
            load_circuit_args = {}
        else:
            tslots = 10
            load_circuit_args = {'num_tslots': tslots, 'evo_time': tslots}

        # Measurements in circuit seems not to be supported for
        # pulse-level simulation.  We are measuring manually at the
        # end of every simulation.
        circ = copy.deepcopy(self.circ)
        circ.remove_gate_or_measurement(name=self.CIRC_MEASURE_LABEL,
                                        remove='all')

        processor.pulse_mode = "discrete"
        processor.load_circuit(circ, **load_circuit_args)

        if noise: processor.add_noise(noise)

        self.processor = processor
        self.state = _state.PROCESSOR_SET

    # ----------------------------------------------------------------
    # for state: PROCESSOR_SET

    def processorset_plot_pulses(self):
        #TODO
        pass

    def processorset_run_pl(self, N: int):
        print("\nSimulation: Pulse-Level")
        self._assert_gr_equal(_state.PROCESSOR_SET)

        #TODO

# end of class SimState
# ********************************************************************
