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

from .StatePlotter import *

import sys, os, re, copy

import enum

import numpy as np

import qutip as qt
import qutip.qip.circuit as cc
import qutip.qip.qasm as qs
import qutip.parallel as pa
import qutip.qip.device as dv
import qutip.qip.noise as ns

# ********************************************************************

class SimState:
    # ----------------------------------------------------------------
    # subclasses can override this

    # Not thread safe, do not access member variables!  Just use the
    # method parameters and local variables!
    #
    # Called in every simulation instance before simulation starts.
    #
    # SIM_INPUT: The predefined input of the circuit during
    #            simulation.
    #
    # RETURN: Your custom ciruit input.
    def pre_simulation(self, custom_args: dict, sim_input: qt.Qobj) \
                       -> qt.Qobj:
        return sim_input

    # Not thread safe, do not access member variables!  Just use the
    # method parameters and local variables!
    #
    # Called in every simulation instance after simulation finished
    # and before measurement starts.
    #
    # SIM_OUTPUT: The predefined output of the circuit during
    #             simulation.
    # MEASUREMENT_OPS: Either or:
    #                  * an Obs of type qt.Qobj
    #                  * list of Measurement Operators
    #
    # RETURN: Your custom ciruit (SIM_OUTPUT, MEASUREMENT_OPS).
    def pre_measurement(self, custom_args: dict, sim_output: qt.Qobj,
                        measurement_ops: object) -> (qt.Qobj, object):
        return sim_output, measurement_ops

    # Function for additional output after simulation.  The
    # SIM_RESULTS argument has the type:
    #
    #   SIM_RESULTS['hash_key']
    #       = [energy: int, collapsed: qt.Qobj, count: int]
    def analyse_sim_result(self, sim_results: dict, sim_runs: int):
        pass

    # ----------------------------------------------------------------
    # private stuff

    class _state (enum.Flag):
        INITIALIZED = enum.auto()
        CIRC_ALLOCED = enum.auto()
        CIRC_LOADED = enum.auto()
        INPUT_SET = enum.auto()
        PROCESSOR_SET = enum.auto()

        def __str__(self) -> str:
            return "%s(%d)" \
                   % (super().__str__().split('.')[-1], self.value)

    # ----------------------------------------------------------------
    # static utility functions

    def state_toint(state: qt.Qobj) -> int:
        return sum([i*int(state[i][0][0].real)
                    if state[i][0][0] != 0.0 else 0
                    for i in range(state.shape[0])])

    def state_tobit(state: qt.Qobj, n_shift: int) -> int:
        number = SimState.state_toint(state)

        # ABS() required to prevent masking negative 2-complements
        # numbers.
        return 1 if abs(number) & (1 << n_shift) else 0

    def int_tofock(N: int, number: int) -> qt.Qobj:
        num_abs = abs(number)

        if num_abs >= 2**N:
            raise AssertionError(
              "Number %d has more than %d bits!" % (num_abs, N))

        result = qt.Qobj(qt.basis(2**N, num_abs), dims=[[2]*N, [1]*N])

        return result if number >= 0 else -result

    # ----------------------------------------------------------------
    # constants and some utility functions

    FILENAME_PREFIX      = os.path.splitext(sys.argv[0])[0]

    SVG_FILENAME         = FILENAME_PREFIX + '-circ.svg'
    QASM_FILENAME        = FILENAME_PREFIX + '-circ.qasm'

    SVG_PULSE_FILEMASK  = "%s-pulse-%%s.svg" % (FILENAME_PREFIX)
    SVG_STATES_FILEMASK = "%s-states-%%s.svg" % (FILENAME_PREFIX)

    def qindex(self, i: int) -> int:
        return self.N - i - 1

    def cindex(self, i: int) -> int:
        return self.cbits_N - i - 1

    # ----------------------------------------------------------------

    # N      : Number of quantum bits for circuit.
    # CBITS_N: Number of classical bits for measuring.
    def __init__(self, N: int, cbits_N: int = 0):
        self.N = N
        self.cbits_N = cbits_N

        self.measurement_ops = qt.tensor([qt.sigmaz()]*self.N)
        self.plotter_state = StatePlotter(self.N)

        self.state = self._state.INITIALIZED

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

    # Just to assist printing output
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

    def _process_map_result(self, map_result: list):
        results = {}
        for hash_key, energy, collapsed, custom_args in map_result:
            if not hash_key in results.keys():
                results[hash_key] = [1, energy, collapsed, custom_args]
            else: results[hash_key][0] += 1

        self.plotter_state = StatePlotter(self.N)
        self.plotter_state.add(self.input, 'in')

        plot_state = qt.zero_ket(2**self.N, dims=[[2]*self.N, [1]*self.N])
        sim_runs = len(map_result)
        for count, energy, collapsed, custom_args in results.values():
            freq = count/sim_runs
            custom_args_str = (", custom_args=" + str(custom_args)) \
                               if len(custom_args) > 0 else ""
            print("Frequency %s for energy=%.2f%s %s" % (freq, energy,
                  custom_args_str, self._state2str(collapsed)))
            plot_state += np.sqrt(freq) * collapsed

        self.plotter_state.add(plot_state, 'out')

        self.analyse_sim_result(results, sim_runs)

    # ----------------------------------------------------------------
    # for state: INITIALIZED

    def init_set_measurement_ops(self, measurement_ops):
        self._assert_gr_equal(self._state.INITIALIZED)
        self.measurement_ops = measurement_ops

    # Returns an empty circuit to simulate.  Add this via
    # CIRCALLOCED_LOAD().
    def init_new_circ(self) -> cc.QubitCircuit:
        print("\nQuantum-Circuit:")
        self._assert(self._state.INITIALIZED)

        result = cc.QubitCircuit(self.N, num_cbits=self.cbits_N)

        self.state = self._state.CIRC_ALLOCED
        return result

    # ----------------------------------------------------------------
    # for state: CIRC_ALLOCED

    # Load circuit gotten from INIT_NEW_CIRC() into SimState simulator.
    def circalloced_load(self, circ: cc.QubitCircuit):
        self._assert(self._state.CIRC_ALLOCED)

        self.circ = circ

        print("%s\n%s"
              % (self.circ.gates, self.circ.propagators(expand=False)))

        self.state = self._state.CIRC_LOADED

    # ----------------------------------------------------------------
    # for state: CIRC_LOADED

    # Save a visual representation of the quantum circuit as SVG.
    def circloaded_save_svg(self):
        self._assert_gr_equal(self._state.CIRC_LOADED)

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

    # Save the quantum circuit as Open Quantum Assembly Language
    # (QASM).
    def circloaded_save_qasm(self):
        self._assert_gr_equal(self._state.CIRC_LOADED)

        try:
            qs.save_qasm(self.circ, self.QASM_FILENAME)

            print("QASM: circuit written to '%s'"
                  % (self.QASM_FILENAME))
        except Exception as e:
            print("QASM: Could not write '%s'! %s"
                  % (self.QASM_FILENAME, str(e)))

    # Quantum state to input for simulation.
    def circloaded_set_input(self, input: qt.Qobj):
        print("\nInput:")
        self._assert(self._state.CIRC_LOADED)

        self.input = input

        self.plotter_state = StatePlotter(self.N)
        self.plotter_state.add(self.input, 'in')

        print(self._state2str(input))
        self.state = self._state.INPUT_SET

    # ----------------------------------------------------------------
    # for state: INPUT_SET

    # Plot input state to SVG file.
    def inputset_save_states_svg(self, filename_postfix: str,
                                 is_fock_dec: bool=False):
        self._assert_gr_equal(self._state.INPUT_SET)

        self.plotter_state.set_is_fock_dec(is_fock_dec)

        filename = self.SVG_STATES_FILEMASK % (filename_postfix)
        try:
            self.plotter_state.save(filename)

            print("SVG      : states plotted to '%s'" % (filename))
        except Exception as e:
            print("SVG : Could not write '%s'! %s" % (filename, str(e)))

    # ***

    # Run statisitics for quantum circuit.  Means to apply unitary
    # transformations to the input state.  If CBITS_N was set to 0 in
    # __INIT__() then the result will not be measured, the state
    # vector will be print instead.
    def inputset_statistics(self, **sim_args):
        sim_args.setdefault('precompute_unitary', True)

        print("\nStatistics:")
        self._assert_gr_equal(self._state.INPUT_SET)

        sim = cc.CircuitSimulator(self.circ, **sim_args)

        stat_result = sim.run_statistics(self.input)

        stat_probs  = stat_result.get_probabilities()
        stat_states = stat_result.get_final_states()

        self.plotter_state = StatePlotter(self.N)
        self.plotter_state.add(self.input, 'in')

        plot_state = qt.zero_ket(2**self.N, dims=[[2]*self.N, [1]*self.N])
        for i in range(len(stat_probs)):
            plot_state += np.sqrt(stat_probs[i]) * stat_states[i]
            print("Probability %s for %s"
                  % (stat_probs[i], self._state2str(stat_states[i])))
        self.plotter_state.add(plot_state, 'out')

    # ***

    def _inputset_run_ol_map(self, i: int, sim: cc.CircuitSimulator):
        custom_args = {}
        sim_input = self.pre_simulation(custom_args, self.input)

        result = sim.run(sim_input)

        sim_output, meas_ops = self.pre_measurement(custom_args,
                     result.get_final_states(0), self.measurement_ops)
        energy, collapsed = qt.measurement.measure(sim_output, meas_ops)
        hash_key       = str(collapsed) + str(custom_args)

        return hash_key, energy, collapsed, custom_args

    # Run an 'operator-level' circuit simulation.
    def inputset_run_ol(self, N: int, **sim_args):
        sim_args.setdefault('precompute_unitary', True)

        print("\nSimulation: Operator-Level (applying unitaries)")
        self._assert_gr_equal(self._state.INPUT_SET)

        sim = cc.CircuitSimulator(self.circ, **sim_args)

        map_result = pa.parallel_map(self._inputset_run_ol_map,
          range(N), task_args=(sim,), progress_bar=True)

        self._process_map_result(map_result)

    # ***

    # Set a processor for 'pulse-level' circuit simulation, and add
    # noise to the pulses optionally.
    def inputset_set_processor(self, processor: dv.Processor,
                               noise: ns.Noise = None):
        procname = processor.__class__.__name__
        noisename = noise.__class__.__name__ if noise else 'no-noise'
        print("\nProcessor: %s, using %s" % (procname, noisename))

        self._assert_gr_equal(self._state.INPUT_SET)

        if isinstance(processor, dv.ModelProcessor):
            load_circuit_args = {}
        else:
            tslots = 40
            load_circuit_args = {'num_tslots': tslots, 'evo_time': tslots}

        processor.pulse_mode = "discrete"

        # Argument (, compiler=GateCompiler()) sub-class required for
        # user defined gates.
        processor.load_circuit(self.circ, **load_circuit_args)

        self.noise = noise
        if noise:
            processor.add_noise(noise)
            self.noise = noise
        else:
            self.noise = None
        self.procname = procname
        self.processor = processor
        self.state = self._state.PROCESSOR_SET

    # ----------------------------------------------------------------
    # for state: PROCESSOR_SET

    # Plot pulses to SVG file.
    def processorset_plot_pulses(self):
        self._assert_gr_equal(self._state.PROCESSOR_SET)

        fig, axis = self.processor.plot_pulses()

        # Add noise to plot
        if self.noise:
            noisy_qobjevo, _ = self.processor.get_qobjevo(noisy=True)
            noisy_pulse = noisy_qobjevo.to_list()
            for i in range(1, len(noisy_pulse), 2):
                noisy_coeff = noisy_pulse[i][1] + noisy_pulse[i+1][1]
                axis[i//2].step(noisy_qobjevo.tlist, noisy_coeff)

        filename = self.SVG_PULSE_FILEMASK % (self.procname)
        try:
            fig.savefig(filename, format='svg', transparent=True)

            print("SVG      : pulses plotted to '%s'" % (filename))
        except Exception as e:
            print("SVG: Could not write '%s'! %s" % (filename, str(e)))

    # ***

    def _processorset_run_pl_map(self, i: int, processor: dv.Processor):
        custom_args = {}
        sim_input = self.pre_simulation(custom_args, self.input)

        result = self.processor.run_state(sim_input)

        sim_output, meas_ops = self.pre_measurement(custom_args,
                               result.states[-1], self.measurement_ops)
        energy, collapsed = qt.measurement.measure(sim_output, meas_ops)
        hash_key       = str(collapsed) + str(custom_args)

        return hash_key, energy, collapsed, custom_args

    # Run a 'pulse-level' circuit simulation.
    def processorset_run_pl(self, N: int):
        print("\nSimulation: Pulse-Level")
        self._assert_gr_equal(self._state.PROCESSOR_SET)

        map_result = pa.parallel_map(self._processorset_run_pl_map,
          range(N), task_args=(self.processor,), progress_bar=True)

        self._process_map_result(map_result)

    # ***

# end of class SimState
# ********************************************************************
