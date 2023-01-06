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


__all__ = ['DefaultSim']

from .SimState import *

import numpy as np

import qutip as qt
import qutip.qip.device as dv

# ********************************************************************

class DefaultSim (SimState):

    # Run all file outputs, statistics and simulations.  Requires that
    # a circuit was loaded via CIRCALLOCED_LOAD() and input set via
    # CIRCLOADED_SET_INPUT.
    def circloaded_run_all(self, ol_runs=2000, pl_runs=250):
        self._assert_gr_equal(self._state.INPUT_SET)

        # ************************************************************
        # Save a visual representation of the quantum circuit as SVG.

        self.circloaded_save_svg()

        # ************************************************************
        # Save the quantum circuit as Open Quantum Assembly Language
        # (QASM).

        self.circloaded_save_qasm()

        # ************************************************************
        # Run statisitics for quantum circuit.

        self.inputset_statistics()

        # ************************************************************
        # Run an 'operator-level' circuit simulation.

        self.inputset_run_ol(ol_runs)

        # ************************************************************
        # Setup a processor for 'pulse-level' circuit simulation.

        # Set True to use a (realistic) ModelProcessor, otherwise an
        # 'Optimal Control' will be used to predict optimal control
        # pulses for the user defined Hamiltonians.
        if True:
            processor = dv.LinearSpinChain(self.N)
            #processor = dv.CircularSpinChain(self.N) # Noise not working
            #processor = dv.DispersiveCavityQED(self.N, num_levels=2) # ?
        else:
            processor = dv.OptPulseProcessor(
                        self.N, drift=qt.tensor([qt.sigmaz()]*self.N))
            processor.add_control(qt.sigmax(), cyclic_permutation=True)
            processor.add_control(qt.sigmay(), cyclic_permutation=True)
            processor.add_control(qt.tensor([qt.sigmay()]*self.N),
                                  cyclic_permutation=True)

        noise = qt.qip.noise.RandomNoise(dt=0.01,
                rand_gen=np.random.normal, loc=0.00, scale=0.02)

        # For User-Gates it is required to compile it into pulses.
        # For now it is not implemented yet, so we skip the
        # pulse-level simulation in such a case.
        if not self.circ.user_gates:
            self.inputset_set_processor(processor, noise)

            # ********************************************************
            # Plot pulses to SVG file.

            self.processorset_plot_pulses()

            # ********************************************************
            # Run a 'pulse-level' circuit simulation.

            self.processorset_run_pl(pl_runs)

            # ********************************************************

# end of class DefaultSim
# ********************************************************************
