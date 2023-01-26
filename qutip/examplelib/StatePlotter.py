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


__all__ = ['StatePlotter']

import sys, os

import numpy as np
from matplotlib import pyplot as mp

import qutip as qt

# ********************************************************************

class StatePlotter:
    # ----------------------------------------------------------------
    # constants

    TITLE = "Circuit Input"

    LABEL_X = "time $t$"
    LABEL_Y = "probability $\\left|\\alpha\\right|^2$"

    STATIC_TIME = 0.8

    # ----------------------------------------------------------------
    # private stuff

    def _psi2probs_summed(self, psi: qt.Qobj) -> np.array:
        return np.matmul(np.tri(self.n, self.n),
                         np.abs(np.array(psi))**2)

    def _labels_x(self) -> list:
        result = []
        for t in self.t_arr:
            int_t = int(t)
            if t != int_t:
                result.append('')
                continue
            result.append('$\\left|\\psi_' + str(int_t) + '\\right>$')

        return result

    # ----------------------------------------------------------------

    # N: Number of quantum bits
    def __init__(self, N: int):
        self.n = 2**N

        self.t_arr   = np.array([])
        self.psi_arr = np.array([[]]*self.n)

        self.psi_counter = 0

    def add(self, psi: qt.Qobj):
        self.t_arr = np.append(self.t_arr, self.psi_counter)
        self.t_arr = np.append(self.t_arr,
                               self.psi_counter + self.STATIC_TIME)

        self.psi_arr = np.append(self.psi_arr,
                                 self._psi2probs_summed(psi), axis=1)
        self.psi_arr = np.append(self.psi_arr,
                                 self._psi2probs_summed(psi), axis=1)

        self.psi_counter += 1

    def show(self):
        fig, axs = mp.subplots(1, 1, figsize=(9, 5), sharex=True)

        axs.set_title(self.TITLE)
        axs.set_xlabel(self.LABEL_X)
        axs.set_ylabel(self.LABEL_Y)
        axs.set_xticks(self.t_arr, labels=self._labels_x())

        axs.fill_between(self.t_arr, self.psi_arr[0])
        for i in range(1, len(self.psi_arr)):
            axs.fill_between(self.t_arr,
                             self.psi_arr[i-1], self.psi_arr[i])

        mp.show()

# end of class StatePlotter
# ********************************************************************
