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
import matplotlib as ma
from matplotlib import pyplot as mp

import qutip as qt

# ********************************************************************

class StatePlotter:
    # ----------------------------------------------------------------
    # constants

    LABEL_X = "time $t$"
    LABEL_Y = "Fock base $\\mapsto$" \
              + " probability $\\left|\\alpha\\right|^2$"

    STATIC_TIME = 0.8

    LABEL_XTICK_FMT = "$\\left|\\psi_{%s}\\right>$"

    LABEL_LEGEND_FMT = "$\\left|%s\\right> \\mapsto %%.2f$"
    LABEL_LEGEND_SIZE           = 12
    LABEL_LEGEND_OFFSET_X       =  0.03
    LABEL_LEGEND_OFFSET_Y       = -0.015
    LABEL_LEGEND_HIDE_THRESHOLD =  0.02

    POLY_FACECOLOR_RGB = [0.9]*3
    POLY_EDGECOLOR_RGB = [0.0]*3

    # ----------------------------------------------------------------
    # private stuff

    def _int2bin(self, v: int) -> str:
        result = ''

        v = abs(v)
        for i in range(self.N):
            result = ('1' if v & 0x1 else '0') + result
            v >>= 1

        return result

    def _labels_x(self) -> list:
        result = []
        for i in range(len(self.t_arr)):
            int_t = int(self.t_arr[i])
            if self.t_arr[i] != int_t: continue
            ind_str = str(int_t) if   self.t_name_arr[i//2] == None \
                                 else self.t_name_arr[i//2]
            result.append(self.LABEL_XTICK_FMT % (ind_str))

        return result

    def _labels_legend(self) -> list:
        result = []

        for i in range(self.n):
            result.append(self.LABEL_LEGEND_FMT % (self._int2bin(i)))

        return result

    def _prepare_plot(self, title: str) \
                            -> (ma.figure.Figure, ma.axes.Axes):
        fig, axs = mp.subplots(1, 1, figsize=(9, 5), sharex=True)

        labels_x = self._labels_x()
        if title: axs.set_title(title)
        axs.spines[['top', 'right']].set_visible(False)
        axs.set_xlabel(self.LABEL_X)
        axs.set_ylabel(self.LABEL_Y)
        axs.set_xticks(list(range(len(labels_x))), labels=labels_x)

        labels_legend = self._labels_legend()
        polys = axs.stackplot(self.t_arr, self.psi_arr,
                              labels=labels_legend, baseline='zero',
                              colors=[self.POLY_FACECOLOR_RGB])

        for v in polys: v.set_edgecolor(self.POLY_EDGECOLOR_RGB)

        for i in range(len(self.t_arr)):
            labels = []
            labels_prob = []
            for n in range(len(labels_legend)):
                int_t = int(self.t_arr[i])

                if self.t_arr[i] != int_t \
                   or self.psi_arr[n][i] == 0: continue

                labels_prob.append(self.psi_arr[n][i])
                if self.psi_arr[n][i] < self.LABEL_LEGEND_HIDE_THRESHOLD:
                    labels.append(None)
                else:
                    labels.append(labels_legend[n] % (self.psi_arr[n][i]))

            y_offset = 0
            for j in range(len(labels)):
                y = y_offset + labels_prob[j]/2
                y_offset += labels_prob[j]

                if labels[j] == None: continue

                axs.annotate(labels[j],
                  (self.t_arr[i] + self.LABEL_LEGEND_OFFSET_X,
                               y + self.LABEL_LEGEND_OFFSET_Y),
                  fontsize=self.LABEL_LEGEND_SIZE)

        return fig, axs

    # ----------------------------------------------------------------

    # N: Number of quantum bits
    def __init__(self, N: int):
        self.N = N
        self.n = 2**N

        self.t_name_arr = []
        self.t_arr      = np.array([])
        self.psi_arr    = np.array([[]]*self.n)

        self.psi_counter = 0

    def add(self, psi: qt.Qobj, t_name: str=None):
        psi_unit = psi.unit()

        if (psi != psi_unit):
            print("Warning: StatePlotter.add(): |psi> is not a unit"
                  + " vector!  Adapting normalization.")

        self.t_name_arr.append(t_name)

        self.t_arr = np.append(self.t_arr, self.psi_counter)
        self.t_arr = np.append(self.t_arr,
                               self.psi_counter + self.STATIC_TIME)

        self.psi_arr = np.append(self.psi_arr,
                                 np.abs(np.array(psi_unit))**2, axis=1)
        self.psi_arr = np.append(self.psi_arr,
                                 np.abs(np.array(psi_unit))**2, axis=1)

        self.psi_counter += 1

    def show(self,
             title: str='Unitary evolution of $\\left|\\psi\\right>$'):
        fig, axs = self._prepare_plot(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))

        mp.show()

    def save(self, filename: str, title: str=None):
        fig, axs = self._prepare_plot(title)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

        fig.savefig(filename, format='svg', transparent=True)

# end of class StatePlotter
# ********************************************************************
