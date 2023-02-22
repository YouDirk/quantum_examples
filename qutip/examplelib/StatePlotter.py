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
    LABEL_Y = "Fock base $\\left| n \\right> \\mapsto$" \
              + " probability $\\left|\\alpha_n\\right|^2 \\angle$" \
              + " phase $\\varphi_n$"

    STATIC_TIME = 0.9

    LABEL_XTICK_FMT = "$\\left|\\psi_{%s}\\right>," \
                      + " \\varphi_{global} = %.2f \\pi$"

    LABEL_LEGEND_FMT = "$\\left|%s\\right> \\mapsto" \
                       + " %%.2f \\angle %%.2f \\pi$"
    LABEL_LEGEND_SIZE           = 12
    LABEL_LEGEND_OFFSET_X       =  0.03
    LABEL_LEGEND_OFFSET_Y       = -0.015
    LABEL_LEGEND_HIDE_THRESHOLD =  0.02

    POLY_FACECOLOR_MAX_RGBA = [*[0.95]*3, 1.0]
    POLY_FACECOLOR_MIN_RGBA = [*[0.25]*3, 1.0]
    POLY_FACECOLOR_DIFF_RGBA = np.subtract(
        POLY_FACECOLOR_MAX_RGBA, POLY_FACECOLOR_MIN_RGBA)

    POLY_EDGECOLOR_RGBA = [*[0.0]*3, 1.0]

    HATCH_LINEWIDTH    = 0.3
    HATCH_IGNORE_DELTA = 1e-6

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
        len_t_arr = 2*self.psi_counter

        result = []
        for i in range(len_t_arr):
            int_t = int(self.t_arr[i])
            if self.t_arr[i] != int_t: continue
            ind_str = str(int_t) if   self.t_name_arr[i//2] == None \
                                 else self.t_name_arr[i//2]
            result.append(self.LABEL_XTICK_FMT
                          % (ind_str, self.psi_phase_arr[0][i]))

        return result

    def _labels_legend(self) -> list:
        result = []

        for i in range(self.n):
            result.append(self.LABEL_LEGEND_FMT % (self._int2bin(i)))

        return result

    def __hatch_select_range(self, begin: int) -> float:
        return 1/8 + begin/4  - (.0 if begin <= 3 else 2.)
    def __hatch_select_isrange(self, r_begin: float, r_end: float,
                               phase: float) -> bool:
        if r_begin >= .0 and r_end < .0:
            # BEGIN=3: phase >= 1/8 + 3/4 or phase <= -1/8 - 3/4
            return phase >= r_begin or phase <= r_end
        return phase >= r_begin and phase <= r_end
    # BEGIN: [0..3] test if phase is in range for
    #               1/8 + BEGIN/4 .. 1/8    + (BEGIN + 1)/4
    #            or opposite
    #               1/8 + OPPOSITE/4 .. 1/8 + (OPPOSITE + 1)/4
    # PHASE: Range [-PI .. PI]
    def _hatch_select(self, begin: int, phase: float) -> bool:
        OPPOSITE = 4
        r1_begin = self.__hatch_select_range(begin)
        r1_end   = self.__hatch_select_range(begin + 1)
        r2_begin = self.__hatch_select_range(begin + OPPOSITE)
        r2_end   = self.__hatch_select_range(begin + OPPOSITE+1)

        return self.__hatch_select_isrange(r1_begin, r1_end, phase) \
            or self.__hatch_select_isrange(r2_begin, r2_end, phase)

    def _prepare_plot(self, title: str) \
                            -> (ma.figure.Figure, ma.axes.Axes):
        fig, axs = mp.subplots(1, 1, figsize=(9, 5), sharex=True)

        labels_x = self._labels_x()
        psi_arr = np.empty([self.n, 0]) \
            if self.psi_arr.shape[0] == 0 else self.psi_arr

        if title: axs.set_title(title)
        axs.spines[['top', 'right']].set_visible(False)
        axs.set_xlabel(self.LABEL_X)
        axs.set_ylabel(self.LABEL_Y)
        axs.set_xticks(list(range(len(labels_x))), labels=labels_x,
                       ha='left')

        labels_legend = self._labels_legend()
        # len(POLYS): SELF.n*SELF.PSI_COUNTER
        polys = axs.stackplot(self.t_arr, psi_arr,
                              labels=labels_legend, baseline='zero',
                              colors=[self.POLY_FACECOLOR_MAX_RGBA],
                              edgecolors=[self.POLY_EDGECOLOR_RGBA])

        len_t_arr = 2*self.psi_counter
        for i in range(len_t_arr):
            labels = []
            labels_prob = []
            for n in range(len(labels_legend)):
                int_t = int(self.t_arr[i])
                n_psi_arr = i//2*self.n + n
                cur_phase = self.psi_phase_arr[n][i]

                if self.t_arr[i] != int_t \
                   or self.psi_arr[n_psi_arr][i] == 0: continue

                face_scale = np.multiply(self.POLY_FACECOLOR_DIFF_RGBA,
                                         abs(cur_phase))
                face_color = np.subtract(self.POLY_FACECOLOR_MAX_RGBA,
                                         face_scale)
                polys[n_psi_arr].set_facecolor(face_color)

                hatch = None
                if     cur_phase >    - self.HATCH_IGNORE_DELTA \
                   and cur_phase <      self.HATCH_IGNORE_DELTA \
                   or  cur_phase < -1 + self.HATCH_IGNORE_DELTA \
                   or  cur_phase >  1 - self.HATCH_IGNORE_DELTA:
                    hatch = None
                elif self._hatch_select(3, cur_phase):
                    hatch = '-'
                elif self._hatch_select(0, cur_phase):
                    hatch = '/'
                elif self._hatch_select(1, cur_phase):
                    hatch = '|'
                elif self._hatch_select(2, cur_phase):
                    hatch = '\\'
                polys[n_psi_arr].set_hatch(hatch)

                labels_prob.append(self.psi_arr[n_psi_arr][i])
                if self.psi_arr[n_psi_arr][i] \
                   < self.LABEL_LEGEND_HIDE_THRESHOLD:
                    labels.append(None)
                else:
                    labels.append(labels_legend[n]
                      % (self.psi_arr[n_psi_arr][i], cur_phase))

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

        # len(): SELF.PSI_COUNTER
        self.t_name_arr    = []
        # len(): 2*SELF.PSI_COUNTER
        self.t_arr         = np.empty([0])
        # len(): SELF.n*SELF.PSI_COUNTER X 2*SELF.PSI_COUNTER
        self.psi_arr       = np.empty([0, 0])
        # len(): SELF.n X 2*SELF.PSI_COUNTER
        self.psi_phase_arr = np.empty([self.n, 0])

        self.psi_counter = 0

        mp.rcParams['hatch.linewidth'] = self.HATCH_LINEWIDTH

    def add(self, psi: qt.Qobj, t_name: str=None):
        if not psi.isket:
            raise AssertionError("Psi is not a ket vector!")

        psi_unit = psi.unit()
        len_psi_y = self.n*self.psi_counter

        if (psi != psi_unit):
            print("Warning: StatePlotter.add(): |psi> is not a unit"
                  + " vector!  Adapting normalization.")

        self.t_name_arr.append(t_name)

        self.t_arr = np.append(self.t_arr, self.psi_counter)
        self.t_arr = np.append(self.t_arr,
                               self.psi_counter + self.STATIC_TIME)

        cur_psi_arr = np.abs(np.array(psi_unit))**2
        self.psi_arr = np.block([
            [self.psi_arr, np.zeros([len_psi_y, 2])],
            [np.zeros([self.n, 2*self.psi_counter]),
             cur_psi_arr, cur_psi_arr]
        ])

        # PSI_PHASE >= -1 && PSI_PHASE <= 1  [angular velocity (* PI)]
        cur_psi_phase_arr = np.angle(np.array(psi_unit))/np.pi
        self.psi_phase_arr = np.block([
            self.psi_phase_arr, cur_psi_phase_arr, cur_psi_phase_arr])

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
