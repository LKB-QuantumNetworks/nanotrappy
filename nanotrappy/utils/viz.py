from __future__ import annotations
from copy import copy
from nanotrappy.trapping.geometry import AxisX, AxisY, AxisZ
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from operator import itemgetter
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.signal import find_peaks

import re
import itertools
import mplcursors
import time

from nanotrappy.utils.utils import *
from nanotrappy.utils.physicalunits import *

_custom_highlight_kwargs = dict(
    # Only the kwargs corresponding to properties of the artist will be passed.
    # Line2D.
    color="tab:red",
    markeredgecolor="tab:red",
    linewidth=3,
    markeredgewidth=3,
    # PathCollection.
    facecolor="tab:red",
    edgecolor="tab:red",
)

_custom_annotation_kwargs = dict(
    bbox=dict(
        boxstyle="round,pad=.5",
        fc="#F6D3D4",  # "tab:red",
        alpha=1,
        ec="#F6D3D4",  # "k",
    ),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3",
        shrinkB=0,
        ec="#F6D3D4",  # "k",
        fc="#F6D3D4",
    ),
)

# font = {"size": 22}
import matplotlib

# matplotlib.rc("font", **font)
matplotlib.rcParams["axes.unicode_minus"] = False


class Viz:
    """Class that contains all the visualization methods.

    Attributes:
        simul (Simulation object): Simulation object contaning a trap,
            a system and a surface. For most of the methods in the class,
            the simulations have to be run beforehand.
        trapping_axis (str): Axis perpendicular to the structure along which
            we want to trap the atoms. Important for the 3 1D plot method.
            Either "X", "Y" or "Z".

    """

    def __init__(self, simul, trapping_axis):
        # Trapping_axis is the one perpendicular to the surface if one is defined
        self.trapping_axis = trapping_axis
        self.simul = simul

        ## Convenient process of str input, to match arxiv version. Will be removed
        if isinstance(self.trapping_axis, str):
            if self.trapping_axis == "X":
                self.trapping_axis = AxisX()
            elif self.trapping_axis == "Y":
                self.trapping_axis = AxisY()
            elif self.trapping_axis == "Z":
                self.trapping_axis = AxisZ()
            else:
                raise ValueError("Non valid name for trapping axis. Choose between X, Y or Z.")

    def plot_trap(self, mf=0, Pranges=[10, 10], increments=[0.1, 0.1]):
        """Shows a 2D plot of the total potential with power sliders
        Only available if a 2D simulation has been run.

        Args:
            plane (str): As we are dealing with 2D plots, we have to specify
            the plane we are looking at to choose the right coordinates for plotting.
            mf (int): Mixed mf state we want to plot. In 2D we can only
            specify one integer. Default to 0.
            Pranges (list): List with the maximum values of the beam powers
            we want to display on the sliders. Defaults to [10,10]
        Raise:
            TypeError: if only a 1D computation of the potential has been
            done before plotting.

        Returns:
            (tuple): containing:

                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)

        """

        if len(Pranges) != len(self.simul.trap.beams):
            raise ValueError(
                "When specifying the upper ranges of P for plotting, you have to give as many as many values as there are beams."
            )

        _, mf = check_mf(self.simul.atomicsystem.f, mf)

        dimension = self.simul.geometry.get_dimension()
        # coord1, coord2 = set_axis_from_plane(plane, self.simul)
        if dimension == 2:
            mf_index = int(mf + self.simul.atomicsystem.f)

            axis1, axis2 = self.simul.geometry.get_base_axes()
            coord1 = axis1.fetch_in(self.simul)
            coord2 = axis2.fetch_in(self.simul)
            # coord1, coord2 = getattr(self.simul, axis1.name), getattr(self.simul, axis2.name)

            trap = np.real(self.simul.total_potential())[:, :, mf_index]
            trap_noCP = np.real(self.simul.total_potential_noCP[:, :, mf_index])
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.5, bottom=0.1)
            # the norm TwoSlopeNorm allows to fix the 0 of potential to the white color, so that we can easily distinguish between positive and negative values of the potential
            a = ax.pcolormesh(
                coord1 / nm,
                coord2 / nm,
                np.transpose(trap),
                shading="gouraud",
                norm=colors.TwoSlopeNorm(
                    vmin=min(np.min(trap_noCP), -0.001), vcenter=0, vmax=max(np.max(trap_noCP) * 2, 0.001)
                ),
                cmap="seismic_r",
            )
            cbar = plt.colorbar(a)
            cbar.set_label("Total potential (mK)", rotation=270, labelpad=12, fontsize=14)

            ax.set_xlabel("%s (nm)" % (self.simul.geometry.name[0].lower()), fontsize=14)
            ax.set_ylabel("%s (nm)" % (self.simul.geometry.name[1].lower()), fontsize=14)
            plt.setp(ax.spines.values(), linewidth=1.5)
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.set_title(
                "2D plot of trapping potential \n for mf = %s in the %s plane" % (mf, self.simul.geometry.name.upper()),
                fontsize=18,
            )

            ax.margins(x=0)
            axcolor = "lightgoldenrodyellow"
            slider_ax = []
            axes = []

            for (k, beam) in enumerate(self.simul.trap.beams):
                axes.append(plt.axes([0.15 + k * 0.08, 0.1, 0.03, 0.75], facecolor=axcolor))
                slider_ax.append(
                    Slider(
                        axes[k],
                        "Power \n Beam %s (mW)" % (k + 1),
                        0,
                        Pranges[k],
                        valinit=self.simul.trap.beams[k].get_power()[0] * 1e3,
                        valstep=increments[k],
                        orientation="vertical",
                    )
                )

        elif dimension == 1:
            mf_index = mf + [self.simul.atomicsystem.f]

            # x = getattr(self.simul, self.simul.geometry.name)
            x = self.simul.geometry.fetch_in(self.simul)
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.27)
            jet = cm = plt.get_cmap("Greys")
            cNorm = colors.Normalize(vmin=-1, vmax=len(mf))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            a = []

            trap = np.real(self.simul.total_potential())
            trap_noCP = np.real(self.simul.total_potential_noCP)
            ax.set_xlabel("%s (nm)" % (self.simul.geometry.name), fontsize=14)
            ax.set_ylabel("E (mK)", fontsize=14)
            plt.setp(ax.spines.values(), linewidth=1.5)
            ax.axhline(y=0, color="black", linestyle="--", linewidth=2)
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.set_title(
                "1D plot of trapping potential \n for mf = %s along %s " % (mf, self.simul.geometry.name), fontsize=18
            )

            for k in range(len(mf_index)):
                colorVal = "k"  # scalarMap.to_rgba(k)
                a = a + plt.plot(
                    x / nm,
                    trap[:, mf_index[k]],
                    color=colorVal,
                    label="m$_f$ = %s" % (mf[k]),
                    linewidth=2 + 3 / len(self.simul.mf_all),
                )

            if len(mf) == 1 and len(self.simul.trap.beams) == 2:
                (b,) = plt.plot(
                    x / nm,
                    np.real(self.simul.trap.beams[0].get_power()[0] * np.real(self.simul.potentials[0, :, mf_index[0]])),
                    color="blue",
                    linewidth=2,
                )
                (r,) = plt.plot(
                    x / nm,
                    np.real(self.simul.trap.beams[1].get_power()[0] * np.real(self.simul.potentials[1, :, mf_index[0]])),
                    color="red",
                    linewidth=2,
                )
            else:
                pass
                # plt.legend()

            axcolor = "lightgoldenrodyellow"
            slider_ax = []
            axes = []
            for (k, beam) in enumerate(self.simul.trap.beams):
                axes.append(plt.axes([0.25, 0.15 - k * 0.1, 0.6, 0.03], facecolor=axcolor))
                slider_ax.append(
                    Slider(
                        axes[k],
                        "Power \n Beam %s (mW)" % (k + 1),
                        0,
                        Pranges[k],
                        valinit=self.simul.trap.beams[k].get_power()[0]*1e3,
                        valstep=increments[k],
                    )
                )
                slider_ax[k].label.set_size(14)

            cursor = mplcursors.cursor(
                a,
                highlight=True,
                highlight_kwargs=_custom_highlight_kwargs,
                annotation_kwargs=_custom_annotation_kwargs,
            )

            @cursor.connect("add")
            def on_add(sel):
                artist = sel.artist
                label = artist.get_label() or ""
                mf = self.simul.atomicsystem.f + int(label.split()[2])

                label = f"Choice : {label}"
                idx = int(sel.target.index)

                temp_vec = self.simul.total_vecs[idx, mf]
                temp_vec = np.abs(temp_vec) ** 2
                decomp = f"State : {vec_to_string(temp_vec)}"

                x, y = sel.target
                textx = f"x = {x:.1f} nm"
                texty = f"y = {y:.2f} mK"

                size = max(len(textx), len(texty), len(decomp))
                label = label.center(size, "-")
                text = f"{label}\n{textx}\n{texty}\n{decomp}"
                sel.annotation.set_text(text)

        def updateP(val):
            if dimension == 1:
                for selection in cursor.selections:
                    cursor.remove_selection(selection)
                P = []
                for (k, slider) in enumerate(slider_ax):
                    P.append(slider.val * mW)
                self.simul.trap.set_powers(P)
                trap = np.real(self.simul.total_potential()) 
                for k in range(len(mf)):
                    trap_k = trap[:, mf_index[k]]
                    a[k].set_ydata(trap_k)

                if len(mf) == 1 and len(self.simul.trap.beams) == 2:
                    b.set_ydata(
                        np.real(
                            self.simul.trap.beams[0].get_power()[0] * np.real(self.simul.potentials[0, :, mf_index[0]])
                        )
                    )
                    r.set_ydata(
                        np.real(
                            self.simul.trap.beams[1].get_power()[0] * np.real(self.simul.potentials[1, :, mf_index[0]])
                        )
                    )

            elif dimension == 2:
                P = []
                for (k, slider) in enumerate(slider_ax):
                    P.append(slider.val * mW)
                self.simul.trap.set_powers(P)
                trap_2D = self.simul.total_potential()[:, :, mf_index]
                a.set_array(np.transpose(np.real(self.simul.total_potential_noCP[:, :, mf_index])).ravel())
                a.autoscale()
                a.set_array(np.transpose(np.real(trap_2D)).ravel())

            fig.canvas.draw_idle()

        for slider in slider_ax:
            slider.on_changed(updateP)

        plt.show()

        return fig, ax, slider_ax

    def restrict_trap_from_surfaces(self, mf=0):
        """Returns the truncation of both the specified axis and the trap along that direction, setting 0 for the coordinate at the edge of the structure.

        Args:
            axis (str): axis along which we are looking at the trap.
            coord1 (float): First coordinate on the orthogonal plane to the
            trapping axis. If axis is Y, coord1 should be the one on X.
            coord2 (float): Second coordinate on the orthogonal plane to the
            trapping axis.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.

        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.

        Returns:
            (tuple): containing:

                - int: Index of the specified mf state in the array
                - float: Position of the edge of the structure (taken either from the Surface object or given by the user).
                - array: New coordinates, with 0 at the edge of the structure and negative values truncated.
                - array: Corresponding truncation of the trapping potential.
        """
        _, mf = check_mf(self.simul.atomicsystem.f, mf)
        mf_index = int(mf + self.simul.atomicsystem.f)

        old_geometry = copy(self.simul.geometry)

        self.simul.geometry = self.trapping_axis
        coord_main = self.trapping_axis.fetch_in(self.simul)
        trap_main = np.real(self.simul.compute())[0][:, mf_index]

        # self.simul.geometry = self.trapping_axis.normal_plane.get_base_axes()[0]
        # coord_1 = self.simul.geometry.fetch_in(self.simul)
        # trap_1 = np.real(self.simul.compute())[0][:, mf_index]

        # self.simul.geometry = self.trapping_axis.normal_plane.get_base_axes()[1]
        # coord_2 = self.simul.geometry.fetch_in(self.simul)
        # trap_2 = np.real(self.simul.compute())[0][:, mf_index]

        self.simul.geometry = old_geometry

        for surface in self.simul.surface:
            coord_main, trap_main = surface.get_slab(coord_main, trap_main, self.simul, self.trapping_axis)

        return coord_main, trap_main  # , coord_1, trap_1, coord_2, trap_2

        # return mf_index, edge, y_outside, trap_outside

    def plot_3axis(self, mf=0, Pranges=[10, 10], increments=[0.1, 0.1]):
        """Shows 3 1D plots of the total potential with power sliders,
        and trapping frequencies for each axis if possible.
        Starts by simulating a 1D trap along the trapping_axis attribute of
        the Viz object and finds the minimum.
        Once found, simulates 1D traps in the 2 other orthogonal directions
        and finds the associated frequency.
        When looking at nanostructure with possible different trapping axis
        (like nanofibers), a new Viz object has to be defined
        in order to use this method.

        Args:
            coord1 (float): First coordinate on the orthogonal plane to the
                trapping axis. If trapping axis is Y, coord1 should be the one on X.
            coord2 (float): Second coordinate on the orthogonal plane to the
                trapping axis.
            mf (int): integer between -F and +F. No list possible here.
            Pranges (list): List with the maximum values of the beam powers we
                want to display on the sliders. Defaults to [10,10]

        Returns:
            (tuple): containing:

                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)
        """
        _, mf = check_mf(self.simul.atomicsystem.f, mf)
        mf_index = int(mf + self.simul.atomicsystem.f)

        if len(mf) > 1:
            raise ValueError("This 3D plot can only be done for one specific mf at a time")

        mf_shift = mf + self.simul.atomicsystem.f

        main_axis = self.trapping_axis
        axis1, axis2 = self.trapping_axis.normal_plane.get_base_axes()
        axis_name_list = [main_axis.name, axis1.name, axis2.name]

        main_axis_data = main_axis.fetch_in(self.simul)
        axis1_data, axis2_data = np.reshape(axis1.fetch_in(self.simul),-1), np.reshape(axis2.fetch_in(self.simul),-1)

        if len(self.simul.E[0].shape) != 4:
            print("[WARNING] 3D Electric fields must be fed in the Simulation class in order to use this function")
        else:
            # y_out_main, trap_out_main, y_out_1, trap_out_1, y_out_2, trap_out_2 = self.restrict_trap_from_surfaces(
            #     mf=mf
            # )
            y_out_main, trap_out_main = self.restrict_trap_from_surfaces(mf=mf)
            ymin_ind, y_min, trap_depth, trap_prominence, _ = self.get_min_trap(y_out_main, trap_out_main)

            omega_1, omega_main, omega_2 = 0, 0, 0
            if not np.isnan(y_min):
                # min_pos = np.zeros(3)
                # min_pos[main_axis.index] = y_min  # + edge
                # min_pos[axis1.index] = main_axis.coordinates[0]
                # min_pos[axis2.index] = main_axis.coordinates[1]

                ax1, ax2 = self.trapping_axis.complete_orthogonal_basis(position=y_min)
                old_geometry = copy(self.simul.geometry)

                self.simul.geometry = ax1
                y_out_1 = self.simul.geometry.fetch_in(self.simul)
                trap_out_1 = np.real(self.simul.compute())[0][:, mf_index]
                omega_1 = self.get_trapfreq(y_out_1, trap_out_1)

                self.simul.geometry = ax2
                y_out_2 = self.simul.geometry.fetch_in(self.simul)
                trap_out_2 = np.real(self.simul.compute())[0][:, mf_index]
                omega_2 = self.get_trapfreq(y_out_2, trap_out_2)

                self.simul.geometry = old_geometry

                omega_main = self.get_trapfreq(y_out_main, trap_out_main)

                # omega_1 = self.get_trapfreq(y_out_1, trap_out_1)

                # omega_2 = self.get_trapfreq(y_out_2, trap_out_2)

            fig, ax = plt.subplots(3, figsize=(15, 10))
            plt.subplots_adjust(left=0.25)
            axcolor = "lightgoldenrodyellow"
            props = dict(boxstyle="round", facecolor=axcolor, alpha=0.5)

            textstr = "\n".join(
                (
                    r"$\mathrm{trap \, position}=%.2f (nm) $" % (y_min * 1e9,),
                    r"$\mathrm{depth}=%.2f (mK) $" % (trap_depth,),
                    r"$\omega_%s =%.2f (kHz) $"
                    % (
                        self.trapping_axis.name,
                        omega_main * 1e-3,
                    ),
                    r"$\omega_%s =%.2f (kHz) $"
                    % (
                        axis1.name,
                        omega_1 * 1e-3,
                    ),
                    r"$\omega_%s =%.2f (kHz) $"
                    % (
                        axis2.name,
                        omega_2 * 1e-3,
                    ),
                )
            )

            box = plt.text(
                -0.3, 0.6, textstr, transform=ax[2].transAxes, fontsize=14, verticalalignment="top", bbox=props
            )

            slider_ax = []
            axes = []
            for (k, beam) in enumerate(self.simul.trap.beams):
                axes.append(plt.axes([0.1 + k * 0.05, 0.32, 0.03, 0.5], facecolor=axcolor))
                print(self.simul.trap.beams[k].get_power())
                slider_ax.append(
                    Slider(
                        axes[k],
                        "Power \n Beam %s \n (mW)" % (k + 1),
                        0,
                        Pranges[k],
                        valinit=self.simul.trap.beams[k].get_power() * 1e3,
                        valstep=increments[k],
                        orientation="vertical",
                    )
                )

            index_1 = np.argmin(np.abs(axis1_data - main_axis.coordinates[0]))
            index_2 = np.argmin(np.abs(axis2_data - main_axis.coordinates[1]))

            (ly,) = ax[0].plot(y_out_main, trap_out_main, linewidth=3, color="darkblue")
            ax[0].set_ylim([-2, 2])
            if not np.isnan(y_min):
                (point,) = ax[0].plot(y_out_main[int(ymin_ind)], trap_out_main[int(ymin_ind)], "ro")
                (lx,) = ax[1].plot(axis1_data, trap_out_1, linewidth=2, color="royalblue")
                (lz,) = ax[2].plot(axis2_data, trap_out_2, linewidth=2, color="royalblue")
                (point1,) = ax[1].plot(axis1_data[index_1], trap_out_1[index_1], "ro")
                (point2,) = ax[2].plot(axis2_data[index_2], trap_out_2[index_2], "ro")

            else:
                (lx,) = ax[1].plot(axis1_data, np.zeros((len(axis1_data),)), linewidth=2, color="royalblue")
                (lz,) = ax[2].plot(axis2_data, np.zeros((len(axis2_data),)), linewidth=2, color="royalblue")

            plt.grid(alpha=0.5)
            for k in range(len(ax)):
                ax[k].set_xlabel("%s (m)" % (axis_name_list[k].lower()), fontsize=14)
                plt.setp(ax[k].spines.values(), linewidth=2)
                ax[k].axhline(y=0, color="black", linestyle="--", linewidth=2)
                ax[k].tick_params(axis="both", which="major", labelsize=14)
            ax[0].set_title("Total dipole trap for mf = %s in the 3 directions" % (mf[0]), fontsize=18)

            fig.text(0.21, 0.5, "Potential (mK)", ha="center", va="center", rotation="vertical", fontsize=14)

            def updateP(val):
                P = [slider.val * mW for (k, slider) in enumerate(slider_ax)]
                self.simul.trap.set_powers(P)
                y_out_main, trap_out_main = self.restrict_trap_from_surfaces(mf=mf)
                ymin_ind, y_min, trap_depth, trap_prominence, _ = self.get_min_trap(y_out_main, trap_out_main)

                omega_1, omega_main, omega_2 = 0, 0, 0
                if not np.isnan(y_min):
                    ax1, ax2 = self.trapping_axis.complete_orthogonal_basis(position=y_min)
                    old_geometry = copy(self.simul.geometry)

                    self.simul.geometry = ax1
                    y_out_1 = self.simul.geometry.fetch_in(self.simul)
                    trap_out_1 = np.real(self.simul.compute())[0][:, mf_index]
                    omega_1 = self.get_trapfreq(y_out_1, trap_out_1)

                    self.simul.geometry = ax2
                    y_out_2 = self.simul.geometry.fetch_in(self.simul)
                    trap_out_2 = np.real(self.simul.compute())[0][:, mf_index]
                    omega_2 = self.get_trapfreq(y_out_2, trap_out_2)

                    self.simul.geometry = old_geometry

                    omega_main = self.get_trapfreq(y_out_main, trap_out_main)

                    lx.set_ydata(trap_out_1)
                    lz.set_ydata(trap_out_2)
                    point.set_data(y_out_main[ymin_ind], trap_out_main[ymin_ind])
                    point1.set_data(axis1_data[index_1], trap_out_1[index_1])
                    point2.set_data(axis2_data[index_2], trap_out_2[index_2])

                    ax[1].set_ylim([trap_out_1.min(), trap_out_1.max()])
                    ax[2].set_ylim([trap_out_2.min(), trap_out_2.max()])
                    ax[0].set_ylim([2 * trap_depth, 2 * trap_out_main.max()])

                    textstr = "\n".join(
                        (
                            r"$\mathrm{trap \, position}=%.2f (nm) $" % (y_min * 1e9,),
                            r"$\mathrm{depth}=%.2f (mK) $" % (trap_depth,),
                            r"$\omega_%s =%.2f (kHz) $"
                            % (
                                self.trapping_axis.name,
                                omega_main * 1e-3,
                            ),
                            r"$\omega_%s =%.2f (kHz) $"
                            % (
                                axis1.name,
                                omega_1 * 1e-3,
                            ),
                            r"$\omega_%s =%.2f (kHz) $"
                            % (
                                axis2.name,
                                omega_2 * 1e-3,
                            ),
                        )
                    )

                else:
                    textstr = r"$\mathrm{depth}=%.2f (mK) $" % (trap_depth,)

                box.set_text(textstr)
                ly.set_ydata(np.squeeze(np.real(trap_out_main)))

            for slider in slider_ax:
                slider.on_changed(updateP)
            plt.show()
            return fig, ax, slider_ax

    def get_min_trap(self, y_outside, trap_outside, edge_no_surface=None, verbose = True):
        """Finds the minimum of the trap (ie total_potential()) computed in the simulation object

        Args:
            y_outside (array): truncated coordinates
            trap_outside (array): truncated 1D trap
            axis (str): axis along which we are looking at the trap.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure.
                Only needed when no Surface is specified.
                When a Surface object is given, it is found automatically with the CP masks.
                Defaults to None.

        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.

        Returns:
            (tuple): containing:

                - int: Index of the position of the minimum
                (from the outside coordinate, putting the surface at 0).
                - float: Position of the trap minimum when putting the surface at 0.
                - float: Trap depth (ie, value of the trap at the minimum).
                - float: Height of the potential barrier for the atoms
                (ie difference between the trap depth and the closest local maxima).
                - float: Idx of left prominence if exists
        """
        verboseprint = print if verbose else lambda *a, **k: None
        
        # l = len(y_outside)
        if edge_no_surface is not None:
            threshold = np.argmin(np.abs(y_outside - (edge_no_surface + 30e-9))) #no closer than 30 nm
        else : 
            threshold = 0
        # print("threshold :", threshold)
        # threshold = int(0.08*l) #8% of the whole range of y
        if np.ndim(trap_outside) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")
        local_minima = find_peaks(-trap_outside, distance=10, prominence=1e-5)
        # print(local_minima)
        if len(local_minima[0]) == 0:
            verboseprint("[WARNING] No local minimum found")
            return np.nan, np.nan, 0, 0, np.nan
        elif len(local_minima[0]) == 1 and local_minima[0][0] > threshold:
            verboseprint("[INFO] One local miminum found at %s" % (y_outside[local_minima[0][0]]))
            return (
                local_minima[0][0],
                y_outside[local_minima[0][0]],
                trap_outside[local_minima[0][0]],
                -local_minima[1]["prominences"][0],
                local_minima[1]["left_bases"][0],
            )
        elif len(local_minima[0]) == 1 and local_minima[0][0] <= threshold:
            verboseprint("[WARNING] One local minimum found but too close to the edge of the structure")
            return np.nan, np.nan, 0, 0, np.nan
        else:
            args = local_minima[0]
            prom = local_minima[1]["prominences"][args > threshold]
            base = local_minima[1]["left_bases"][args > threshold]
            args = args[args > threshold] #Removing the ones too close to the surface
            verboseprint(args)
            try :
                args = [args[np.argmax(prom)]]
            except : 
                ValueError("[WARNING] Many local minima found but none above the threshold")
                return np.nan, np.nan, 0, 0, np.nan
            # print(args)
            verboseprint(
                "[WARNING] Many local minima found, taking only the biggest one into account at %s"
                % (y_outside[args[0]])
            )
            return (
                args[0],
                y_outside[args[0]],
                trap_outside[args[0]],
                prom[0],
                base[0],
            )

    def get_trapfreq(self, y_outside, trap_outside, edge_no_surface=None, fit_range=None):
        """Finds the value of the trapping frequency (in Hz) along the specified axis

        Args:
            simul (Simulation object): A simulation object with computation of 1D potential already run.
            axis (str): axis along which we want to compute the trapping frequency.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.
            fit_range (float) : width (in points) for the quadratic fit around the trap minimum. If not specified, the range is taken as half the distance between the trap position and the peak base. 
        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.

        Returns:
            float: Trapping frequency along the axis (in Hz)
        """

        if np.ndim(trap_outside) >= 3:
            raise TypeError("The trap given must be one-dimensional")

        min_pos_index, min_pos, depth, height, height_idx = self.get_min_trap(y_outside, trap_outside, edge_no_surface, False)
        # print(min_pos, height_idx)
        if np.isnan(min_pos) and self.simul.geometry.name == 'x':
            trap_outside3 = np.concatenate((trap_outside, trap_outside, trap_outside))
            y_outside3 = np.concatenate(
                (y_outside - (y_outside[-1] - y_outside[0]), y_outside, y_outside + (y_outside[-1] - y_outside[0]))
            )
            min_pos_index, min_pos, depth, height, height_idx = self.get_min_trap(
                y_outside3, trap_outside3, edge_no_surface
            )
            if np.isnan(min_pos):
                print("[WARNING] No local minimum along the axis. Cannot compute trapping frequency.")
                return 0
            else:
                pass
        elif np.isnan(min_pos) and self.simul.geometry.name != 'x':
            return 0
        
        height_pos = y_outside[height_idx]  ## Gives the position of the barrier
        yleft = min_pos - (min_pos - height_pos) / 2
        yright = min_pos + (min_pos - height_pos) / 2
                
        idx_left = find_nearest(y_outside, yleft)
        idx_right = find_nearest(y_outside, yright)
        
        if fit_range != None :
            idx_left = min_pos_index - fit_range
            idx_right = min_pos_index + fit_range
        
        if idx_right == idx_left:
            return 0
        # print(idx_right, idx_left)
        fit = np.polyfit(y_outside[idx_left:idx_right], trap_outside[idx_left:idx_right], 2)

        p = np.poly1d(fit)
        der_fit = np.real(np.gradient(p(y_outside), y_outside))
        der2_fit = np.gradient(der_fit, y_outside)
        index_min = np.argmin(np.abs(y_outside - min_pos))
        moment2 = der2_fit[index_min]
        trap_freq = np.sqrt((moment2 * kB * mK) / (self.simul.atomicsystem.mass)) * (1 / (2 * np.pi))
        return trap_freq
    
    def get_FWHM(self, y_outside, trap_outside, edge_no_surface=None):
        """Finds the value of the Full Width Half Maximum 

        Args:
            simul (Simulation object): A simulation object with computation of 1D potential already run.
            axis (str): axis along which we want to compute the trapping frequency.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.
        """
        if np.ndim(trap_outside) >= 3:
            raise TypeError("The trap given must be one-dimensional")

        def lin_interp(x, y, i, half):
            return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
        
        min_pos_index, min_pos, depth, height, height_idx = self.get_min_trap(y_outside, trap_outside, edge_no_surface)
        # height_pos = y_outside[height_idx]  ## Gives the position of the barrier
        # yleft = min_pos - (min_pos - height_pos) / 2
        # yright = min_pos + (min_pos - height_pos) / 2
        # idx_left, idx_right = find_nearest(y_outside, yleft),find_nearest(y_outside, yright)
        # absy = np.abs(trap_outside[idx_left:idx_right])
        # y_cut = y_outside[idx_left:idx_right]
        absy = -trap_outside
        half = np.max(np.abs(depth))/2.0
        print("half", half)
        signs = np.sign(np.add(absy, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        print("zero crossings", zero_crossings_i)
        # print("min index", min_pos_index)
        zeros = findKClosestElements(zero_crossings_i, 2, min_pos_index)
        print(zeros)
        hmx = [lin_interp(y_outside, absy, zeros[0], half),
                lin_interp(y_outside, absy, zeros[1], half)]
        fwhm = np.abs(hmx[1] - hmx[0])
        return fwhm

    def ellipticity_plot(self, projection_axis):
        if self.simul.dimension == "2D":
            projection_axis_index = set_axis_index(projection_axis)
            E_amp = np.sqrt(np.sum(abs(self.simul.Etot) ** 2, axis=0))
            E_amp3 = np.stack((E_amp, E_amp, E_amp))
            E_norm = self.simul.Etot / E_amp3
            C = np.cross(E_norm, np.conjugate(E_norm), axisa=0, axisb=0, axisc=0)
            Cz = np.imag(C[projection_axis_index])
            fig, ax = plt.subplots()
            pcm = ax.pcolormesh(
                self.simul.coord1,
                self.simul.coord2,
                np.transpose(Cz),
                cmap="twilight",
                vmin=-1,
                vmax=1,
                shading="gouraud",
            )
            plt.colorbar(pcm)
            plt.title("Ellipticity")

        elif self.simul.dimension == "1D":
            C = np.cross(self.simul.Etot, np.conjugate(self.simul.Etot), axisa=0, axisb=0, axisc=0)
            normal_axis_index = set_axis_index(set_normal_axis(self.simul.plane))
            Cz = np.imag(C[normal_axis_index])
            fig, ax = plt.subplots()
            pcm = plt.plot(self.simul.coord, Cz)
            plt.title("Ellipticity")
        return Cz

    def optimize(self, ymin=0, Pmin1=0, Pmax1=10, Pstep1=1, Pmin2=0, Pmax2=10, Pstep2=1, mf = 0):
        _, mf = check_mf(self.simul.atomicsystem.f, mf)
        mf_index = int(mf + self.simul.atomicsystem.f)
        Prange1 = list(reversed(np.arange(Pmin1, Pmax1, Pstep1)))
        Prange2 = np.arange(Pmin2, Pmax2, Pstep2)

        res_pos = np.zeros((len(Prange1), len(Prange2)))
        res_depth = np.zeros((len(Prange1), len(Prange2)))
        res_height = np.zeros((len(Prange1), len(Prange2)))
        res_freq = np.zeros((len(Prange1), len(Prange2)))

        yidx = find_nearest(self.simul.geometry.fetch_in(self.simul), ymin)
        yout = self.simul.geometry.fetch_in(self.simul)[yidx:]

        for i, P1 in progressbar_enumerate(Prange1, "\n Optimizing: ", 40):
        # for i, P1 in enumerate(Prange1):
            for j, P2 in enumerate(Prange2):
                # for i, P1 in enumerate(Prange):
                #     for j, P2 in enumerate(Prange):
                self.simul.trap.set_powers([P1, P2])
                pot = np.real(self.simul.total_potential()[0, yidx:, mf_index])
                min_idx, min_pos, depth, height, height_idx = self.get_min_trap(yout, pot)
                # sys.stdout.write("depth")
                ######### frequencies
                if np.isnan(height_idx):
                    trap_freq = 0
                else:
                    try :
                        height_pos = yout[height_idx]  ## Gives the position of the barrier
                        yleft = min_pos - (min_pos - height_pos) / 2
                        yright = min_pos + (min_pos - height_pos) / 2
                        idx_left = find_nearest(yout, yleft)
                        idx_right = find_nearest(yout, yright)
                        fit = np.polyfit(yout[idx_left:idx_right], pot[idx_left:idx_right], 2)
    
                        p = np.poly1d(fit)
                        yinterp = np.linspace(yout[0], yout[-1], 500)
                        der_fit = np.real(np.gradient(p(yinterp), yinterp))
                        der2_fit = np.gradient(der_fit, yinterp)
                        index_min = np.argmin(np.abs(yinterp - min_pos))
                        moment2 = der2_fit[index_min]
                        trap_freq = np.sqrt((moment2 * kB * mK) / (self.simul.atomicsystem.mass)) * (1 / (2 * np.pi)) / kHz
                    except :
                        trap_freq = 0
                ##################

                if abs(depth) > 10:
                    depth = 0
                if abs(height) > 10:
                    height = 0

                depth = abs(depth)
                height = abs(height)
                res_pos[i, j] = min_pos
                res_depth[i, j] = depth
                res_height[i, j] = height
                res_freq[i, j] = trap_freq

        nan_to_zeros(res_pos, res_height, res_depth, res_freq)
        res_pos *= 1e9

        return res_pos, res_depth, res_height, res_freq

    def optimize_and_show(self, ymin=0, Pmin1=0, Pmax1=10, Pstep1=1, Pmin2=0, Pmax2=10, Pstep2=1, mf = 0):
        ################################################################################################################
        ############################################ Optimization procedure ############################################
        ################################################################################################################
        # blockPrint()
        opt_pos, opt_depth, opt_height, opt_freq = self.optimize(
            ymin=ymin, Pmin1=Pmin1, Pmax1=Pmax1, Pstep1=Pstep1, Pmin2=Pmin2, Pmax2=Pmax2, Pstep2=Pstep2, mf = mf
        )
        # enablePrint()
        ################################################################################################################
        ################################################## Pretty show #################################################
        ################################################################################################################
        methods = [
            None,
            "none",
            "nearest",
            "bilinear",
            "bicubic",
            "spline16",
            "spline36",
            "hanning",
            "hamming",
            "hermite",
            "kaiser",
            "quadric",
            "catrom",
            "gaussian",
            "bessel",
            "mitchell",
            "sinc",
            "lanczos",
        ]

        color_map = "coolwarm"  # "viridis"  # "gnuplot2"

        Prange1 = list(reversed(np.arange(Pmin1, Pmax1, Pstep1)))
        Prange1 = np.asarray(Prange1)
        Prange2 = np.arange(Pmin2, Pmax2, Pstep2)

        # ax1 = plt.subplot(223)
        # im1 = ax1.imshow(
        #     opt_pos,
        #     cmap=color_map,
        #     interpolation="lanczos",
        #     extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
        #     aspect="auto",
        # )

        # plt.colorbar(im1, ax=ax1)
        # ax1.set_title("Trap position (nm)")
        # ax1.set_xlabel("P2 (mW)")
        # ax1.set_ylabel("P1 (mW)")

        #################################
        # ax4 = plt.subplot(224)
        fig = plt.figure(figsize=(3.4, 2.5))  # in inches
        ax4 = fig.add_subplot()
        im4 = ax4.imshow(
            opt_freq,
            cmap=color_map,
            interpolation = "nearest",
            # interpolation="lanczos",
            extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
            aspect="auto",
        )
        # for (j, i), label in np.ndenumerate(opt):
        #     plt.text(i, j, label, ha="center", va="center")
        cbar = plt.colorbar(im4, ax=ax4)
        cbar.ax.set_ylabel("Trap frequency (kHz)", fontsize=8)
        ax4.set_title("Trap frequency (nm)")
        ax4.set_xlabel("P2 (mW)")
        ax4.set_ylabel("P1 (mW)")

        plt.tight_layout()

        ################################

        # ax2 = plt.subplot(221)
        # im2 = ax2.imshow(
        #     opt_depth,
        #     cmap=color_map,
        #     interpolation="lanczos",
        #     extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
        #     aspect="auto",
        # )
        # plt.colorbar(im2, ax=ax2)
        # ax2.set_title("Trap depth (mK)")
        # ax2.set_xlabel("P2 (mW)")
        # ax2.set_ylabel("P1 (mW)")
        # idxs2 = np.unravel_index(opt_depth.argmax(), opt_depth.shape)
        # ax2.plot(
        #     (Prange2[idxs2[1]] + 0.5 * Pstep2) / mW,
        #     (Prange1[idxs2[0]] + 0.5 * Pstep1) / mW,
        #     "o",
        #     color="red",
        #     markersize=12,
        # )

        ##################################

        # ax3 = plt.subplot(211)
        fig = plt.figure(figsize=(3.4, 2.5))  # in inches
        ax3 = fig.add_subplot()  # plt.subplot()
        im3 = ax3.imshow(
            opt_height,
            cmap=color_map,
            interpolation="nearest",
            # interpolation="lanczos",
            extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
            aspect="auto",
            rasterized=True,
        )
        maximas = np.zeros(len(Prange1))
        for i, _ in enumerate(Prange1):
            maximas[i] = Prange2[np.argmax(opt_height[i, :])]
        max_fit = np.polyfit(maximas / mW, Prange1 / mW, 2)
        max_p = np.poly1d(max_fit)
        a = ax3.plot(maximas / mW, max_p(maximas / mW), "--", color="white", lw=3)
        cursor = mplcursors.cursor(
            a,
            highlight=True,  # , highlight_kwargs=_custom_highlight_kwargs, annotation_kwargs=_custom_annotation_kwargs
        )

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            label = artist.get_label() or ""
            x, y = sel.target
            textx = f"P2 = {x:.1f} nm"
            texty = f"P1 = {y:.2f} mK"
            text = f"{textx}\n{texty}"
            sel.annotation.set_text(text)

        # ax3.plot(maximas / mW, Prange1 / mW, "o", color="green")
        cbar = plt.colorbar(im3, ax=ax3)
        cbar.ax.set_ylabel("Trap depth (mK)", fontsize=8)
        cbar.ax.tick_params(axis="both", which="major", labelsize=8)
        # ax3.set_title("Trap height (mK)")
        ax3.set_xlabel("P2 (mW)", fontsize=8)
        ax3.set_ylabel("P1 (mW)", fontsize=8)
        ax3.tick_params(axis="both", which="major", labelsize=8)
        idxs3 = np.unravel_index(opt_height.argmax(), opt_height.shape)
        # ax3.plot(
        #     (Prange2[idxs3[1]] + 0.5 * Pstep2) / mW,
        #     (Prange1[idxs3[0]] + 0.5 * Pstep1) / mW,
        #     "o",
        #     color="red",
        #     markersize=12,
        # )

        # plt.suptitle(f"Optimal position found : P1 = {Prange1[idxs3[0]]/mW:.2f} mW, P2 = {Prange2[idxs3[1]]/mW:.2f} mW")
        plt.tight_layout(pad=0.2, h_pad=0, w_pad=0.1)
        plt.show()
        fig.savefig("optimizer_figure.pdf", dpi=600)


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        self.allowed_vals = kwargs.pop("allowed_vals", None)
        self.previous_val = kwargs["valinit"]
        Slider.__init__(self, *args, **kwargs)
        if self.allowed_vals.all() == None:
            self.allowed_vals = [self.valmin, self.valmax]
        for k in range(len(self.allowed_vals)):
            if self.orientation == "vertical":
                self.hline = self.ax.axhline(self.allowed_vals[k], 0, 1, color="r", lw=1)
            else:
                self.vline = self.ax.axvline(self.allowed_vals[k], 0, 1, color="r", lw=1)

    def set_val(self, val):
        discrete_val = self.allowed_vals[abs(val - self.allowed_vals).argmin()]
        val = discrete_val
        xy = self.poly.xy
        if self.orientation == "vertical":
            xy[1] = 0, val
            xy[2] = 1, val
        else:
            xy[2] = val, 1
            xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(val)
