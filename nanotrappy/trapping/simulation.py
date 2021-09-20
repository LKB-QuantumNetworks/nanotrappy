from nanotrappy.trapping.beam import Beam
from nanotrappy.trapping.structures import PeriodicStructure, Structure
from nanotrappy.trapping.atomicsystem import atomiclevel, atomicsystem
from nanotrappy.utils.shiftmatrices import deltascalar, deltavector, deltatensor
from nanotrappy.utils.physicalunits import *
import nanotrappy.utils.materials as Nm
from operator import itemgetter
import legume
import numpy as np
from arc import *
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.signal import find_peaks
from matplotlib import gridspec
from nanotrappy.utils.viz import DiscreteSlider
from nanotrappy.utils.utils import *
import json
from datetime import datetime
import os, errno, time
import nanotrappy.utils.vdw as vdw
from scipy import linalg as LA


def convert_fields_to_spherical_basis(Ex, Ey, Ez):  ## returns Ep, Em, E0 (quantization axis 'z')

    return -(Ex + Ey * 1j) / np.sqrt(2), (Ex - Ey * 1j) / np.sqrt(2), Ez


def dict_sum(d1, d2):  ###not useful anymore
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) & set(d2)}


def parse_lmbdas(data_folder):
    """This function opens all the .npy file of the given data_folder and stores the wavelengths of the modes in an array.

    Args:
        data_folder (str): Absolute path of the folder containing the computed available modes.

    Returns:
        array : np array with all the available wavelengths.
    """
    lmbdas_modes = np.array([])
    files = np.array([f for f in os.listdir(data_folder) if f.endswith(".npy")])
    for filename in files:
        l = np.load(data_folder + "//" + filename, allow_pickle=True)[0]
        lmbdas_modes = np.append(lmbdas_modes, l)
    return lmbdas_modes


from PyQt5.QtCore import QFile, QThread, pyqtSignal, QObject, pyqtSlot


class Simulation(QObject):
    """Simulation class. Bundle of all the predefined objects that are needed for computing the potentials.

    Attributes:
        atomic system (atomicsytem): Atomic system that we want to trap (has an atom and a hyperfine level as attributes among others)
        material (material): See NanoTrap.utils.materials for available materials. They can be easily added by the user.
        trap (trap): trap object with the beams used for the specified trap scheme.
        surface (surface): Plane or Cylinder, to get a local mask for the CP interaction (always computed as -C3/r**3).
        data_folder (str): Folder where your modes with the right formatting are saved. The class will fetch the modes corresponding to the trap in this folder (if they exist).

    """

    _signal = pyqtSignal()
    benchmark = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self, atomicsystem, material, trap, data_folder, *args):

        if not args:
            args = vdw.NoSurface()

        super().__init__()
        self._data_folder = data_folder
        self._atomicsystem = atomicsystem
        self._trap = trap
        self._material = material
        self._C3 = atomicsystem.get_C3(material, atomicsystem.state)
        self.surface = args

        self.already_saved = False
        self.mf_all = np.arange(-atomicsystem.f, atomicsystem.f + 1)
        self.set_data()

    @property
    def data_folder(self):
        return self._data_folder

    @property
    def trap(self):
        return self._trap

    @property
    def material(self):
        return self._material

    @property
    def atomicsystem(self):
        return self._atomicsystem

    @property
    def C3(self):
        return self._C3

    @property
    def lmbdas_modes(self):
        """Returns a list of all the wavelengths available in the modes present in the data folder

        Note:
            Passed as property so it is updated anytime the trap is changed
        """
        lmbdas_modes = parse_lmbdas(self.data_folder)
        return lmbdas_modes

    def set_wavelengths_indices(self):
        """Compares the wavelengths of the beams specified for the trap with the wavelengths of the available modes in the data folder and returns the list of the file indices that correspond.
        The wavelengths are rounded to 0.01 nm before comparison.

            Note:
                Passed as property so it is updated anytime the trap is changed

            Raises:
                ValueError: If at least one wavelength wanted for the trap cannot be found in the data folder.
        """
        self.wavelengths_indices = np.array([], dtype=int)
        for elem in self.trap.lmbdas:
            idx = np.where(np.isclose(self.lmbdas_modes, elem, atol=1e-11))
            if len(idx[0]) != 0:
                self.wavelengths_indices = np.append(self.wavelengths_indices, idx)
            else:
                raise ValueError(
                    "Demanded wavelengths for trap not found in data folder, the possible wavelengths are the following: ",
                    np.sort(self.lmbdas_modes),
                )

    def set_data(self):
        self.E = []
        self.set_wavelengths_indices()
        files = np.array([f for f in os.listdir(self.data_folder) if f.endswith(".npy")])
        print("[INFO] Files used for computing the trap :", files[self.wavelengths_indices])
        for filename in files[self.wavelengths_indices]:
            raw_data = np.load(self.data_folder + "//" + filename, allow_pickle=True)
            self.x = np.array(raw_data[1])
            self.y = np.array(raw_data[2])
            self.z = np.array(raw_data[3])
            self.E.append(raw_data[4])

    def restrict2D(self, plane, coord_value):
        """This method restricts the 3D arrays of the electric fields to 2D arrays
        in a chosen plane and coordinate in the axis perpendicular to said plane.

        Args:
            plane (str): Either "XY", "YZ" or "XZ".
            coord_value (float): Coordinate of slice in the perpendicular axis (m).

        Returns:
            list : list of the electric fields for the trap in 2D arrays.
        """
        E = [np.array([]) for k in range(len(self.E))]
        if plane == "XY":
            coord3 = 3
            self.coord = self.z
        elif plane == "YZ":
            coord3 = 1
            self.coord = self.x
        elif plane == "XZ":
            coord3 = 2
            self.coord = self.y
        index_coord = np.argmin(np.abs(self.coord - coord_value))

        for j in range(len(self.E)):  # number of wavelengths
            E[j] = self.E[j].take(index_coord, axis=coord3)
        return E

    def restrict1D(self, axis, x1, x2):
        """This method restricts the 3D arrays of the electric fields to 1D arrays.

        Args:
            axis (str): Either "X", "Y" or "Z".
            x1 (float): First coordinate in the orthogonal plane. The coordinates should be given in the 'XYZ' order. If axis is 'Y', x1 would correspond to 'X', x2 to 'Z' (m).
            x2 (float): Second coordinate in the orthogonal plane (m).

        Returns:
            list : list of the electric fields for the trap in 1D arrays.
        """
        E = [np.array([]) for k in range(len(self.E))]
        if axis == "X":
            self.coord1 = self.y
            self.coord2 = self.z
            index_coord_1 = np.argmin(np.abs(self.coord1 - x1))
            index_coord_2 = np.argmin(np.abs(self.coord2 - x2))

            for j in range(len(self.E)):  # number of wavelengths
                E[j] = self.E[j][:, :, index_coord_1, index_coord_2]
            return E
        elif axis == "Y":
            self.coord1 = self.x
            self.coord2 = self.z
            index_coord_1 = np.argmin(np.abs(self.coord1 - x1))
            index_coord_2 = np.argmin(np.abs(self.coord2 - x2))

            for j in range(len(self.E)):  # number of wavelengths
                E[j] = self.E[j][:, index_coord_1, :, index_coord_2]
            return E
        elif axis == "Z":
            self.coord1 = self.x
            self.coord2 = self.y
            index_coord_1 = np.argmin(np.abs(self.coord1 - x1))
            index_coord_2 = np.argmin(np.abs(self.coord2 - x2))

            for j in range(len(self.E)):  # number of wavelengths
                E[j] = self.E[j][:, index_coord_1, index_coord_2, :]
            return E

    def exists_in_past_simulations(self):
        """This method checks if the simulation about to be run has already been saved in the "simulations" folder, by checking the .json parameter files.

        Returns:
            (tuple): tuple containing:

                - bool : True if simulation already exists, False otherwise.
                - str : Name of the file if the simulation has already been run.
        """
        self.set_current_params()
        try:
            os.listdir(self.data_folder + "/simulations")
        except FileNotFoundError:
            os.mkdir(self.data_folder + "/simulations")

        for file in os.listdir(self.data_folder + "/simulations"):
            if file.endswith(".json"):
                with open(self.data_folder + "/simulations/" + file) as json_file:
                    params = json.load(json_file)

                # initializing compare keys
                comp_keys = ["Atomic system", "Material", "Trap wavelengths", "Considered state", "Geometry", "Surface"]

                # Compare Dictionaries on certain keys using all()
                res = all(params.get(key) == self.params.get(key) for key in comp_keys)
                if res:
                    self.data_file = file[:-4] + "npy"
                    self.E_file = file[:-5] + "E_field.npy"
                    return True
        return False

    def inverse_propagation_direction(self, array):
        return -np.conj(array)

    def compute_potential(self, plane, coord):
        """Computes the potential in 2D for each Beam or BeamPair separately, for all possible mf states, and returns the total potential which is the weighted sum of both with the powers.
        If a Surface has been set when initializing the simulation, the Casimir-Polder surface interactions are included.
        Checks (with exists_in_past_simulation method) if a simulation with the same parameters has already been saved. If so, fetches the data without running a new simulation.
        If not, creates a new .json with the parameters and a new .npy with the potentials object (of shape (number of beams,length of coordinate 1,length of coordinate 2,number of possible mf states)).

        Args:
            plane (str): As the electrical fields specified are 3D, a slice has to be chosen. Either "XY", "YZ" or "XZ".
            coord (float): Coordinate of the plane on the orthogonal axis (m).

        Returns:
           array : total potential with shape(length of axis,number of possible mf states)
        """
        self.dimension = "2D"
        self.plane = plane
        self.coord = coord
        self.plane_coord1, self.plane_coord2 = None, None
        self.axis_name = None
        self.coord1, self.coord2 = set_axis_from_plane(plane, self)

        CP_mask = []
        for surface in self.surface:
            CP_mask.append(surface.set_mask2D(self, plane, coord))
        CP_mask = -self.C3 * np.array(CP_mask) / kB / mK

        self.CP = np.zeros((CP_mask.shape[-2], CP_mask.shape[-1], len(self.mf_all)))
        for k in range(len(self.mf_all)):
            self.CP[:, :, k] = np.sum(CP_mask, axis=0)
        mf_shift = self.mf_all + self.atomicsystem.f

        if self.exists_in_past_simulations():
            print("[INFO] Reusing data from file: %s" % (self.data_file))
            self.potentials = np.load(self.data_folder + "/simulations/" + self.data_file, allow_pickle=True)
            self.Etot = np.load(self.data_folder + "/simulations/" + self.E_file, allow_pickle=True)
            self.already_saved = True
            return self.total_potential()
        else:
            print("[INFO] New simulation")
            if np.ndim(self.E[0]) == 4:
                E = self.restrict2D(plane, coord)
            else:
                E = self.E.copy()

            self.potentials = np.zeros(
                (len(self.trap.beams), len(self.coord1), len(self.coord2), len(mf_shift)), dtype="complex"
            )
            self.vecs = np.zeros(
                (len(self.trap.beams), len(self.coord1), len(self.coord2), len(mf_shift), len(mf_shift)),
                dtype="complex",
            )  # last column contains the eigenvectors

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.

            beam_number = 0
            potential_number = 0
            run = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    print("[INFO] Computing potential for a single beam...")
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    self.Etot = E[beam_number]
                    for i in progressbar(range(len(self.coord1)), "\n Computing: ", 40):
                        for j in range(len(self.coord2)):
                            Ep, Em, E0 = convert_fields_to_spherical_basis(
                                self.Etot[0, i, j], self.Etot[1, i, j], self.Etot[2, i, j]
                            )
                            val, vec = self.atomicsystem.potential(Ep, Em, E0)
                            self.potentials[potential_number, i, j, :] = np.real(val[mf_shift])
                            for id, elt in enumerate(vec):
                                self.vecs[potential_number, i, j, id, :] = elt

                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())

                    # sum light shifts and then diagonalize
                    if beam.get_lmbda()[0] != beam.get_lmbda()[1]:
                        print(
                            "[INFO] Computing potential for the running wave with two beams with different frequencies"
                        )
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        E_fwd = E[beam_number]
                        alpha0_f, alpha1_f, alpha2_f = self.atomicsystem.set_alphas(beam.get_lmbda()[0])
                        alpha0_b, alpha1_b, alpha2_b = self.atomicsystem.set_alphas(beam.get_lmbda()[1])
                        self.Etot = E_fwd + E_bwd

                        for i in progressbar(range(len(self.coord1)), "\n Computing: ", 40):
                            for j in range(len(self.coord2)):
                                Ep, Em, E0 = convert_fields_to_spherical_basis(
                                    E_fwd[0, i, j], E_fwd[1, i, j], E_fwd[2, i, j]
                                )
                                tot_shift_f = (
                                    -1.0 * alpha0_f * self.atomicsystem.deltascalar(Ep, Em, E0)
                                    - 1.0 * alpha1_f * self.atomicsystem.deltavector(Ep, Em, E0)
                                    - 1.0 * alpha2_f * self.atomicsystem.deltatensor(Ep, Em, E0)
                                )

                                Ep, Em, E0 = convert_fields_to_spherical_basis(
                                    E_bwd[0, i, j], E_bwd[1, i, j], E_bwd[2, i, j]
                                )
                                tot_shift_b = (
                                    -1.0 * alpha0_b * self.atomicsystem.deltascalar(Ep, Em, E0)
                                    - 1.0 * alpha1_b * self.atomicsystem.deltavector(Ep, Em, E0)
                                    - 1.0 * alpha2_b * self.atomicsystem.deltatensor(Ep, Em, E0)
                                )

                                tot_shift = (tot_shift_f + tot_shift_b) / (mK * kB)
                                vals, vec = LA.eig(tot_shift)
                                idx = vals.argsort()
                                self.potentials[potential_number, i, j, :] = 0.5 * np.real(vals[idx][mf_shift])
                                for id, elt in enumerate(vec[idx]):
                                    self.vecs[potential_number, i, j, id, :] = elt
                        print("[INFO] Done.")

                    else:
                        print("[INFO] Computing potential for the standing wave...")
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        self.Etot = E[beam_number] + E_bwd

                        for i in progressbar(range(len(self.coord1)), "\n Computing: ", 40):
                            for j in range(len(self.coord2)):
                                Ep, Em, E0 = convert_fields_to_spherical_basis(
                                    self.Etot[0, i, j], self.Etot[1, i, j], self.Etot[2, i, j]
                                )
                                val, vec = self.atomicsystem.potential(Ep, Em, E0)
                                self.potentials[potential_number, i, j, :] = 0.5 * np.real(val[mf_shift])
                                for id, elt in enumerate(vec):
                                    self.vecs[potential_number, i, j, id, :] = elt
                        print("[INFO] Done.")
                else:
                    raise ValueError("Something bad happened. The trap should only contain Beam type objects")

                potential_number += 1
                beam_number += 1 if beam.isBeam() else 2

            self.already_saved = False
            return self.total_potential()

    @pyqtSlot()
    def compute_potential_gui(self, plane, coord):
        """Computes the potential in 2D for each Beam or BeamPair separately, for all possible mf states, and returns the total potential which is the weighted sum of both with the powers.
        If a Surface has been set when initializing the simulation, the Casimir-Polder surface interactions are included.
        Checks (with exists_in_past_simulation method) if a simulation with the same parameters has already been saved. If so, fetches the data without running a new simulation.
        If not, creates a new .json with the parameters and a new .npy with the potentials object (of shape (number of beams,length of coordinate 1,length of coordinate 2,number of possible mf states)).

        Args:
            plane (str): As the electrical fields specified are 3D, a slice has to be chosen. Either "XY", "YZ" or "XZ".
            coord (float): Coordinate of the plane on the orthogonal axis (m).

        Returns:
           array : total potential with shape(length of axis,number of possible mf states)
        """
        self.dimension = "2D"
        self.plane = plane
        self.coord = coord
        self.plane_coord1, self.plane_coord2 = None, None
        self.axis_name = None
        self.coord1, self.coord2 = set_axis_from_plane(plane, self)

        CP_mask = []
        for surface in self.surface:
            CP_mask.append(surface.set_mask2D(self, plane, coord))
        CP_mask = -self.C3 * np.array(CP_mask) / kB / mK

        self.CP = np.zeros((CP_mask.shape[-2], CP_mask.shape[-1], len(self.mf_all)))
        for k in range(len(self.mf_all)):
            self.CP[:, :, k] = np.sum(CP_mask, axis=0)
        mf_shift = self.mf_all + self.atomicsystem.f

        if self.exists_in_past_simulations():
            print("[INFO] Reusing data from file: %s" % (self.data_file))
            self.potentials = np.load(self.data_folder + "/simulations/" + self.data_file, allow_pickle=True)
            self.Etot = np.load(self.data_folder + "/simulations/" + self.E_file, allow_pickle=True)
            self.already_saved = True
            self.finished.emit()
            return self.total_potential()
        else:
            print("[INFO] New simulation")
            if np.ndim(self.E[0]) == 4:
                E = self.restrict2D(plane, coord)
            else:
                E = self.E.copy()

            self.potentials = np.zeros(
                (len(self.trap.beams), len(self.coord1), len(self.coord2), len(mf_shift)), dtype="complex"
            )

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.

            beam_number = 0
            potential_number = 0
            run = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    print("[INFO] Computing potential for a single beam...")
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    self.Etot = E[beam_number]

                    t0 = time.time()
                    for i in range(len(self.coord1)):
                        for j in range(len(self.coord2)):
                            Ep, Em, E0 = convert_fields_to_spherical_basis(
                                self.Etot[0, i, j], self.Etot[1, i, j], self.Etot[2, i, j]
                            )
                            val, _ = self.atomicsystem.potential(Ep, Em, E0)
                            self.potentials[potential_number, i, j, :] = np.real(val[mf_shift])
                        self._signal.emit()
                        if i == 10:
                            if beam_number == 0:
                                self.benchmark.emit(time.time() - t0)

                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())

                    # sum light shifts and then diagonalize
                    if beam.get_lmbda()[0] != beam.get_lmbda()[1]:
                        print(
                            "[INFO] Computing potential for the running wave with two beams with different frequencies"
                        )
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        E_fwd = E[beam_number]
                        alpha0_f, alpha1_f, alpha2_f = self.atomicsystem.set_alphas(beam.get_lmbda()[0])
                        alpha0_b, alpha1_b, alpha2_b = self.atomicsystem.set_alphas(beam.get_lmbda()[1])
                        self.Etot = E_fwd + E_bwd

                        t0 = time.time()
                        for i in range(len(self.coord1)):
                            for j in range(len(self.coord2)):
                                Ep, Em, E0 = convert_fields_to_spherical_basis(
                                    E_fwd[0, i, j], E_fwd[1, i, j], E_fwd[2, i, j]
                                )
                                tot_shift_f = (
                                    -1.0 * alpha0_f * deltascalar(Ep, Em, E0, self.atomicsystem.f)
                                    - 1.0 * alpha1_f * deltavector(Ep, Em, E0, self.atomicsystem.f)
                                    - 1.0 * alpha2_f * deltatensor(Ep, Em, E0, self.atomicsystem.f)
                                )

                                Ep, Em, E0 = convert_fields_to_spherical_basis(
                                    E_bwd[0, i, j], E_bwd[1, i, j], E_bwd[2, i, j]
                                )
                                tot_shift_b = (
                                    -1.0 * alpha0_b * deltascalar(Ep, Em, E0, self.atomicsystem.f)
                                    - 1.0 * alpha1_b * deltavector(Ep, Em, E0, self.atomicsystem.f)
                                    - 1.0 * alpha2_b * deltatensor(Ep, Em, E0, self.atomicsystem.f)
                                )

                                tot_shift = (tot_shift_f + tot_shift_b) / (mK * kB)
                                vals, vec = LA.eig(tot_shift)
                                idx = vals.argsort()
                                self.potentials[potential_number, i, j, :] = 0.5 * np.real(vals[idx][mf_shift])
                            self._signal.emit()
                            if i == 10:
                                if beam_number == 0:
                                    self.benchmark.emit(time.time() - t0)
                        print("[INFO] Done.")

                    else:
                        print("[INFO] Computing potential for the standing wave...")
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        self.Etot = E[beam_number] + E_bwd

                        t0 = time.time()
                        for i in range(len(self.coord1)):
                            for j in range(len(self.coord2)):
                                Ep, Em, E0 = convert_fields_to_spherical_basis(
                                    self.Etot[0, i, j], self.Etot[1, i, j], self.Etot[2, i, j]
                                )
                                val, _ = self.atomicsystem.potential(Ep, Em, E0)
                                self.potentials[potential_number, i, j, :] = 0.5 * np.real(val[mf_shift])
                            self._signal.emit()
                            if i == 10:
                                if beam_number == 0:
                                    self.benchmark.emit(time.time() - t0)
                        print("[INFO] Done.")
                else:
                    raise ValueError("Something bad happened. The trap should only contain Beam type objects")

                potential_number += 1
                beam_number += 1 if beam.isBeam() else 2

            self.already_saved = False
            self.finished.emit()
            return self.total_potential()

    @pyqtSlot()
    def compute_potential_gui2(self, plane, coord):
        """Computes the potential in 2D for each Beam or BeamPair separately, for all possible mf states, and returns the total potential which is the weighted sum of both with the powers.
        If a Surface has been set when initializing the simulation, the Casimir-Polder surface interactions are included.
        Checks (with exists_in_past_simulation method) if a simulation with the same parameters has already been saved. If so, fetches the data without running a new simulation.
        If not, creates a new .json with the parameters and a new .npy with the potentials object (of shape (number of beams,length of coordinate 1,length of coordinate 2,number of possible mf states)).

        Args:
            plane (str): As the electrical fields specified are 3D, a slice has to be chosen. Either "XY", "YZ" or "XZ".
            coord (float): Coordinate of the plane on the orthogonal axis (m).

        Returns:
           array : total potential with shape(length of axis,number of possible mf states)
        """
        self.dimension = "2D"
        self.plane = plane
        self.coord = coord
        self.plane_coord1, self.plane_coord2 = None, None
        self.axis_name = None
        self.coord1, self.coord2 = set_axis_from_plane(plane, self)

        CP_mask = []
        for surface in self.surface:
            CP_mask.append(surface.set_mask2D(self, plane, coord))
        CP_mask = -self.C3 * np.array(CP_mask) / kB / mK

        self.CP = np.zeros((CP_mask.shape[-2], CP_mask.shape[-1], len(self.mf_all)))
        for k in range(len(self.mf_all)):
            self.CP[:, :, k] = np.sum(CP_mask, axis=0)
        mf_shift = self.mf_all + self.atomicsystem.f

        if self.exists_in_past_simulations():
            print("[INFO] Reusing data from file: %s" % (self.data_file))
            self.potentials = np.load(self.data_folder + "/simulations/" + self.data_file, allow_pickle=True)
            self.Etot = np.load(self.data_folder + "/simulations/" + self.E_file, allow_pickle=True)
            self.already_saved = True
            self._signal.emit()
            self.finished.emit()
            return self.total_potential()
        else:
            print("[INFO] New simulation")
            if np.ndim(self.E[0]) == 4:
                E = self.restrict2D(plane, coord)
            else:
                E = self.E.copy()

            self.potentials = np.zeros(
                (len(self.trap.beams), len(self.coord1), len(self.coord2), len(mf_shift)), dtype="complex"
            )

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.

            beam_number = 0
            potential_number = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    print("[INFO] Computing potential for a single beam...")
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    self.Etot = E[beam_number]
                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())
                    E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                    # fig,ax = plt.subplots()
                    # plt.imshow(np.imag(E_bwd[2,:,:]))
                    self.Etot = E[beam_number] + E_bwd

                    # fig,ax = plt.subplots()
                    # plt.imshow(np.imag(Etot[2,:,:]))
                    # plt.colorbar()

                else:
                    raise ValueError("Something bad happened. The trap should only contain Beam type objects")

                t0 = time.time()
                for i in range(len(self.coord1)):
                    for j in range(len(self.coord2)):
                        Ep, Em, E0 = convert_fields_to_spherical_basis(
                            self.Etot[0, i, j], self.Etot[1, i, j], self.Etot[2, i, j]
                        )
                        val, _ = self.atomicsystem.potential(Ep, Em, E0)
                        self.potentials[potential_number, i, j, :] = np.real(val[mf_shift])
                    self._signal.emit()
                    if i == 10:
                        if beam_number == 0:
                            self.benchmark.emit(time.time() - t0)
                potential_number += 1
                beam_number += 1 if beam.isBeam() else 2

            self.already_saved = False
            self.finished.emit()
            return self.total_potential()

    def total_potential(self):
        """Uses the potentials attributes of the Simulation object for each beam to return their weighted sum with the specified powers

        Returns:
            total potential : array with shape(length coordinate 1,length coordinate 2,number of possbile mf states)
        """
        self.total_potential_noCP = np.zeros(np.shape(self.potentials[0]), dtype="float")
        self.total_vecs = np.zeros(np.shape(self.vecs[0]), dtype="complex")
        for (i, potential) in enumerate(self.potentials):
            # if self.trap.beams[i].__class__.__name__ == "Beam" :
            if isinstance(self.trap.beams[i], Beam):
                mult = 1
            else:
                mult = 2
            self.total_potential_noCP = self.total_potential_noCP + mult * self.trap.beams[i].get_power() * potential
            self.total_vecs = mult * self.trap.beams[i].get_power() * self.vecs[i]

        norm_total_vecs = np.linalg.norm(self.total_vecs, axis=-1)
        for m in range(self.total_vecs.shape[-1]):
            self.total_vecs[..., m] /= norm_total_vecs

        return self.total_potential_noCP + self.CP

    def compute_potential_1D(self, axis, coord1, coord2, contrib="scalar"):
        """Computes the potential in 2D for each Beam or BeamPair separately, for all possible mf states, and returns the total potential which is the weighted sum of both with the powers.
        If a Surface has been set when initializing the simulation, the Casimir-Polder surface interactions are included.
        Checks (with exists_in_past_simulation method) if a simulation with the same parameters has already been saved. If so, fetches the data without running a new simulation.
        If not, creates a new .json with the parameters and a new .npy with the potentials object (of shape (number of beams,length of axis,number of possible mf states)).

        Args:
            axis (str): As the electrical fields specified are 3D, an axis of observation has to be specified. Either "X", "Y" or "Z".
            coord1 (float): First coordinate of the axis position in the orthogonal plane. (m)
            coord2 (float): Second coordinate of the axis position in the orthogonal plane. (m)

        Returns:
           array : total potential with shape(length of axis,number of possible mf states)
        """
        self.dimension = "1D"
        self.coord = None
        self.plane = None
        self.plane_coord1, self.plane_coord2 = coord1, coord2
        self.axis_name = axis
        self.axis = set_axis_from_axis(axis, self)

        mf_shift = self.mf_all + self.atomicsystem.f

        CP_mask = []
        for surface in self.surface:
            CP_mask.append(surface.set_mask1D(self, axis, self.plane_coord1, self.plane_coord2))
        CP_mask = -self.C3 * np.array(CP_mask) / kB / mK

        self.CP = np.array(
            [
                np.sum(CP_mask, axis=0),
            ]
            * len(self.mf_all)
        ).transpose()

        if self.exists_in_past_simulations():
            print("[INFO] Reusing data from file: %s" % (self.data_file))
            self.potentials = np.load(self.data_folder + "/simulations/" + self.data_file, allow_pickle=True)
            self.Etot = np.load(self.data_folder + "/simulations/" + self.E_file, allow_pickle=True)
            self.already_saved = True
            return self.total_potential()
        else:
            print("[INFO] New simulation")
            if np.ndim(self.E[0]) == 4:
                E = self.restrict1D(self.axis_name, coord1, coord2)
            elif np.ndim(self.E[0]) == 3:
                raise ValueError(
                    "You must give the electric fields in all 3 directions (one dimension can be of size 1)"
                )
            else:
                E = self.E.copy()

            self.potentials = np.zeros((len(self.trap.beams), len(self.axis), len(mf_shift)), dtype="complex")
            self.vecs = np.zeros((len(self.trap.beams), len(self.axis), len(mf_shift), len(mf_shift)), dtype="complex")

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.

            beam_number = 0
            potential_number = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    self.Etot = E[beam_number]
                    for i in range(len(self.axis)):
                        Ep, Em, E0 = convert_fields_to_spherical_basis(
                            self.Etot[0, i], self.Etot[1, i], self.Etot[2, i]
                        )
                        val, vec = self.atomicsystem.potential(Ep, Em, E0)
                        # val, _ = self.atomicsystem.potential_partial(Ep,Em,E0,contrib=contrib)
                        self.potentials[potential_number, i, :] = np.real(val[mf_shift])
                        for id, elt in enumerate(vec):
                            self.vecs[potential_number, i, id, :] = elt

                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())
                    E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                    self.Etot = E[beam_number] + E_bwd

                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())

                    if beam.get_lmbda()[0] != beam.get_lmbda()[1]:
                        print(
                            "[INFO] Computing potential for the running wave with two beams with different frequencies"
                        )
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        E_fwd = E[beam_number]
                        alpha0_f, alpha1_f, alpha2_f = self.atomicsystem.set_alphas(beam.get_lmbda()[0])
                        alpha0_b, alpha1_b, alpha2_b = self.atomicsystem.set_alphas(beam.get_lmbda()[1])
                        self.Etot = E_fwd + E_bwd
                        for i in range(len(self.axis)):
                            Ep, Em, E0 = convert_fields_to_spherical_basis(E_fwd[0, i], E_fwd[1, i], E_fwd[2, i])
                            tot_shift_f = (
                                -1.0 * alpha0_f * self.atomicsystem.deltascalar(Ep, Em, E0)
                                - 1.0 * alpha1_f * self.atomicsystem.deltavector(Ep, Em, E0)
                                - 1.0 * alpha2_f * self.atomicsystem.deltatensor(Ep, Em, E0)
                            )
                            Ep, Em, E0 = convert_fields_to_spherical_basis(E_bwd[0, i], E_bwd[1, i], E_bwd[2, i])
                            tot_shift_b = (
                                -1.0 * alpha0_b * self.atomicsystem.deltascalar(Ep, Em, E0)
                                - 1.0 * alpha1_b * self.atomicsystem.deltavector(Ep, Em, E0)
                                - 1.0 * alpha2_b * self.atomicsystem.deltatensor(Ep, Em, E0)
                            )

                            tot_shift = (tot_shift_f + tot_shift_b) / (mK * kB)
                            vals, vec = LA.eig(tot_shift)
                            idx = vals.argsort()
                            self.potentials[potential_number, i, :] = 0.5 * np.real(vals[idx][mf_shift])
                            for id, elt in enumerate(vec[idx]):
                                self.vecs[potential_number, i, id, :] = elt
                        print("[INFO] Done.")

                    else:
                        print("[INFO] Computing potential for the standing wave...")
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        self.Etot = E[beam_number] + E_bwd
                        for i in range(len(self.axis)):
                            Ep, Em, E0 = convert_fields_to_spherical_basis(
                                self.Etot[0, i], self.Etot[1, i], self.Etot[2, i]
                            )
                            val, _ = self.atomicsystem.potential(Ep, Em, E0)
                            self.potentials[potential_number, i, :] = 0.5 * np.real(val[mf_shift])
                            for id, elt in enumerate(vec):
                                self.vecs[potential_number, i, id, :] = elt
                        print("[INFO] Done.")
                else:
                    raise ValueError("Something bad happened. The trap should only contain Beam type objects")

                potential_number += 1
                beam_number += 1 if beam.isBeam() else 2

            self.already_saved = False
            # self.finished.emit()
            return self.total_potential()

    @pyqtSlot()
    def compute_potential_1D_gui(self, axis, coord1, coord2, contrib="scalar"):
        """Computes the potential in 2D for each Beam or BeamPair separately, for all possible mf states, and returns the total potential which is the weighted sum of both with the powers.
        If a Surface has been set when initializing the simulation, the Casimir-Polder surface interactions are included.
        Checks (with exists_in_past_simulation method) if a simulation with the same parameters has already been saved. If so, fetches the data without running a new simulation.
        If not, creates a new .json with the parameters and a new .npy with the potentials object (of shape (number of beams,length of axis,number of possible mf states)).

        Args:
            axis (str): As the electrical fields specified are 3D, an axis of observation has to be specified. Either "X", "Y" or "Z".
            coord1 (float): First coordinate of the axis position in the orthogonal plane. (m)
            coord2 (float): Second coordinate of the axis position in the orthogonal plane. (m)

        Returns:
           array : total potential with shape(length of axis,number of possible mf states)
        """
        self.dimension = "1D"
        self.coord = None
        self.plane = None
        self.plane_coord1, self.plane_coord2 = coord1, coord2
        self.axis_name = axis
        self.axis = set_axis_from_axis(axis, self)

        mf_shift = self.mf_all + self.atomicsystem.f

        CP_mask = []
        for surface in self.surface:
            CP_mask.append(surface.set_mask1D(self, axis, self.plane_coord1, self.plane_coord2))
        CP_mask = -self.C3 * np.array(CP_mask) / kB / mK

        self.CP = np.array(
            [
                np.sum(CP_mask, axis=0),
            ]
            * len(self.mf_all)
        ).transpose()

        if self.exists_in_past_simulations():
            print("[INFO] Reusing data from file: %s" % (self.data_file))
            self.potentials = np.load(self.data_folder + "/simulations/" + self.data_file, allow_pickle=True)
            self.Etot = np.load(self.data_folder + "/simulations/" + self.E_file, allow_pickle=True)
            self.already_saved = True
            self.finished.emit()
            return self.total_potential()
        else:
            print("[INFO] New simulation")
            if np.ndim(self.E[0]) == 4:
                E = self.restrict1D(self.axis_name, coord1, coord2)
            elif np.ndim(self.E[0]) == 3:
                raise ValueError(
                    "You must give the electric fields in all 3 directions (one dimension can be of size 1)"
                )
            else:
                E = self.E.copy()

            self.potentials = np.zeros((len(self.trap.beams), len(self.axis), len(mf_shift)), dtype="complex")

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.

            beam_number = 0
            potential_number = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    self.Etot = E[beam_number]

                    t0 = time.time()
                    for i in range(len(self.axis)):
                        Ep, Em, E0 = convert_fields_to_spherical_basis(
                            self.Etot[0, i], self.Etot[1, i], self.Etot[2, i]
                        )
                        # val, vec = self.atomicsystem.potential(Ep, Em, E0)
                        val, _ = self.atomicsystem.potential_partial(Ep, Em, E0)
                        self.potentials[potential_number, i, :] = np.real(val[mf_shift])
                    self._signal.emit()
                    if i == 10:
                        if beam_number == 0:
                            self.benchmark.emit(time.time() - t0)

                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())
                    E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                    self.Etot = E[beam_number] + E_bwd

                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())

                    if beam.get_lmbda()[0] != beam.get_lmbda()[1]:
                        print(
                            "[INFO] Computing potential for the running wave with two beams with different frequencies"
                        )
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        E_fwd = E[beam_number]
                        alpha0_f, alpha1_f, alpha2_f = self.atomicsystem.set_alphas(beam.get_lmbda()[0])
                        alpha0_b, alpha1_b, alpha2_b = self.atomicsystem.set_alphas(beam.get_lmbda()[1])
                        self.Etot = E_fwd + E_bwd

                        t0 = time.time()
                        for i in range(len(self.axis)):
                            Ep, Em, E0 = convert_fields_to_spherical_basis(E_fwd[0, i], E_fwd[1, i], E_fwd[2, i])
                            tot_shift_f = (
                                -1.0 * alpha0_f * self.atomicsystem.deltascalar(Ep, Em, E0)
                                - 1.0 * alpha1_f * self.atomicsystem.deltavector(Ep, Em, E0)
                                - 1.0 * alpha2_f * self.atomicsystem.deltatensor(Ep, Em, E0)
                            )
                            Ep, Em, E0 = convert_fields_to_spherical_basis(E_bwd[0, i], E_bwd[1, i], E_bwd[2, i])
                            tot_shift_b = (
                                -1.0 * alpha0_b * self.atomicsystem.deltascalar(Ep, Em, E0)
                                - 1.0 * alpha1_b * self.atomicsystem.deltavector(Ep, Em, E0)
                                - 1.0 * alpha2_b * self.atomicsystem.deltatensor(Ep, Em, E0)
                            )

                            tot_shift = (tot_shift_f + tot_shift_b) / (mK * kB)
                            vals, vec = LA.eig(tot_shift)
                            idx = vals.argsort()
                            self.potentials[potential_number, i, :] = 0.5 * np.real(vals[idx][mf_shift])
                        self._signal.emit()
                        if i == 10:
                            if beam_number == 0:
                                self.benchmark.emit(time.time() - t0)
                        print("[INFO] Done.")

                    else:
                        print("[INFO] Computing potential for the standing wave...")
                        E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                        self.Etot = E[beam_number] + E_bwd

                        t0 = time.time()
                        for i in range(len(self.axis)):
                            Ep, Em, E0 = convert_fields_to_spherical_basis(
                                self.Etot[0, i], self.Etot[1, i], self.Etot[2, i]
                            )
                            val, _ = self.atomicsystem.potential(Ep, Em, E0)
                            self.potentials[potential_number, i, :] = 0.5 * np.real(val[mf_shift])
                        self._signal.emit()
                        if i == 10:
                            if beam_number == 0:
                                self.benchmark.emit(time.time() - t0)
                        print("[INFO] Done.")
                else:
                    raise ValueError("Something bad happened. The trap should only contain Beam type objects")

                potential_number += 1
                beam_number += 1 if beam.isBeam() else 2

            self.already_saved = False
            self.finished.emit()
            return self.total_potential()

    def set_current_params(self):
        """Sets a dictionnary with all the relevant parameters of the Simulation object and returns it.

        Returns:
            dict : parameters of the simulation
        """

        self.lmbdas_params = np.array([])
        self.P_params = np.array([])

        for (k, beam) in enumerate(self.trap.beams):
            self.lmbdas_params = np.append(self.lmbdas_params, beam.get_lmbda())
            self.P_params = np.append(self.P_params, beam.get_power())
            self.lmbdas_params.resize(2 * (k + 1), refcheck=False)
            self.P_params.resize(2 * (k + 1), refcheck=False)

        self.lmbdas_params.resize(4, refcheck=False)
        self.P_params.resize(4, refcheck=False)

        lambda1pair1 = str(self.lmbdas_params[0] / nm) + " nm"
        power1pair1 = str(self.P_params[0] / mW) + " mW"
        lambda2pair1 = str(self.lmbdas_params[1] / nm) + " nm"
        power2pair1 = str(self.P_params[1] / mW) + " mW"

        lambda1pair2 = str(self.lmbdas_params[2] / nm) + " nm"
        power1pair2 = str(self.P_params[2] / mW) + " mW"
        lambda2pair2 = str(self.lmbdas_params[3] / nm) + " nm"
        power2pair2 = str(self.P_params[3] / mW) + " mW"
        now = datetime.now()
        current_time = now.strftime("%d/%m/%Y %H:%M:%S")

        self.params = {
            "Time of simulation": current_time,
            "Atomic system": {
                "species": type(self.atomicsystem.atom).__name__,
                "groundstate": str(self.atomicsystem.groundstate),
                "hyperfine level": int(self.atomicsystem.f),
            },
            "Material": str(self.material),
            "Trap wavelengths": {
                "lambda 1 pair 1": lambda1pair1,
                "lambda 2 pair 1": lambda2pair1,
                "lambda 1 pair 2": lambda1pair2,
                "lambda 2 pair 2": lambda2pair2,
            },
            "Trap powers": {
                "power 1 pair 1": power1pair1,
                "power 2 pair 1": power2pair1,
                "power 1 pair 2": power1pair2,
                "power 2 pair 2": power2pair2,
            },
            "Considered state": str(self.atomicsystem.state),
            "Geometry": {
                "2D": self.dimension == "2D",
                "2D plane": self.plane,
                "2D orthogonal coord": self.coord,
                "1D": self.dimension == "1D",
                "1D axis": self.axis_name,
                "1D coord1": self.plane_coord1,
                "1D coord2": self.plane_coord2,
            },
            "Surface": [surface.params for surface in self.surface],
            "Data_folder": self.data_folder,
        }

        print("[INFO] Simulation parameters set")
        return self.params

    def save(self):
        """Saves both the parameters dictionnary into a .json file and the potentials attribute into a .npy."""
        if not self.already_saved:

            current_time = self.params["Time of simulation"]
            current_time = current_time.replace("/", "_")
            current_time = current_time.replace(":", "_")
            current_time = current_time.replace(" ", "_")

            filename_params = self.data_folder + "/simulations/" + str(current_time) + ".json"
            filename_data = self.data_folder + "/simulations/" + str(current_time) + ".npy"
            E_filename_data = self.data_folder + "/simulations/" + str(current_time) + "E_field.npy"

            if not os.path.exists(os.path.dirname(filename_params)):
                try:
                    os.makedirs(os.path.dirname(filename_params))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            with open(filename_params, "w") as fp:
                json.dump(self.params, fp)

            np.save(filename_data, self.potentials)
            np.save(E_filename_data, self.Etot)

            self.already_saved = True
        else:
            print("This simulation has already been saved, see %s" % (self.data_file))
