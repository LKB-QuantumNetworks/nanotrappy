from nanotrappy.trapping.beam import Beam
from nanotrappy.trapping.atomicsystem import atomiclevel, atomicsystem
from nanotrappy.utils.physicalunits import *
from nanotrappy.utils.utils import *
from nanotrappy.trapping.simulator import SequentialSimulator, ParallelSimulator
from nanotrappy.trapping.geometry import *

import nanotrappy.utils.vdw as vdw
import numpy as np
from arc import *
import json
from datetime import datetime
import os, errno
from scipy import linalg as LA

from tqdm.contrib.concurrent import process_map


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


class Simulation:
    """Simulation class. Bundle of all the predefined objects that are needed for computing the potentials.

    Attributes:
        atomic system (atomicsytem): Atomic system that we want to trap (has an atom and a hyperfine level as attributes among others)
        material (material): See NanoTrap.utils.materials for available materials. They can be easily added by the user.
        trap (trap): trap object with the beams used for the specified trap scheme.
        surface (surface): Plane or Cylinder, to get a local mask for the CP interaction (always computed as -C3/r**3).
        data_folder (str): Folder where your modes with the right formatting are saved. The class will fetch the modes corresponding to the trap in this folder (if they exist).

    """

    def __init__(self, atomicsystem, material, trap, data_folder, *args):

        if not args:
            args = [vdw.NoSurface()]

        super().__init__()
        self._data_folder = data_folder
        self._atomicsystem = atomicsystem
        self._trap = trap
        self._material = material
        self._C3 = atomicsystem.get_C3(material, atomicsystem.state)
        self.surface = args

        self.geometry = AxisX(coordinates=(0, 0))
        self.simulator = SequentialSimulator()

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

    def set_simulator(self, simulator):
        self.simulator = simulator

    def set_wavelengths_indices(self):
        """Compares the wavelengths of the beams specified for the trap with the wavelengths of the available modes in the data folder and returns the list of the file indices that correspond.
        The wavelengths are rounded to 0.01 nm before comparison.

            Note:
                Passed as property so it is updated anytime the trap is changed

            Raises:
                ValueError: If at least one wavelength wanted for the trap cannot be found in the data folder.
        """
        self.wavelengths_indices = np.array([], dtype=int)
        for k,elem in enumerate(self.trap.lmbdas):
            idx = np.where(np.isclose(self.lmbdas_modes, elem, atol=1e-11))
            if len(idx[0]) == 1 :
                self.wavelengths_indices = np.append(self.wavelengths_indices, idx)
            elif len(idx[0]) >= 2 and self.trap.indices[k] is None :  
                print(self.trap.indices[k])
                files = [f for f in os.listdir(self.data_folder) if f.endswith(".npy")]

                print("Various files with corresponding wavelengths found in data folder, the possible filenames are the following: ",
                np.array(files)[idx])
                idx_chosen =int(input("Please choose filenumber you want to use (has to be integer) : "))
                real_idx = files.index(np.array(files)[idx][idx_chosen])
                self.wavelengths_indices = np.append(self.wavelengths_indices, real_idx)
            elif len(idx[0]) >= 2 and self.trap.indices[k] is not None :
                self.wavelengths_indices = np.append(self.wavelengths_indices, self.trap.indices[k])
            else:
                raise ValueError(
                    "Demanded wavelengths for trap not found in data folder, the possible wavelengths are the following: ",
                    np.sort(self.lmbdas_modes),
                )

    def set_data(self):
        self.E = []
        self.set_wavelengths_indices()
        print(self.wavelengths_indices)
        files = np.array([f for f in os.listdir(self.data_folder) if f.endswith(".npy")])
        print("[INFO] Files used for computing the trap :", files[self.wavelengths_indices])
        for filename in files[self.wavelengths_indices]:
            raw_data = np.load(self.data_folder + "//" + filename, allow_pickle=True)
            self.x = np.array(raw_data[1])
            self.y = np.array(raw_data[2])
            self.z = np.array(raw_data[3])
            self.E.append(raw_data[4])

    def get_data_dimension(self):
        return np.ndim(self.E[0]) - 1

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
                comp_keys = ["Atomic system", "Material", "Trap wavelengths", "Considered state", "Geometry", "Surface", "File names"]

                # Compare Dictionaries on certain keys using all()
                res = all(params.get(key) == self.params.get(key) for key in comp_keys)
                if res:
                    self.data_file = file[:-4] + "npy"
                    self.E_file = file[:-5] + "E_field.npy"
                    self.vecs_file = file[:-5] + "vecs.npy"
                    return True
        return False

    def inverse_propagation_direction(self, array):
        return -np.conj(array)

    def compute(self):
        mf_shift = self.mf_all + self.atomicsystem.f

        CP_masks = []
        for surface in self.surface:
            CP_masks.append(surface.set_mask(self))
        CP_masks = -self.C3 * np.array(CP_masks) / kB / mK

        self.CP = np.sum(CP_masks, axis=0)

        if self.exists_in_past_simulations():
            print("[INFO] Reusing data from file: %s" % (self.data_file))
            self.potentials = np.load(self.data_folder + "/simulations/" + self.data_file, allow_pickle=True)
            self.Etot = np.load(self.data_folder + "/simulations/" + self.E_file, allow_pickle=True)
            self.vecs = np.load(self.data_folder + "/simulations/" + self.vecs_file, allow_pickle=True)
            self.already_saved = True
            # return self.total_potential()

        else:
            print("[INFO] New simulation")

            E = FieldGeometryProcessor.restrict(self, self.geometry)

            size = E[0].shape[1:]  ## tuple with (length of axis1, length of axis2)

            self.potentials = np.zeros((len(self.trap.beams), *size, len(mf_shift)), dtype="complex")
            self.vecs = np.zeros(
                (len(self.trap.beams), *size, len(mf_shift), len(mf_shift)),
                dtype="complex",
            )  # last column contains the eigenvectors

            mf_shift = self.mf_all + self.atomicsystem.f

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.

            beam_number = 0
            potential_number = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    print("[INFO] Computing potential for a single beam...")
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    self.Etot = E[beam_number]

                    self.simulator.simulate(self, potential_number, mf_shift)

                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())
                    E_bwd = self.inverse_propagation_direction(E[beam_number + 1])
                    E_fwd = E[beam_number]
                    
                    p0 = beam.get_power()[0]
                    self.Etot = E_fwd + E_bwd*np.sqrt(beam.get_power()[1]/p0)
                    # beam.set_power(p0)
                    # sum light shifts and then diagonalize to have mean, even if frequencies are close
                    if beam.get_lmbda()[0] != beam.get_lmbda()[1]:
                        # print(
                        #     "[INFO] Computing potential for the running wave with two beams with different frequencies"
                        # )
                        self.simulator.simulate_pair(self, beam, E_fwd, E_bwd*np.sqrt(beam.get_power()[1]/p0), potential_number, mf_shift)
                        # print("[INFO] Done.")

                    else:
                        print("[INFO] Computing potential for the standing wave...")
                        self.simulator.simulate(self, potential_number, mf_shift)
                        print("[INFO] Done.")
                elif beam.isBeamSum():
                    self.atomicsystem.set_alphas(beam.get_lmbda()[0]) #Only superpositions with the same lambda are allowed !
                    Etot = []
                    for k,beam_component in enumerate(beam.beams):
                        Ek = E[k]
                        Etot += [Ek*np.sqrt(beam.get_power()[k])]

                    self.Etot = np.sum(np.array(Etot),axis = 0)
                    # print(self.Etot.shape)
                    self.simulator.simulate(self, potential_number, mf_shift)
                        
                else:
                    raise ValueError("Something bad happened. The trap should only contain Beam type objects")

                potential_number += 1
                beam_number += 1 if beam.isBeam() else 2

            self.already_saved = False
            return
            # return self.total_potential()

    def total_potential(self):
        """Uses the potentials attributes of the Simulation object for each beam to return their weighted sum with the specified powers

        Returns:
            total potential : array with shape(length coordinate 1,length coordinate 2,number of possbile mf states)
        """
        self.total_potential_noCP = np.zeros(np.shape(self.potentials[0]), dtype="float")
        self.total_vecs = np.zeros(np.shape(self.vecs[0]), dtype="complex")
        for (i, potential) in enumerate(self.potentials):
            if self.trap.beams[i].isBeamPair():
                p = self.trap.beams[i].get_power()[0]
            elif self.trap.beams[i].isBeam():
                p = self.trap.beams[i].get_power()
            elif self.trap.beams[i].isBeamSum():
                p = 1
            self.total_potential_noCP = self.total_potential_noCP + p * np.real(potential)
            self.total_vecs = p * self.vecs[i]
            
        norm_total_vecs = np.linalg.norm(self.total_vecs, axis=-1)

        number_mf_levels = self.total_vecs.shape[-1]
        for m in range(number_mf_levels):
            self.total_vecs[..., m] /= norm_total_vecs

        return np.squeeze(self.total_potential_noCP + np.dstack([self.CP] * number_mf_levels))

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
                "2D": self.geometry.dimension == 2,
                "2D plane": self.geometry.name if self.geometry.isPlane() else None,
                "2D orthogonal coord": self.geometry.normal_coord if self.geometry.isPlane() else None,
                "1D": self.geometry.dimension == 1,
                "1D axis": self.geometry.name if self.geometry.isAxis() else None,
                "1D coord1": self.geometry.coordinates[0] if self.geometry.isAxis() else None,
                "1D coord2": self.geometry.coordinates[1] if self.geometry.isAxis() else None,
            },
            "Surface": [surface.params for surface in self.surface],
            "Data_folder": self.data_folder,
            "File names": { f"{i}": file for (i, file) in enumerate(np.array([f for f in os.listdir(self.data_folder) if f.endswith(".npy")])[self.wavelengths_indices])}
        }

        # print("[INFO] Simulation parameters set")
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
            vecs_filename_data = self.data_folder + "/simulations/" + str(current_time) + "vecs.npy"

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
            np.save(vecs_filename_data, self.vecs)

            self.already_saved = True
        else:
            print("This simulation has already been saved, see %s" % (self.data_file))
