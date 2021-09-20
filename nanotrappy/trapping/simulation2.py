from nanotrappy.trapping.structures import PeriodicStructure, Structure
from nanotrappy.trapping.atomicsystem import atomiclevel, atomicsystem
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
import os, errno
import nanotrappy.utils.vdw as vdw
import multiprocessing
import itertools

def process(*args):
    i,j, atomicsystem, Etot, mf_shift = args[0]
    E0, Ep, Em = convert_fields_to_spherical_basis(Etot[0,i,j],Etot[1,i,j],Etot[2,i,j])
    val, _ = atomicsystem.potential(Ep,Em,E0)
    return val
    #self.potentials[potential_number,i,j,:] = np.real(val[mf_shift])

def convert_fields_to_spherical_basis(Ex,Ey,Ez): ## returns Ep, Em, E0 (quantization axis 'z')
    
    return -(Ex+Ey*1j)/np.sqrt(2), (Ex-Ey*1j)/np.sqrt(2), Ez

def dict_sum(d1,d2): ###not useful anymore
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) & set(d2)}
    
def parse_lmbdas(data_folder):
    """This function opens all the .npy file of the given data_folder and stores the wavelengths of the modes in an array.
    
    Args:
        data_folder (str): Absolute path of the folder containing the computed available modes.
    
    Returns:
        array : np array with all the available wavelengths.
    """
    lmbdas_modes = np.array([])
    files = np.array([f for f in os.listdir(data_folder) if f.endswith('.npy')])
    # for filename in os.listdir(data_folder):
    #     if filename.endswith(".npy") :
    for filename in files:
        l = np.load(data_folder + "//" + filename,allow_pickle = True)[0]
        lmbdas_modes = np.append(lmbdas_modes,l)
    return lmbdas_modes
    
def inverse_propagation_direction(array, axis):
    """Simulates a counterpropagating beam by reversing the component of the field corresponding to the axis of propagation, leaving everything else unchanged.
    
    Args:
        array (str): Electric field array with first coordinate being the 3 components Ex, Ey and Ez.
        axis (str): Propagation axis. Either 'X', 'Y' or 'Z'.
    
    Returns:
        array : Electric field array of shape
    """
    propagation_axis = set_axis_index(axis)
    inversed_array = array.copy()
    inversed_array[propagation_axis,...] = -1*array[propagation_axis,...]
    return inversed_array


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
    _signal = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self,atomicsystem,material,trap, data_folder,surface = vdw.NoSurface()):
        
        super().__init__()
        self._data_folder = data_folder
        self._atomicsystem = atomicsystem
        self._trap = trap
        self._material = material
        self._C3 = atomicsystem.get_C3(material,atomicsystem.state)
        self.surface = surface
        
        self.already_saved = False
        self.mf_all = np.arange(-atomicsystem.f,atomicsystem.f+1)
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

    # def get_trap(self):
    #     return self._trap

    # def set_trap(self, trap):
    #     self._trap = trap
    #     self.wavelength_indices = self.set_wavelength_indices(self._trap)
    #     self.data = self.set_data()
    #     self.x = self.set_x()
    #     self.y = self.set_y()
    #     self.z = self.set_z()
    #     self.E = self.set_E()
        
    # trap = property(get_trap,set_trap,None,"trap")

    # def get_datafolder(self):
    #     return self._trap

    # def set_datafolder(self, state):

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
        self.wavelengths_indices = np.array([],dtype = int)
        
        for elem in self.trap.lmbdas:
            idx = np.where(np.isclose(self.lmbdas_modes, elem, atol = 1e-11))
            if len(idx[0]) != 0:
                self.wavelengths_indices = np.append(self.wavelengths_indices,idx)
            else:
                raise ValueError('Demanded wavelengths for trap not found in data folder')

    def set_data(self):
        self.E = []
        self.set_wavelengths_indices()
        # for filename in np.array(os.listdir(self.data_folder))[self.wavelengths_indices]:
        #     if filename.endswith(".npy") :
        files = np.array([f for f in os.listdir(self.data_folder) if f.endswith('.npy')])
        for filename in files[self.wavelengths_indices]:
            raw_data =  np.load(self.data_folder + "//" + filename,allow_pickle = True) 
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
         
        for j in range(len(self.E)):#number of wavelengths           
            E[j] = self.E[j].take(index_coord, axis = coord3)              
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
            
            for j in range(len(self.E)):#number of wavelengths           
                #E[j] = self.E[j].take(index_coord_1, axis = coord3)  
                E[j] = self.E[j][:,:,index_coord_1,index_coord_2]            
            return E   
        elif axis == "Y":
            self.coord1 = self.x
            self.coord2 = self.z
            index_coord_1 = np.argmin(np.abs(self.coord1 - x1))
            index_coord_2 = np.argmin(np.abs(self.coord2 - x2))
            
            for j in range(len(self.E)):#number of wavelengths           
                #E[j] = self.E[j].take(index_coord_1, axis = coord3)  
                E[j] = self.E[j][:,index_coord_1,:,index_coord_2]            
            return E   
        elif axis == "Z":
            self.coord1 = self.x
            self.coord2 = self.y   
            index_coord_1 = np.argmin(np.abs(self.coord1 - x1))
            index_coord_2 = np.argmin(np.abs(self.coord2 - x2))
            
            for j in range(len(self.E)):#number of wavelengths           
                #E[j] = self.E[j].take(index_coord_1, axis = coord3)  
                E[j] = self.E[j][:,index_coord_1, index_coord_2,:]            
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
            os.listdir(self.data_folder+"/simulations")
        except FileNotFoundError:
            os.mkdir(self.data_folder+"/simulations")
            
        for file in os.listdir(self.data_folder+"/simulations"):
            if file.endswith(".json"):
                with open(self.data_folder+"/simulations/" + file) as json_file:
                    params = json.load(json_file)
                    
                # initializing compare keys 
                comp_keys = ['Atomic system', 'Material', 'Trap wavelengths', 'Considered state','Geometry','Surface']
                
                # Compare Dictionaries on certain keys using all()
                res = all(params.get(key) == self.params.get(key) for key in comp_keys)
                if res:
                    self.data_file = file[:-4]+"npy"
                    return True
        return False
    

    @pyqtSlot() 
    def compute_potential(self,plane,coord,bypass = False):
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
        self.coord1, self.coord2 = set_axis_from_plane(plane,self)
        CP_mask = self.surface.set_mask2D(self, plane, coord)
        # self.C3 = self.atomicsystem.get_C3(self.material,self.atomicsystem.state)
        CP_mask = -self.C3*CP_mask/kB/mK
        
        self.CP = np.zeros((CP_mask.shape[0],CP_mask.shape[1],len(self.mf_all)))
        for k in range(len(self.mf_all)):
            self.CP[:,:,k] = CP_mask
        #self.CP = np.array([CP_mask,]*len(self.mf_all))
        
        mf_shift = self.mf_all + self.atomicsystem.f
        
        if bypass == True:
            print("Bypassing checks that the current params correspond to the simulation")
            self.signal.emit(200)
            self.finished.emit()
            return self.total_potential()
            
        if self.exists_in_past_simulations():
            print("Reusing data from file: %s" %(self.data_file))
            self.potentials = np.load(self.data_folder+"/simulations/" + self.data_file, allow_pickle = True)
            self.already_saved = True
            self._signal.emit(200)
            self.finished.emit()
            return self.total_potential()
        else:           
            print("New simulation")
            if np.ndim(self.E[0])==4:
                E = self.restrict2D(plane,coord)
            else:
                E = self.E.copy()
             
            self.potentials = np.zeros((len(self.trap.beams),len(self.coord1),len(self.coord2),len(mf_shift)),dtype = "complex")

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.
    
            beam_number = 0
            potential_number = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    Etot = E[beam_number]
                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())
                    E_bwd = inverse_propagation_direction(E[beam_number+1],self.trap.propagation_axis)
                    Etot = E[beam_number] + E_bwd
                else:
                    raise ValueError('Something bad happened. The trap should only contain Beam type objects')
            
                # for i in range(len(self.coord1)):
                #     for j in range (len(self.coord2)):
                #         E0, Ep, Em = convert_fields_to_spherical_basis(Etot[0,i,j],Etot[1,i,j],Etot[2,i,j])
                #         val, _ = self.atomicsystem.potential(Ep,Em,E0)
                #         self.potentials[potential_number,i,j,:] = np.real(val[mf_shift])
                dim1 = len(self.coord1)
                dim2 = len(self.coord2)
                p= multiprocessing.Pool()
                input = ((i,j,self.atomicsystem,Etot,mf_shift) for i,j in itertools.product(range(dim2), repeat=2) if i < dim1)
                output = ((i,j) for i,j in itertools.product(range(dim2), repeat=2) if i < dim1)
                results_np = np.zeros((dim1,dim2,len(mf_shift)))
                with p:
                    results = p.map(process, input)
                    for i,elt in enumerate(output):
                        self.potentials[potential_number][elt][:] = np.real(results[i][mf_shift])
                    #self._signal.emit(i)

                potential_number += 1
                beam_number += (1 if beam.isBeam() else 2)
            
            self.already_saved = False
            self.finished.emit()
            return self.total_potential()

    def total_potential(self):
        """Uses the potentials attributes of the Simulation object for each beam to return their weighted sum with the specified powers
        
        Returns:
            total potential : array with shape(length coordinate 1,length coordinate 2,number of possbile mf states)
        """
        self.total_potential_noCP = np.zeros(np.shape(self.potentials[0]),dtype = 'float')
        for (i,potential) in enumerate(self.potentials):
            self.total_potential_noCP = self.total_potential_noCP + np.mean(self.trap.beams[i].get_power())*potential
        return self.total_potential_noCP +self.CP

    def compute_potential_1D(self,axis,coord1,coord2, bypass = False):
        """Computes the potential in 2D for each Beam or BeamPair separately, for all possible mf states, and returns the total potential which is the weighted sum of both with the powers. 
        If a Surface has been set when initializing the simulation, the Casimir-Polder surface interactions are included.
        Checks (with exists_in_past_simulation method) if a simulation with the same parameters has already been saved. If so, fetches the data without running a new simulation.
        If not, creates a new .json with the parameters and a new .npy with the potentials object (of shape (number of beams,length of axis,number of possible mf states)).

        Args:
            plane (str): As the electrical fields specified are 3D, a slice has to be chosen. Either "XY", "YZ" or "XZ".
            coord (float): Coordinate of the plane on the orthogonal axis
            
        Returns:
           array : total potential with shape(length of axis,number of possible mf states)
        """
        self.dimension = "1D"
        self.coord = None
        self.plane = None
        self.plane_coord1, self.plane_coord2 = coord1, coord2
        self.axis = set_axis_from_axis(axis,self)
        CP_mask = self.surface.set_mask1D(self, axis, self.plane_coord1, self.plane_coord2)
        # self.C3 = self.atomicsystem.get_C3(self.material,self.atomicsystem.state)
        CP_mask = -self.C3*CP_mask/kB/mK             
        self.CP = np.array([CP_mask,]*len(self.mf_all)).transpose()

        mf_shift = self.mf_all + self.atomicsystem.f

        if bypass == True:
            print("Bypassing checks that the current params correspond to the simulation")
            self._signal.emit(200)
            self.finished.emit()
            return self.total_potential()
            
        if self.exists_in_past_simulations():
            print("Reusing data")
            self.potentials = np.load(self.data_folder+"/simulations/" + self.data_file, allow_pickle = True)
            self.already_saved = True
            self._signal.emit(200)
            self.finished.emit()
            return self.total_potential()
        else:           
            print("New simulation")
            if np.ndim(self.E[0])>=3:
                E = self.restrict1D(axis,coord1, coord2)
            else:
                E = self.E.copy()
             
            self.potentials = np.zeros((len(self.trap.beams),len(self.axis),len(mf_shift)),dtype = "complex")

            ## Here we return a table, containing the potential induced by each Beam component of the Trap.
            ## Thus, a BeamPair will return only one potential, as well as a Beam.
            
            beam_number = 0
            potential_number = 0
            for beam in self.trap.beams:
                if beam.isBeam():
                    self.atomicsystem.set_alphas(beam.get_lmbda())
                    Etot = E[beam_number]
                elif beam.isBeamPair():
                    self.atomicsystem.set_alphas_contrapropag(*beam.get_lmbda())
                    E_bwd = inverse_propagation_direction(E[beam_number+1],self.trap.propagation_axis)
                    Etot = E[beam_number] + E_bwd
                else:
                    raise ValueError('Something bad happened. The trap should only contain Beam type objects')
            
                for i in range(len(self.axis)):
                    E0, Ep, Em = convert_fields_to_spherical_basis(Etot[0,i],Etot[1,i],Etot[2,i])
                    val, _ = self.atomicsystem.potential(Ep,Em,E0)
                    self.potentials[potential_number,i,:] = np.real(val[mf_shift])
                    self._signal.emit(i)
                
                potential_number += 1
                beam_number += (1 if beam.isBeam() else 2)
            
            self.already_saved = False
            self.finished.emit()
            return self.total_potential()

    def set_current_params(self):
        """Sets a dictionnary with all the relevant parameters of the Simulation object and returns it.

        Returns:
            dict : parameters of the simulation
        """
        
        self.lmbdas_params =np.array([])
        self.P_params = np.array([])
        # for beam in self.trap.beams:
        #     if beam.isBeam():
        #         self.lmbdas_params = np.append(self.lmbdas_params,beam.get_lmbda())
        #         self.lmbdas_params = np.append(self.lmbdas_params,0)
        #         self.P_params = np.append(self.P_params,beam.get_power())
        #         self.P_params = np.append(self.P_params,0)
        #     elif beam.isBeamPair():
        #         self.lmbdas_params = np.append(self.lmbdas_params,beam.get_lmbda())
        #         self.P_params = np.append(self.P_params,beam.get_power())
        #         self.P_params = np.append(self.P_params,0)
        #     else:
        #         raise ValueError('Something bad happened. The trap should only contain Beam type objects')
        
        for (k,beam) in enumerate(self.trap.beams):
            self.lmbdas_params = np.append(self.lmbdas_params,beam.get_lmbda())
            self.P_params = np.append(self.P_params,beam.get_power())
            self.lmbdas_params.resize(2*(k+1),refcheck=False)
            self.P_params.resize(2*(k+1), refcheck=False)
            
        self.lmbdas_params.resize(4, refcheck = False)
        self.P_params.resize(4, refcheck = False)    
                
        lambda1pair1 = str(self.lmbdas_params[0]/nm)+' nm'
        power1pair1 = str(self.P_params[0]/mW)+' mW'
        lambda2pair1 = str(self.lmbdas_params[1]/nm)+' nm'
        power2pair1 = str(self.P_params[1]/mW)+' mW'
        
        lambda1pair2 = str(self.lmbdas_params[2]/nm)+' nm'
        power1pair2 = str(self.P_params[2]/mW)+' mW'
        lambda2pair2 =str(self.lmbdas_params[3]/nm)+' nm'
        power2pair2 = str(self.P_params[3]/mW)+' mW' 
        now = datetime.now()
        current_time = now.strftime("%d/%m/%Y %H:%M:%S")
        
        self.params = {
            "Time of simulation": current_time,
            "Atomic system": {
                "species":type(self.atomicsystem.atom).__name__,
                "groundstate": str(self.atomicsystem.groundstate),
                "hyperfine level": int(self.atomicsystem.f)
                # "excited_state": str(self.atomicsystem.excitedstate)
            },
            "Material": str(self.material),
            'Trap wavelengths':{
                'lambda 1 pair 1':lambda1pair1,
                'lambda 2 pair 1':lambda2pair1,
                'lambda 1 pair 2':lambda1pair2,
                'lambda 2 pair 2':lambda2pair2
                },
            'Trap powers':{
                'power 1 pair 1':power1pair1,
                'power 2 pair 1':power2pair1,
                'power 1 pair 2':power1pair2,
                'power 2 pair 2':power2pair2
                },
            "Considered state": str(self.atomicsystem.state),
            "Geometry":{
                "2D": self.dimension == "2D",
                "2D plane": self.plane,
                "2D orthogonal coord": self.coord,
                "1D": self.dimension == "1D",
                "1D coord1":self.plane_coord1,
                "1D coord2":self.plane_coord2
            },
            "Surface":self.surface.params,            
            "Data_folder":self.data_folder
        }
        
        print("Simulation parameters set")
        return self.params
   
    def save(self):
        """Saves both the parameters dictionnary into a .json file and the potentials attribute into a .npy.
        """
        if not self.already_saved :
            
            current_time = self.params['Time of simulation'] 
            current_time = current_time.replace('/','_')
            current_time = current_time.replace(":","_")
            current_time = current_time.replace(" ","_")
            
            filename_params = self.data_folder + '/simulations/' + str(current_time) + '.json'
            filename_data = self.data_folder + '/simulations/' + str(current_time) + '.npy'
            
            if not os.path.exists(os.path.dirname(filename_params)):
                try:
                    os.makedirs(os.path.dirname(filename_params))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            
            with open(filename_params, 'w') as fp:
                json.dump(self.params, fp)
                
            np.save(filename_data, self.potentials)
            
            self.already_saved = True
        else :
            print("This simulation has already been saved, see %s" %(self.data_file))
        




    