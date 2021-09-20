from NanoTrap.trapping.structures import PeriodicStructure, Structure
from NanoTrap.trapping.atomicsystem import atomiclevel, atomicsystem_dico
import NanoTrap.utils.physicalunits as pu
import NanoTrap.utils.materials as Nm
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
from NanoTrap.utils.viz import DiscreteSlider

coef_norm = 1

def convert_fields_to_spherical_basis(Ex,Ey,Ez):   ## returns Ep, Em, E0 (quantization axis 'z')
    return -(Ex+Ey*1j)/np.sqrt(2), (Ex-Ey*1j)/np.sqrt(2), Ez

def dict_sum(d1,d2):
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) & set(d2)}

def is_first(axe,plane):
    if axe == plane[0]:
        return True
    else:
        return False

def is_second(axe,plane):
    if axe == plane[1]:
        return True
    else:
        return False
   
class Outside_Simulation():
    def __init__(self,system,trap,E,wavelengths,powers,labels,x,y,z,material):
        """
        Careful : E must be an array of dictionnaries, even if there's only one field, the fields must be in the same
        order as the lambdas given for the trap
        """
        self.system = system
        self.trap = trap
        self.x = x
        self.y = y
        self.z = z
        self.E = E
        
        self.wavelengths = wavelengths
        self.powers = powers
        self.labels = labels
        #If powers are not given, we assume it is normalized to 1 W
        #have to be able to give only 1 value of W

        self.mask_red = {}
        self.mask_blue = {}
        self.trap_2D = {}
        
        self.material = material

        self.mask_red1D = {}
        self.mask_blue1D = {}
        self.trap_1D = {}
        
        self.out_of_plane_coord = 0
        self.inplane_coord = 0
        self.normalized = 0
        self.wavelengths_trap = list(self.trap.wavelengths_trap)  
        self.trap_directions = list(self.trap.directions)
        self.trap_labels = list(self.trap.labels)
        
        self._wavelengths_indices = None
        
        self.already_saved
        
        self.indices_blue = [i for i, x in enumerate(labels) if x == "blue"]
        self.indices_red = [i for i, x in enumerate(labels) if x == "red"]
        
        # indices_blue = np.where(labels == "blue")[0]
        # indices_red = np.where(labels == "red")[0]
    
    # def get_trap(self):
    #     return self.my_trap

    # def set_trap(self, new_trap):
    #     self.my_trap.wavelengths_dict = new_trap.wavelengths_dict
    #     labels = []
    #     directions = []
    #     for k in range(len(list(lambdas.keys()))):
    #         if "blue" in list(lambdas.keys())[k]: 
    #             labels = labels + ["blue"]
    #         else :
    #             labels = labels + ["red"]
    #         if "fwd" in list(lambdas.keys())[k]: 
    #             directions = directions + ["fwd"] 
    #         else :
    #             directions = directions + ["bwd"]
    #     self._trap_labels = labels
    #     self._trap_directions = directions
    #     # generate listsquare when my_list is updated
    #     self.wavelengths_indices = [[x**2 for x in self._trap_wavelengths]]
        
    # self.my_trap = property(get_trap, set_trap, None, 'updates wavelengths_indices')
    
    @property
    def wavelengths_indices(self):
        self.wavelengths_trap = list(self.trap.wavelengths_trap)
        self.trap_directions = list(self.trap.directions)
        self.trap_labels = list(self.trap.labels)
        
        self._wavelengths_indices = []
        for k in range(len(self.wavelengths_trap)):
            index = self.wavelengths.index(self.wavelengths_trap[k])
            if self.trap_labels[k] == self.labels[index] :
                self._wavelengths_indices = self._wavelengths_indices + [index]
        return self._wavelengths_indices
    
    def normalize_E(self):
        if self.normalized == 1:
            raise ValueError("The fields have already been normalized, don't do it again !")
        else :
            E = [{} for k in range(len(self.E))]
            for j in range(len(self.E)):
                for keys, values in self.E[j].items():
                    E[j][keys] = values/np.sqrt(np.abs(self.powers[j]))
            self.E = E
            self.normalized = 1
        return self.E
                
    def restrictE2D(self,plane,coord_value):
        E = [{} for k in range(len(self.E))]
        
        if plane == "XY":
            coord3 = 2
            self.coord = self.z
        elif plane == "YZ":
            coord3 = 0
            self.coord = self.x
        elif plane == "XZ":
            coord3 = 1
            self.coord = self.y
        
        index_coord = np.argmin(np.abs(self.coord - coord_value))
            
        for j in range(len(self.E)):#number of wavelengths at each color
            for keys, values in self.E[j].items():
                slc = [slice(None)] * len(self.E[j]["x"].shape)
                slc[coord3] = slice(index_coord,index_coord+1)
                E[j][keys] = np.squeeze(values[slc])
                
        return E
        
    #####################################                   Masks in 3D                  #################################
    def getE_convert_and_compute(self,state,f,mf_shift,plane,coord,color_index, wavelength_index):
#        E = self.E
        #        E = self.E[wavelength_index]
        if len(np.shape(self.E[0][0]["x"])) == 3 :
            E = self.restrictE2D(plane,coord)
            E = E[color_index][wavelength_index]
        else :
            E = self.E[color_index][wavelength_index]
        res = np.zeros((len(self.coord1),len(self.coord2)),dtype = "complex")
        for i in range(len(self.coord1)):
#        for i in range(40):
            print("i =", i)
            for j in range(len(self.coord2)):
#            for j in range(20):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,j],E['y'][i,j],E['z'][i,j])
                val, vec = self.system.potential(Ep,Em,E0,state,f,self.wavelengths_dict[color_index][wavelength_index])
                res[i,j] = val[mf_shift]
        return res, vec

    def getE_convert_and_compute_contrapropag(self,state,f,mf_shift,plane,color_index,indexes = [0,1]):
        E1 = self.E[color_index][indexes[0]]
        E2 = self.E[color_index][indexes[1]]
        res = np.zeros((len(self.coord1),len(self.coord2)),dtype = "complex")
        E2['x'] = -E2['x']
        E = dict_sum(E1,E2)
        for i in range(len(self.coord1)):
            for j in range(len(self.coord2)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,j],E['y'][i,j],E['z'][i,j])
                val, vec  = self.system.potential_contrapropag(Ep,Em,E0,state,f,self.wavelengths_dict[indexes[0]],self.wavelengths_dict[indexes[1]])
                res[i,j] = val[mf_shift]
        return res, vec

    def set_masks(self,state,f,mf,plane, out_of_plane_coord,wavelength_index_blue, wavelength_index_red):
        mf_shift = mf + f
        
        if self.trap.nred == 0 :
            try:
                self.mask_blue[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)]
                pass
            except KeyError:
                if self.trap.nblue == 0 :
                    raise "Nothing to simulate"
                elif self.trap.nblue == 1 : 
                    res, vec = self.getE_convert_and_compute(state,f,mf_shift,plane,out_of_plane_coord,0,wavelength_index_blue)
                    self.mask_blue[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)] = res
                elif self.trap.nblue == 2 :
                    res, vec = self.getE_convert_and_compute_contrapropag(state,f,mf_shift)
                    self.mask_blue[plane] = res
        elif self.trap.nblue == 0 :
            try:
                self.mask_red[plane,np.round(self.wavelengths_dict[0][wavelength_index_red],11)]
                pass
            except KeyError:
                if self.trap.nred == 1:
                    res, vec = self.getE_convert_and_compute(state,f,mf_shift,plane,out_of_plane_coord,0,wavelength_index_red)
                    self.mask_red[plane,np.round(self.wavelengths_dict[0][wavelength_index_red],11)] = res     
                elif self.trap.nred == 2:
                    res, vec = self.getE_convert_and_compute_contrapropag(state,f,mf_shift)
                    self.mask_red[plane] = res     
        elif self.trap.nblue >= 2 or self.trap.nred >= 2 :
            raise ValueError("nred and nblue must be integers between 0 and 2")
        else:
            try:
                self.mask_red[plane,np.round(self.wavelengths_dict[1][wavelength_index_red],11)]
                self.mask_blue[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)]
                pass
            except KeyError:
                if self.trap.nblue == 1 and self.trap.nred == 1:
                    res1, vec1 = self.getE_convert_and_compute(state,f,mf_shift,plane,out_of_plane_coord,0,wavelength_index_blue)
                    res2, vec2 = self.getE_convert_and_compute(state,f,mf_shift,plane,out_of_plane_coord,1,wavelength_index_red)
                    self.mask_blue[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)] = res1
                    self.mask_red[plane,np.round(self.wavelengths_dict[1][wavelength_index_red],11)] = res2
                elif self.trap.nblue ==2 and self.trap.nred == 1:
                    res1, vec1 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,[0,1])
                    res2, vec2 = self.getE_convert_and_compute(state,f,mf_shift,2)
                    self.mask_blue[plane] = res1
                    self.mask_red[plane] = res2
                elif self.trap.nblue == 1 and self.trap.nred == 2:
                    res1, vec1 = self.getE_convert_and_compute(state,f,mf_shift)
                    res2, vec2 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,[1,2])
                    self.mask_blue[plane] = res1
                    self.mask_red[plane] = res2        
                elif self.trap.nblue ==2 and self.trap.nred == 2 :
                    res1, vec1 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift)
                    res2, vec2 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,[2,3])
                    self.mask_blue[plane] = res1
                    self.mask_red[plane] = res2
        print("Simulation completed, masks available")       

    def simulate(self,state,f,mf,plane,out_of_plane_coord,wavelength_index_blue = None,wavelength_index_red = None):
        mf = np.array(mf)
        if mf.any() > f or mf.any() < -f:
            raise ValueError("m_F should be in the interval [-F,F]")
        else:
            self.mf = mf
            self.f = f
        if plane == "XY":
            self.coord1 = self.x
            self.coord2 = self.y
        elif plane == "YZ":
            self.coord1 = self.y
            self.coord2 = self.z
        elif plane == "XZ":
            self.coord1 = self.x
            self.coord2 = self.z
        self.out_of_plane_coord = out_of_plane_coord
        self.set_masks(state,self.f,self.mf,plane,out_of_plane_coord,wavelength_index_blue,wavelength_index_red)
        if self.trap.nblue == 0:
            self.trap_2D[plane,np.round(self.wavelengths_dict[0][wavelength_index_red],11)] = self.trap.Pred*self.mask_red[plane,np.round(self.wavelengths_dict[0][wavelength_index_red],11)]
        elif self.trap.nred == 0:
            self.trap_2D[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)] = self.trap.Pblue*self.mask_blue[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)]

        else:
            self.trap_2D[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),np.round(self.wavelengths_dict[1][wavelength_index_red],11)] = self.trap.Pblue*self.mask_blue[plane,np.round(self.wavelengths_dict[0][wavelength_index_blue],11)] + self.trap.Pred*self.mask_red[plane,np.round(self.wavelengths_dict[1][wavelength_index_red],11)]
            
        keys = list(self.trap_2D.keys())
        return self.trap_2D[keys[-1]]
    
    #####################################                   Masks in 1D                   ################################
    def getE_convert_and_compute_1D(self,state,f,mf_shift,plane,axe,out_of_plane_coord,in_plane_coord,field_index):
#        E = self.E[wavelength_index]
        if len(np.shape(self.E[0]["x"])) == 3 :
            E = self.restrictE2D(plane,out_of_plane_coord)
        else :
            E = self.E
        E = E[field_index]
        if is_first(axe,plane):
            res = np.zeros((len(self.coord1),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(self.coord2)-in_plane_coord))
            print(index)
            for i in range(len(self.coord1)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,index],E['y'][i,index],E['z'][i,index])
                val, vec = self.system.potential(Ep,Em,E0,state,f,self.wavelengths_dict[color_index][wavelength_index])
                res[i] = val[mf_shift]
        elif is_second(axe,plane):
            res = np.zeros((len(self.coord2),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(self.coord1)-in_plane_coord))
            print(index)
            for j in range(len(self.coord2)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][index,j],E['y'][index,j],E['z'][index,j])
                val, vec = self.system.potential(Ep,Em,E0,state,f,self.wavelengths_dict[color_index][wavelength_index])
                res[j] = val[mf_shift]
        return res, vec

    def getE_convert_and_compute_contrapropag_1D(self,state,f,mf_shift,plane,axe,out_of_plane_coord,in_plane_coord,color_index,wavelength_index):
        try:
            l = len(wavelength_index)
            if l == 1 or l == 2:
                wavelength_index_1 = wavelength_index[0]
                wavelength_index_2 = wavelength_index[-1]
            else : 
                print("You must give 1 or 2 indexes for the wavelengths")
            pass
        except:
            wavelength_index_1 = wavelength_index
            wavelength_index_1 = wavelength_index

        if len(np.shape(self.E[0][0]["x"])) == 3 :
            E = self.restrictE2D(plane,out_of_plane_coord)
        else :
            E = self.E
        E1 = E[color_index][wavelength_index_1]
        E2 = E[color_index][wavelength_index_2]
        # E1,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index_1][1],self.indices[wavelength_index_1][0])
        # E2,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index_2][1],self.indices[wavelength_index_2][0])
        E2['x'] = -E2['x']
        E = dict_sum(E1,E2)
        if is_first(axe,plane):
            res = np.zeros((len(self.coord1),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(self.coord2)-in_plane_coord))
            for i in range(len(self.coord1)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,index],E['y'][i,index],E['z'][i,index])
                val, vec = self.system.potential_contrapropag(Ep,Em,E0,state,f,self.wavelenths_dict[color_index][wavelength_index_1],self.wavelenths_dict[color_index][wavelength_index_2])
                res[i] = val[mf_shift]
        elif is_second(axe,plane):
            res = np.zeros((len(self.coord2),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(self.coord1)-in_plane_coord))
            for j in range(len(self.coord2)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][index,j],E['y'][index,j],E['z'][index,j])
                val, vec = self.system.potential_contrapropag(Ep,Em,E0,state,f,self.wavelengths_dict[color_index][wavelength_index_1],self.wavelengths_dict[color_index][wavelength_index_2])
                res[i] = val[mf_shift]

        return res, vec
    
    def set_masks1D(self,state,f,mf,plane,axe,coord1,coord2, wavelength_index_blue, wavelength_index_red):
        mf_shift = mf + f
        try :
            self.mask_red1D[axe,np.round(self.wavelengths_dict[wavelength_index],11),coord1,coord2]
            self.mask_blue1D[axe,np.round(self.wavelengths_dict[wavelength_index],11),coord1,coord2]
            pass
        except KeyError:
            
            if self.trap.nblue == 0 and self.trap.nred == 0 :
                raise "Nothing to simulate"

            elif self.trap.nblue == 1 and self.trap.nred == 0 :
                res, self.vec = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord1,coord2,0,wavelength_index_blue)
                self.mask_blue1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),coord1,coord2] = res

            elif self.trap.nblue == 0 and self.trap.nred == 1 :
                res, self.vec = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord1,coord2,1,wavelength_index_red)
                self.mask_red1D[axe,np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2] = res

            elif self.trap.nblue == 1 and self.trap.nred == 1:
                res1, self.vec1 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord1,coord2,0,wavelength_index_blue)
                res2, self.vec2 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord1,coord2,1,wavelength_index_red)
                self.mask_blue1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),coord1,coord2] = res1
                self.mask_red1D[axe,np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2] = res2

            elif self.trap.nblue == 2 and self.trap.nred == 0:
                res, self.vec = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord1,coord2,0,wavelength_index_blue)
                self.mask_blue1D[axe] = res

            elif self.trap.nblue == 0 and self.trap.nred == 2:
                res, self.vec = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord1,coord2,1,wavelength_index_red)
                self.mask_red1D[axe] = res

            elif self.trap.nblue ==2 and self.trap.nred == 1:
                res1, self.vec1 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord1,coord2,[0,1])
                res2, self.vec2 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord1,coord2,2)
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            elif self.trap.nblue == 1 and self.trap.nred == 2:
                res1, self.vec1 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord1,coord2,0)
                res2, self.vec2 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord1,coord2,[1,2])
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            elif self.trap.nblue ==2 and self.trap.nred == 2 :
                res1, self.vec1 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord1,coord2,[0,1])
                res2, self.vec2 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord1,coord2,[2,3])
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            else :
                raise ValueError("nred and nblue must be integers between 0 and 2")
            print("Simulation completed, masks available") 
            
    
    
    def simulate1D(self,state,f,mf,plane,axe,coord1,coord2):
        """ mf has to be a list ! """
        mf = np.array(mf)
        if mf.any() > f or mf.any() < -f:
            raise ValueError("m_F should be in the interval [-F,F]")
        else:
            self.mf = mf
            self.f = f
        if plane == "XY":
            self.coord1 = self.x
            self.coord2 = self.y
        elif plane == "YZ":
            self.coord1 = self.y
            self.coord2 = self.z
        elif plane == "XZ":
            self.coord1 = self.x
            self.coord2 = self.z

        self.out_of_plane_coord = coord1
        self.inplane_coord = coord2
        
        self.set_masks1D(state,self.f,self.mf,plane,axe,coord1,coord2)
        for k in range(len(self.wavelengths_indices)):
            
        
        if self.trap.nblue == 0:
            self.trap_1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2] = self.trap.Pred*self.mask_red1D[axe,np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2]
        elif self.trap.nred == 0:
            self.trap_1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2] = self.trap.Pblue*self.mask_blue1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),coord1,coord2]
        else:
            self.trap_1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2] = self.trap.Pblue*self.mask_blue1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),coord1,coord2] + self.trap.Pred*self.mask_red1D[axe,np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2]

        return self.trap_1D[axe,np.round(self.wavelengths_dict[0][wavelength_index_blue],11),np.round(self.wavelengths_dict[1][wavelength_index_red],11),coord1,coord2]
          
    def plot_trap(self,x,y,edge,plane):
        if plane == "XY":
            key1 = list(self.trap_2D.keys())[-1]
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.5, bottom=0.05)
            trap = np.real(self.trap_2D[key1])
            #the norm TwoSlopeNorm allows to fix the 0 of potential to the white color, so that we can easily distinguish between positive and negative values of the potential
            a = plt.pcolormesh(x,y,np.transpose(trap),shading = "gouraud",norm = colors.TwoSlopeNorm(vmin=np.min(trap)/2, vcenter=0, vmax=np.max(trap)*2), cmap = "seismic")
            cbar = plt.colorbar(a)
            y_edge_ind = [i for i in range(len(self.y)) if self.y[i] >= edge]
            index_min = np.unravel_index(self.trap_2D[key1][:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),self.trap_2D[key1][:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
            point, = plt.plot(x[index_min[0]],y[index_min[1]+y_edge_ind[0]],'ro')
            plt.xlabel("x (nm)")
            plt.ylabel("y (nm)")
        if plane == "YZ":
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.5, bottom=0.05)
            key1 = list(self.trap_2D.keys())[-1]
            trap = np.real(self.trap_2D[key1])
            a = plt.pcolormesh(x,y,np.transpose(trap),shading = "gouraud",norm = colors.TwoSlopeNorm(vmin=np.min(trap)/2, vcenter=0, vmax=np.max(trap)), cmap = "seismic")
            cbar = plt.colorbar(a)
            y_edge_ind = [i for i in range(len(self.y)) if self.y[i] >= edge]
            index_min = np.unravel_index(self.trap_2D[key1][y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),self.trap_2D[key1][y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
            point, = plt.plot(self.y[index_min[0]+y_edge_ind[0]],self.z[index_min[1]],'ro')
            plt.xlabel("y (nm)")
            plt.ylabel("z (nm)")

        plt.title("2D plot of trapping potentiel in the %s plane" %(plane))

        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'
        axPred = plt.axes([0.23, 0.1, 0.03, 0.75], facecolor=axcolor)
        sPred = Slider(axPred, 'Pred', 0, 50.0, valinit=self.trap.Pred, valstep=0.1,orientation='vertical')
        axPblue = plt.axes([0.15, 0.1, 0.03, 0.75], facecolor=axcolor)
        sPblue = Slider(axPblue, 'Pblue', 0, 50.0, valinit=self.trap.Pblue, valstep=0.1,orientation='vertical')
        if self.trap.nblue != 0:
            axlmbdablue = plt.axes([0.31, 0.1, 0.03, 0.75], facecolor=axcolor)
            slmbdablue = DiscreteSlider(axlmbdablue, 'lmbdablue', min(self.wavelengths_dict[0])/pu.nm, max(self.wavelengths_dict[0])/pu.nm, allowed_vals= np.array(self.wavelengths_dict[0])/pu.nm, valinit=self.wavelengths_dict[0][0],valfmt='%1.3f',orientation='vertical')
        if self.trap.nred != 0:   
            axlmbdared = plt.axes([0.39, 0.1, 0.03, 0.75], facecolor=axcolor)
            slmbdared = DiscreteSlider(axlmbdared, 'lmbdared', min(self.wavelengths_dict[1])/pu.nm, max(self.wavelengths_dict[1])/pu.nm, allowed_vals= np.array(self.wavelengths_dict[1])/pu.nm, valinit=self.wavelengths_dict[1][0],valfmt='%1.3f',orientation='vertical')

        def updateP(val):
            Pb = sPblue.val
            Pr = sPred.val
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap_2D = self.simulate(self.system.groundstate,self.f,self.mf,plane,self.out_of_plane_coord,self.wavelength_index_blue,self.wavelength_index_red)
            a.set_array(np.transpose(np.real(trap_2D)).ravel())
#            a.set_clim(vmin=np.min(np.real(trap_2D)),vmax=np.max(np.real(trap_2D)))
#            cbar.draw_all()
#            a.set_data(np.transpose(np.real(self.trap_2D[plane])))
            if plane == "XY":
                argmin_update = np.unravel_index(trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
                print(argmin_update)
                point.set_data(self.x[argmin_update[0]],self.y[argmin_update[1]+y_edge_ind[0]])

            elif plane == "YZ":
                argmin_update = np.unravel_index(trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
                point.set_data(self.y[argmin_update[0]+y_edge_ind[0]],self.z[argmin_update[1]])
            # print(argmin_update)
            a.autoscale()
            fig.canvas.draw_idle()
            
        def updatel(val):
            print("lambdablue", slmbdablue.val)
            if self.trap.nblue != 0:
                self.wavelength_index_blue = np.argmin(np.abs(np.array(self.wavelengths_dict[0])/pu.nm - slmbdablue.val))
            if self.trap.nred != 0:
                self.wavelength_index_red = np.argmin(np.abs(np.array(self.wavelengths_dict[1])/pu.nm - slmbdared.val))
            print("wb, wr", self.wavelength_index_blue, self.wavelength_index_red)
            trap_2D = self.simulate(self.system.groundstate,self.f,self.mf,plane,self.out_of_plane_coord,self.wavelength_index_blue,self.wavelength_index_red)
            a.set_array(np.transpose(np.real(trap_2D)).ravel())
#            a.set_clim(vmin=np.min(np.real(trap_2D)),vmax=np.max(np.real(trap_2D)))
#            cbar.draw_all()
#            a.set_data(np.transpose(np.real(self.trap_2D[plane])))
            if plane == "XY":
                argmin_update = np.unravel_index(trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
                print(argmin_update)
                point.set_data(self.x[argmin_update[0]],self.y[argmin_update[1]+y_edge_ind[0]])

            elif plane == "YZ":
                argmin_update = np.unravel_index(trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
                point.set_data(self.y[argmin_update[0]+y_edge_ind[0]],self.z[argmin_update[1]])
                print(argmin_update)
            # print(argmin_update)
            a.autoscale()
            fig.canvas.draw_idle()

        sPblue.on_changed(updateP)
        sPred.on_changed(updateP)
        slmbdared.on_changed(updatel)
        slmbdablue.on_changed(updatel)
        plt.show()
        
        return fig, ax, sPblue, sPred, slmbdablue, slmbdared
        
    def plot_trap1D(self,plane,axis,edge1, edge2 = None, add_CP = False,Pranges=[]):
        """ Have to add CP interactions"""
        
        if axis == "X":
            self.coord = self.x
        elif axis == "Y":
            self.coord = self.y
        elif axis == "Z":
            self.coord = self.z
        else :
            print("axis must be either 'X', 'Y' or 'Z'")
            
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.27)
        jet = cm = plt.get_cmap('Greys')
        cNorm  = colors.Normalize(vmin=-1, vmax=len(self.mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        a = []
        
        index_edge1 = np.argmin(np.abs(self.coord -edge1))
        if self.coord[index_edge1] < edge1 :
            index_edge1 += 1
        index_edge = slice(index_edge1, None)

        if edge2 != None:
            index_edge2 = np.argmin(np.abs(self.coord -edge2))
            if self.coord[index_edge2] > edge2 :
                index_edge2 -= 1
            index_edge = slice(index_edge1, index_edge2)

        coord_outside = self.coord[index_edge] - edge1
        
        key = [x for x in list(self.trap_1D.keys()) if axis in x][-1]
        trap_outside = np.real(self.trap_1D[key][index_edge,:])

        
        """Addition of CP int"""
        if add_CP == True:
            C3 = self.system.get_C3(self.material,self.system.groundstate)
            CP = np.array([-C3/(r**3)/pu.kB/pu.mK for r in coord_outside])
            if edge2 != None:
                CP_right = np.array([-C3/(r**3)/pu.kB/pu.mK for r in coord_outside[::-1]]) #Reverses the order of the list
                CP = CP + CP_right
            CP_mf = np.array([CP,]*len(self.mf)).transpose()
            trap_outside = np.real(self.trap_1D[key][index_edge,:]) + CP_mf
#            trap_outside = np.zeros((len(coord_outside),len(self.mf))) + CP_mf

        for k in range(len(self.mf)):
            colorVal = scalarMap.to_rgba(k)
            a = a + plt.plot(coord_outside,trap_outside[:,k], color = colorVal, label = "m$_f$ = %s" %(self.mf[k]), linewidth = 2 + 3/len(self.mf))

        if len(self.mf) == 1:
            b, = plt.plot(coord_outside,np.real(self.trap.Pblue*np.real(self.mask_blue1D[itemgetter(0,1,3,4)(key)][index_edge])), color = 'blue', linewidth = 2) #/(pu.kB*coef_norm*pu.mK))
            r, = plt.plot(coord_outside,np.real(self.trap.Pred*np.real(self.mask_red1D[itemgetter(0,2,3,4)(key)][index_edge])), color = 'red', linewidth = 2) #/(pu.kB*coef_norm*pu.mK))
        
        plt.axhline(y=0, color='black', linestyle='--')
        # plt.ylim(-3,3)
        plt.legend()
        plt.xlabel(axis+" (nm)")
        plt.ylabel("E (mK)")
#        plt.ylim(-5e-14,5e-14)
        plt.title("1D plot of trapping potential along %s " %(axis))

        axcolor = 'lightgoldenrodyellow'
        axPblue = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
        axPred = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

        if len(Pranges) == 2:
            sPblue = Slider(axPblue, 'Pblue (mW)', Pranges[0], Pranges[1], valinit=self.trap.Pblue, valstep=0.001)
            sPred = Slider(axPred, 'Pred (mW)', Pranges[0], Pranges[1], valinit=self.trap.Pred, valstep=0.0001)
        elif len(Pranges) == 4:
            sPblue = Slider(axPblue, 'Pblue (mW)', Pranges[0], Pranges[1], valinit=self.trap.Pblue, valstep=0.001)
            sPred = Slider(axPred, 'Pred (mW)', Pranges[2], Pranges[3], valinit=self.trap.Pred, valstep=0.0001)
        else :
            print("Default ranges for powers")
            sPblue = Slider(axPblue, 'Pblue (mW)', 0, 0.5, valinit=self.trap.Pblue, valstep=0.001)
            sPred = Slider(axPred, 'Pred (mW)', 0, 0.1, valinit=self.trap.Pred, valstep=0.0001)

        def update(val):
            Pb = sPblue.val*pu.mW
            Pr = sPred.val*pu.mW
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap = self.simulate1D(self.system.groundstate,self.f,self.mf,plane,axis,self.out_of_plane_coord,self.inplane_coord,self.wavelength_index_blue,self.wavelength_index_red)

            for k in range(len(self.mf)):
                if add_CP == True :
                    trap_outside = np.real(self.trap_1D[key][index_edge,k]) + CP #/(pu.kB*coef_norm*pu.mK) + CP
                else:
                    trap_outside = np.real(self.trap_1D[key][index_edge,k])#/(pu.kB*coef_norm*pu.mK)
                a[k].set_ydata(trap_outside)

            if len(self.mf) == 1:
                b.set_ydata(np.real(self.trap.Pblue*np.real(self.mask_blue1D[itemgetter(0,1,3,4)(key)][index_edge])))#/(pu.kB*coef_norm*pu.mK))
                r.set_ydata(np.real(self.trap.Pred*np.real(self.mask_red1D[itemgetter(0,2,3,4)(key)][index_edge])))#/(pu.kB*coef_norm*pu.mK))
    #            argmin_update = np.unravel_index(trap_1D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
    #            point.set_data(self.structure.period*x[argmin_update[0]]/nm,self.structure.period*y[argmin_update[1]+y_edge_ind[0]]/nm)
            fig.canvas.draw_idle()
        sPblue.on_changed(update)
        sPred.on_changed(update)
        plt.show()
        
        return fig, ax, sPblue, sPred, trap_outside, coord_outside, 
    
    def simpleplot(self,coord_outside,trap_outside,edge):
        fig, ax = plt.subplots()
        # C3 = self.system.get_C3(self.material,self.system.groundstate)
        # CP = np.array([-C3/(r**3)/pu.kB/pu.mK for r in coord_outside])
        # trap_outside = np.real(self.trap_1D[key][index_edge,:]) + CP_mf
#            trap_outside = np.zeros((len(coord_outside),len(self.mf))) + CP_mf

        a = plt.plot(coord_outside,trap_outside, color = "green", linewidth = 4)
        b, = plt.plot(coord_outside*1e9,np.real(self.trap.Pblue*np.real(self.mask_blue1D[itemgetter(0,1,3,4)(key)][index_edge])), color = 'blue', linewidth = 2) #/(pu.kB*coef_norm*pu.mK))
        r, = plt.plot(coord_outside*1e9,np.real(self.trap.Pred*np.real(self.mask_red1D[itemgetter(0,2,3,4)(key)][index_edge])), color = 'red', linewidth = 2)
  
        
    def plot3traps_verif(self,state,f,mf,zcoord,xcoord):
        if len(self.E[0][0]['x'].shape) != 3 :
            print("3D Electric fields must be fed in the Simulation class in order to use this function")
        else : 
            trap_1D_Y_allw = self.simulate1D(state,f,mf,'XY','Y',zcoord,xcoord,self.wavelength_index_blue,self.wavelength_index_red)
            ymin_ind, y_min, trap_depth, y_outside, trap_Y_outside, CP_at_min = self.find_min_trap_1D(np.squeeze(trap_1D_Y_allw),"Y",0,None)
            print("y_min", y_min)
            if y_min is not None:
                trap_1D_X_allw = self.simulate1D(state,f,mf,"XY","X",zcoord,y_min, self.wavelength_index_blue,self.wavelength_index_red)
                trap_1D_Z_allw = self.simulate1D(state,f,mf,"YZ","Z",xcoord,y_min, self.wavelength_index_blue,self.wavelength_index_red)
        fig, ax  = plt.subplots(3, figsize = (15,10))
        ly, = ax[0].plot(self.y,np.squeeze(np.real(trap_1D_Y_allw)))
#        ax[0].set_ylim([-2, 2])
        if y_min is not None:
            lx, = ax[1].plot(self.x,np.squeeze(np.real(trap_1D_X_allw)))
            lz, = ax[2].plot(self.z,np.squeeze(np.real(trap_1D_Z_allw)))
        else:
            lx, = ax[1].plot(self.x,np.zeros((len(self.x),)))
            lz, = ax[2].plot(self.z,np.zeros((len(self.z),)))
        ax[0].set_xlabel("y (m)")
        ax[1].set_xlabel("x (m)")
        ax[2].set_xlabel("z (m)")
        
        
    def plot1Dtraps_3axis(self,state,f,mf,zcoord,xcoord,edge1 = 0, edge2 = None,Pranges=[]):
        if len(self.E[0][0]['x'].shape) != 3 :
            print("3D Electric fields must be fed in the Simulation class in order to use this function")
        else : 
            trap_1D_Y_allw = np.squeeze(np.real(self.simulate1D(state,f,mf,'XY','Y',zcoord,xcoord,self.wavelength_index_blue,self.wavelength_index_red)))#/(pu.kB*coef_norm*pu.mK)
            ymin_ind, y_min, trap_depth, y_outside, trap_Y_outside, CP_at_min = self.find_min_trap_1D(np.squeeze(trap_1D_Y_allw),"Y",edge1, edge2)
            print("y_min", y_min)
            omegax, omegay, omegaz = 0, 0, 0
            if y_min is not None:
                trap_1D_X_allw = np.squeeze(np.real(self.simulate1D(state,f,mf,"XY","X",zcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
                trap_1D_Z_allw = np.squeeze(np.real(self.simulate1D(state,f,mf,"YZ","Z",xcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
                omegax = self.get_trapfreq(self.x,xcoord,trap_1D_X_allw)
                omegay = self.get_trapfreq(y_outside,y_min,trap_Y_outside)
                omegaz = self.get_trapfreq(self.z,zcoord,trap_1D_Z_allw)

            fig, ax  = plt.subplots(3, figsize = (15,10))
            plt.subplots_adjust(left=0.25)
            axcolor = 'lightgoldenrodyellow'
            props = dict(boxstyle='round', facecolor=axcolor, alpha=0.5)
            
            textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, ), r'$\omega_x=%.2f (kHz) $' % (omegax, ), r'$\omega_y=%.2f (kHz) $' % (omegay, ), r'$\omega_z=%.2f (kHz) $' % (omegaz, )))
            box = plt.text(- 0.4, 0.6, textstr, transform=ax[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            
            #Define axes that will eventually receive the sliders
            axPblue = fig.add_axes([0.05, 0.35, 0.02, 0.55], facecolor = axcolor)
            axPred = fig.add_axes([0.1, 0.35, 0.02, 0.55], facecolor = axcolor)
            axlmbdablue = fig.add_axes([0.15, 0.35, 0.02, 0.55], facecolor = axcolor)
            axlmbdared = fig.add_axes([0.2, 0.35, 0.02, 0.55], facecolor = axcolor)

            if len(Pranges) == 2:
                sPblue = Slider(axPblue, 'Pblue (mW)', Pranges[0], Pranges[1], valinit=self.trap.Pblue, valstep=0.001,orientation='vertical')
                sPred = Slider(axPred, 'Pred (mW)', Pranges[0], Pranges[1], valinit=self.trap.Pred, valstep=0.0001,orientation='vertical')
            elif len(Pranges) == 4:
                sPblue = Slider(axPblue, 'Pblue (mW)', Pranges[0], Pranges[1], valinit=self.trap.Pblue, valstep=0.001,orientation='vertical')
                sPred = Slider(axPred, 'Pred (mW)', Pranges[2], Pranges[3], valinit=self.trap.Pred, valstep=0.0001,orientation='vertical')
            else :
                print("Default ranges for powers")
                sPblue = Slider(axPblue, 'Pblue (mW)', 0, 0.5, valinit=self.trap.Pblue, valstep=0.001,orientation='vertical')
                sPred = Slider(axPred, 'Pred (mW)', 0, 0.1, valinit=self.trap.Pred, valstep=0.0001,orientation='vertical')
                
            slmbdablue = DiscreteSlider(axlmbdablue, 'lmbdablue', valmin = min(self.wavelengths_dict[0])/pu.nm, valmax = max(self.wavelengths_dict[0])/pu.nm, allowed_vals= np.array(self.wavelengths_dict[0])/pu.nm, valinit=self.wavelengths_dict[0][self.wavelength_index_blue],valfmt='%1.3f', orientation='vertical')
            slmbdared = DiscreteSlider(axlmbdared, 'lmbdared', min(self.wavelengths_dict[1])/pu.nm, max(self.wavelengths_dict[1])/pu.nm, allowed_vals= np.array(self.wavelengths_dict[1])/pu.nm, valinit=self.wavelengths_dict[1][self.wavelength_index_red],valfmt='%1.3f',orientation='vertical')
            
            ly, = ax[0].plot(y_outside,trap_Y_outside)
            ax[0].set_ylim([-2, 2])
            if y_min is not None:
                lx, = ax[1].plot(self.x,trap_1D_X_allw)
                lz, = ax[2].plot(self.z,trap_1D_Z_allw)
            else:
                lx, = ax[1].plot(self.x,np.zeros((len(self.x),)))
                lz, = ax[2].plot(self.z,np.zeros((len(self.z),)))
            ax[0].set_xlabel("y (m)")
            ax[1].set_xlabel("x (m)")
            ax[2].set_xlabel("z (m)")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.title("Total dipole trap for mf = 0 in the 3 directions")
            
            def updateP(val):
                Pb = sPblue.val*pu.mW
                Pr = sPred.val*pu.mW
                self.trap.set_powers(Pblue = Pb, Pred = Pr)
                trap_1D_Y = self.simulate1D(state,f,mf,"XY","Y",zcoord,xcoord,self.wavelength_index_blue,self.wavelength_index_red)#/(pu.kB*coef_norm*pu.mK)
                ymin_ind, y_min, trap_depth, y_outside, trap_Y_outside, CP_at_min = self.find_min_trap_1D(np.squeeze(trap_1D_Y),"Y",edge1, edge2)
                print("y_min", y_min)
                print("CP at y_min is, when changing power :", CP_at_min)
                ax[0].set_ylim([-2,trap_Y_outside.max()])
                if y_min is not None:
                    trap_1D_X = np.squeeze(np.real(self.simulate1D(state,f,mf,"XY","X",zcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
                    trap_1D_Z = np.squeeze(np.real(self.simulate1D(state,f,mf,"YZ","Z",xcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
                    lx.set_ydata(trap_1D_X)
                    lz.set_ydata(trap_1D_Z)
                    ax[1].set_ylim([trap_1D_X.min(),trap_1D_X.max()])
                    ax[2].set_ylim([trap_1D_Z.min(),trap_1D_Z.max()])
                    ax[0].set_ylim([trap_depth,trap_Y_outside.max()])
                    omegax = self.get_trapfreq(self.x,xcoord,trap_1D_X)
                    omegay = self.get_trapfreq(y_outside,y_min,trap_Y_outside)
                    omegaz = self.get_trapfreq(self.z,zcoord,trap_1D_Z)
                    textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, ), r'$\omega_x=%.2f (kHz) $' % (omegax, ), r'$\omega_y=%.2f (kHz) $' % (omegay, ), r'$\omega_z=%.2f (kHz) $' % (omegaz, )))
                    box.set_text(textstr)
                else :
                    textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, )))
                    box.set_text(textstr)

                ly.set_ydata(np.squeeze(np.real(trap_Y_outside)))

            def updatel(val):
                self.wavelength_index_blue = np.argmin(np.abs(np.array(self.wavelengths_dict[0])/pu.nm - slmbdablue.val))
                self.wavelength_index_red = np.argmin(np.abs(np.array(self.wavelengths_dict[1])/pu.nm - slmbdared.val))
                trap_1D_Y = np.real(self.simulate1D(state,f,mf,"XY","Y",zcoord,xcoord,self.wavelength_index_blue,self.wavelength_index_red))#/(pu.kB*coef_norm*pu.mK)
                ymin_ind, y_min, trap_depth, y_outside, trap_Y_outside, CP_at_min = self.find_min_trap_1D(np.squeeze(trap_1D_Y),"Y",edge1, edge2)
                print("CP at y_min is, when changing lambda :", CP_at_min)
                print("y_min", y_min)
                ax[0].set_ylim([-2,trap_Y_outside.max()])
                
                if y_min is not None:
                    trap_1D_X = np.squeeze(np.real(self.simulate1D(state,f,mf,"XY","X",zcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
                    trap_1D_Z = np.squeeze(np.real(self.simulate1D(state,f,mf,"YZ","Z",xcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
                    lx.set_ydata(trap_1D_X)
                    lz.set_ydata(trap_1D_Z)
                    ax[1].set_ylim([trap_1D_X.min(),trap_1D_X.max()])
                    ax[2].set_ylim([trap_1D_Z.min(),trap_1D_Z.max()])
                    ax[0].set_ylim([trap_depth,trap_Y_outside.max()])
                    omegax = self.get_trapfreq(self.x,xcoord,trap_1D_X)
                    omegay = self.get_trapfreq(y_outside,y_min,trap_Y_outside)
                    omegaz = self.get_trapfreq(self.z,zcoord,trap_1D_Z)
                    textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, ), r'$\omega_x=%.2f (kHz) $' % (omegax, ), r'$\omega_y=%.2f (kHz) $' % (omegay, ), r'$\omega_z=%.2f (kHz) $' % (omegaz, )))
                    box.set_text(textstr)
                else :
                    textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, )))
                    box.set_text(textstr)
#                    print("trap X at y_min :", trap_1D_X[0])
#                    print("trap Z at y_min :", trap_1D_Z[33])
#                print("trap Y at y_min :", trap_Y_outside[ymin_ind])
#                print("ymin_ind",ymin_ind)
                ly.set_ydata(np.squeeze(np.real(trap_Y_outside)))
       
            sPblue.on_changed(updateP)
            sPred.on_changed(updateP)
            slmbdared.on_changed(updatel)
            slmbdablue.on_changed(updatel)
            return fig, ax, sPblue, sPred, slmbdablue, slmbdared

    def find_min_trap_1D(self,trap1D,axis,edge1, edge2 = None):
        """ return y_outside and trap, you have to give it trap1D already in mK"""
        if axis == "X":
            self.coord = self.x
        elif axis == "Y":
            self.coord = self.y
        elif axis == "Z":
            self.coord = self.z
        else :
            print("axis must be either 'X', 'Y' or 'Z'")
            
        index_edge1 = np.argmin(np.abs(self.coord -edge1))
        if self.coord[index_edge1] < edge1 :
            index_edge1 += 1
        index_edge = slice(index_edge1, None)

        if edge2 != None:
            index_edge2 = np.argmin(np.abs(self.coord -edge2))
            if self.coord[index_edge2] > edge2 :
                index_edge2 -= 1
            index_edge = slice(index_edge1, index_edge2)

        y_outside = self.coord[index_edge] - edge1
        C3 = self.system.get_C3(self.material,self.system.groundstate)
        CP = np.array([-C3/(r**3)/pu.kB/pu.mK for r in y_outside])
        if edge2 != None :
            CP_right = np.array([-C3/(r**3)/pu.kB/pu.mK for r in y_outside[::-1]]) #Reverse the order of the array
            CP = CP + CP_right
        trap_outside = trap1D[index_edge] + CP
        local_minima = find_peaks(-trap_outside,distance = 10, prominence = 5e-4)   

        if len(local_minima[0])==0:
            print("No local minima found")
            return None, None, 0, y_outside, trap_outside, 0
        elif len(local_minima[0])==1 and local_minima[0][0] > 5:
            print("len(local_minima) = 1")
            return local_minima[0][0], y_outside[local_minima[0][0]], -local_minima[1]["prominences"][0], y_outside, trap_outside, CP[local_minima[0][0]]
        #        return np.min(np.real(trap_outside[local_minima[0]]))#, x_outside[local_minima[0]]-Simul.structure.y_edge
        elif len(local_minima[0])==1 and local_minima[0][0] <= 5:
            print('Local minima too close to the edge of the structure')
            return None, None,0, y_outside, trap_outside, 0
        else :
            arg = np.argmin(np.real(trap_outside[local_minima[0]]))
            return local_minima[0][arg], y_outside[local_minima[0][arg]], -local_minima[1]["prominences"][arg], y_outside, trap_outside, CP[local_minima[0][arg]]
        
    def get_trapfreq(self,c_outside,trap_pos,trap_1D):
        try :
            fit = np.polyfit(c_outside, trap_1D, 40)
            pass
        except np.linalg.LinAlgError:
            fit = np.polyfit(c_outside, trap_1D, 20)
            
        p = np.poly1d(fit)
        der_fit = np.real(np.gradient(p(c_outside),c_outside))
        der2_fit = np.gradient(der_fit,c_outside)
        index_min = np.argmin(np.abs(c_outside-trap_pos))
        moment2 = der2_fit[index_min]
        trap_freq = np.sqrt((moment2*pu.kB*pu.mK)/(self.system.atom.mass))*(1/(2*np.pi))*1e-3
        # trap_f = np.sqrt(moment2*pu.kB*pu.mK)
        return trap_freq
    
    #à mettre à jour
    def plot_trap1D_with_splitting(self,state,plane,axis, out_of_plane_coord, in_plane_coord,edge1, edge2 = None, add_CP = False):
        if axis == "X":
            self.coord = self.x
        elif axis == "Y":
            self.coord = self.y
        elif axis == "Z":
            self.coord = self.z
        else :
            print("axis must be either 'X', 'Y' or 'Z'")
            
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1]) 
        ax0 = plt.subplot(gs[0])
        plt.subplots_adjust(bottom=0.27)
        
        """Line plots for the 1st subplot showing the potential"""
        jet = cm = plt.get_cmap('Greys')
        cNorm  = colors.Normalize(vmin=-1, vmax=len(self.mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        
        index_edge1 = np.argmin(np.abs(self.coord -edge1))
        if self.coord[index_edge1] < edge1 :
            index_edge1 += 1
        index_edge = slice(index_edge1, None)

        if edge2 != None:
            index_edge2 = np.argmin(np.abs(self.coord -edge2))
            if self.coord[index_edge2] > edge2 :
                index_edge2 -= 1
            index_edge = slice(index_edge1, index_edge2)

        coord_outside = self.coord[index_edge] - edge1
        
        key = [x for x in list(self.trap_1D.keys()) if axis in x][-1]
        trap_outside = np.real(self.trap_1D[key][index_edge,:])
        # print(trap_outside)
        
        if add_CP == True:
            C3 = self.system.get_C3(self.material,self.system.groundstate)
            CP = np.array([-C3/(r**3)/pu.kB/pu.mK for r in coord_outside])
            if edge2 != None:
                CP_right = np.array([-C3/(r**3)/pu.kB/pu.mK for r in coord_outside[::-1]]) #Reverses the order of the list
                CP = CP + CP_right
            CP_mf = np.array([CP,]*len(self.mf)).transpose()
            trap_outside = np.real(self.trap_1D[key][index_edge,:]) + CP_mf

        a = []
        for k in range(len(self.mf)):
            colorVal = scalarMap.to_rgba(k)
            a = a + ax0.plot(coord_outside,trap_outside[:,k], color = colorVal, label = "m$_f$ = %s" %(self.mf[k]))

        if len(self.mf) == 1:
            b, = ax0.plot(coord_outside,np.real(self.trap.Pblue*np.real(self.mask_blue1D[itemgetter(0,1,3,4)(key)][index_edge])), color = 'blue')
            r, = ax0.plot(coord_outside,np.real(self.trap.Pred*np.real(self.mask_red1D[itemgetter(0,2,3,4)(key)][index_edge])), color = 'red')
        
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend()
        plt.xlabel(axis+" (nm)")
        plt.ylabel("E (mK)")
        plt.title("1D plot of trapping potential along %s " %(axis))


#         jet = cm = plt.get_cmap('Greys')
#         cNorm  = colors.Normalize(vmin=-1, vmax=len(self.mf))
#         scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#         a = []
#         key = list(self.trap_1D.keys())[-1]
#         for k in range(len(self.mf)):
#             colorVal = scalarMap.to_rgba(k)
#             a = a + ax0.plot(np.real(self.trap_1D[key])[:,k], color = colorVal, label = "m$_f$ = %s" %(self.mf[k]))

#         if len(self.mf) == 1:
#             b, = ax0.plot(np.real(self.trap.Pblue*self.mask_blue1D[axe]), color = 'blue')
#             r, = ax0.plot(np.real(self.trap.Pred*self.mask_red1D[axe]), color = 'red')

#         plt.legend()
#         plt.xlabel(axe+" (nm)")
#         plt.ylabel("E (mK)")
# #        plt.ylim(-5e-14,5e-14)
#         plt.title("1D plot of trapping potential along %s " %(axe))
        
                    
        """Matrix plot showing how the eigenvectors are composed in a second subplot"""
        if self.trap.nblue >= 1 and self.trap.nred >= 1:
            vec = self.trap.Pblue*self.vec1 + self.trap.Pred*self.vec2
        elif self.trap.red == 0 :
            vec = self.vec*self.trap.Pblue
        else :
            vec = self.vec*self.trap.Pbred
        ax1 = plt.subplot(gs[1])
        zeemanstates = np.arange(-self.f,self.f+1,1)
        im = ax1.imshow(abs(vec)**2,cmap="viridis")

        ax1.set_xticks(np.arange(len(zeemanstates)))
        ax1.set_yticks(np.arange(len(zeemanstates)))
        ax1.set_xlim(-0.5,len(zeemanstates)-0.5)
        ax1.set_ylim(-0.5,len(zeemanstates)-0.5)
        ax1.set_xticklabels(zeemanstates)
        ax1.set_yticklabels(-zeemanstates)
        text = []
        for i in range(len(zeemanstates)):
            textj = []
            for j in range(len(zeemanstates)):
                textj = textj + [ax1.text(j, i, round((abs(vec)**2)[i, j],2),
                            ha="center", va="center", color="w")]
            text.append(textj)

        ax1.set_title("Mixing of the Zeeman states")
        cax = plt.axes([0.95, 0.2, 0.03,0.73 ])
        cbar = fig.colorbar(mappable = im, cax = cax)
        fig.tight_layout()
        plt.show()

        axcolor = 'lightgoldenrodyellow'
        axPblue = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
        sPblue = Slider(axPblue, 'Pblue', 0, 30.0, valinit=1, valstep=0.5)

        axPred = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
        sPred = Slider(axPred, 'Pred', 0, 10.0, valinit=1, valstep=0.1)

        def update(val):
            Pb = sPblue.val
            Pr = sPred.val
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap = np.real(self.simulate1D(self.system.groundstate,self.f,self.mf,plane,axis,out_of_plane_coord, in_plane_coord,self.wavelength_index_blue,self.wavelength_index_red))
            ax0.set_ylim([trap.min(),trap.max()])
            
            for k in range(len(self.mf)):
                if add_CP == True :
                    trap_outside = np.real(self.trap_1D[key][index_edge,k]) + CP #/(pu.kB*coef_norm*pu.mK) + CP
                else:
                    trap_outside = np.real(self.trap_1D[key][index_edge,k])#/(pu.kB*coef_norm*pu.mK)
                a[k].set_ydata(trap_outside)
                
            if len(self.mf) == 1:
                b.set_ydata(np.real(self.trap.Pblue*np.real(self.mask_blue1D[itemgetter(0,1,3,4)(key)][index_edge])))#/(pu.kB*coef_norm*pu.mK))
                r.set_ydata(np.real(self.trap.Pred*np.real(self.mask_red1D[itemgetter(0,2,3,4)(key)][index_edge])))#/(pu.kB*coef_norm*pu.mK))
                
            # for k in range(len(self.mf)):
            #     a[k].set_ydata(np.real(self.trap_1D[key])[:,k])

            # if len(self.mf) == 1:
            #     b.set_ydata(np.real(self.trap.Pblue*self.mask_blue1D[axis]))
            #     r.set_ydata(np.real(self.trap.Pred*self.mask_red1D[axis]))
                
            if self.trap.nblue >= 1 and self.trap.nred >= 1:
                vec = self.trap.Pblue*self.vec1 + self.trap.Pred*self.vec2
            elif self.trap.red == 0 :
                vec = self.vec*self.trap.Pblue
            else :
                vec = self.vec*self.trap.Pbred
            
            im.set_data(abs(vec)**2)
            im.autoscale()
            # cbar.set_clim(vmin=np.min(abs(vec)**2),vmax=np.max(abs(vec)**2))
            
            for i in range(len(zeemanstates)):
                for j in range(len(zeemanstates)):
                    text[i][j].set_text(round((abs(vec)**2)[i, j],2))

    #            argmin_update = np.unravel_index(trap_1D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
    #            point.set_data(self.structure.period*x[argmin_update[0]]/nm,self.structure.period*y[argmin_update[1]+y_edge_ind[0]]/nm)
            fig.canvas.draw_idle()
                
        sPblue.on_changed(update)
        sPred.on_changed(update)
        
        return fig, ax0, ax1, sPblue, sPred
    
class Simulation():
    def __init__(self,system, structure, trap):
        self.system = system
        self.structure = structure
        self.trap = trap

        self.mask_red = {}
        self.mask_blue = {}
        self.trap_2D = {}

        self.mask_red1D = {}
        self.mask_blue1D = {}
        self.trap_1D = {}

        indices = []
        approx_wavelengths = []
        if self.structure.method == 'legume':
            i = 0
            for k in self.trap.wavelengths_dict:
                if self.trap.wavelengths_dict[k] is not None:
                    f0 = sc.c/self.trap.wavelengths_dict[k]
                    int_index = np.nanargmin(abs(self.structure.guided_freq*sc.c/self.structure.period-f0)) #gives index when array is flattened
                    mode_num = int_index//np.shape(self.structure.guided_freq)[1] #both lines to convert the flattened index in 1x2 array
                    k_num = int_index%np.shape(self.structure.guided_freq)[1]
                    indices.append((mode_num,k_num)) #can be called directly in a array in this form
                    approx_wavelengths.append(self.structure.period/self.structure.guided_freq[indices[i]])
                    i = i + 1
        elif self.structure.method == 'Lumerical' :
            for k in self.trap.wavelengths_dict:
                if self.trap.wavelengths_dict[k] is not None:
                    f0 = sc.c/self.trap.wavelengths_dict[k]
                    lmbda0 = self.trap.wavelengths_dict[k]
                    lmbda = np.squeeze(self.structure.structure.getresult("mode source","neff")["lambda"])
                    indexl0=np.argmin(abs(lmbda-lmbda0))
                    indices.append((indexl0,0))
                    approx_wavelengths.append(lmbda[indexl0])
        self.indices = indices
        self.approx_wavelengths = approx_wavelengths
        print(indices)
        print(approx_wavelengths)

#####################################                   Masks in 3D                  #################################
    def getE_convert_and_compute(self,state,f,mf_shift,plane,coord,N1,N2,wavelength_index):
        E,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index][1],self.indices[wavelength_index][0],N1,N2)
        res = np.zeros((len(x),len(y)),dtype = "complex")
        for i in range(len(y)):
            print("i =", i)
            for j in range(len(x)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,j],E['y'][i,j],E['z'][i,j])
                val, vec = self.system.potential(Ep,Em,E0,state,f,self.approx_wavelengths[wavelength_index])
                res[j,i] = val[mf_shift]
        return res, vec

    def getE_convert_and_compute_contrapropag(self,state,f,mf_shift,plane,coord,N1,N2,wavelength_index_1,wavelength_index_2):
        E1,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index_1][1],self.indices[wavelength_index_1][0],N1,N2)
        E2,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index_2][1],self.indices[wavelength_index_2][0],N1,N2)
        res = np.zeros((len(x),len(y)),dtype = "complex")
        E2['x'] = -E2['x']
        E = dict_sum(E1,E2)
        for i in range(len(y)):
            for j in range(len(x)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,j],E['y'][i,j],E['z'][i,j])
                val, vec  = self.system.potential_contrapropag(Ep,Em,E0,state,f,self.approx_wavelengths[wavelength_index_1],self.approx_wavelengths[wavelength_index_2])
                res[j,i] = val[mf_shift]
        return res, vec

    def set_masks(self,state,f,mf,plane,coord,N1,N2):
        mf_shift = mf + f
        try :
            self.mask_red[plane]
            self.mask_blue[plane]
            pass
        except KeyError:
            if self.trap.nblue == 0 and self.trap.nred == 0 :
                raise "Nothing to simulate"

            elif self.trap.nblue == 1 and self.trap.nred == 0 :
                res, vec = self.getE_convert_and_compute(state,f,mf_shift,plane,coord,N1,N2,0)
                self.mask_blue[plane] = res

            elif self.trap.nred == 1 and self.trap.nblue == 0:
                res, vec = self.getE_convert_and_compute(state,f,mf_shift,plane,coord,N1,N2,0)
                self.mask_red[plane] = res

            elif self.trap.nblue == 1 and self.trap.nred == 1:
                res1, self.vec1 = self.getE_convert_and_compute(state,f,mf_shift,plane,coord,N1,N2,0)
                res2, self.vec2 = self.getE_convert_and_compute(state,f,mf_shift,plane,coord,N1,N2,1)
                self.mask_blue[plane] = res1
                self.mask_red[plane] = res2

            elif self.trap.nblue == 2 and self.trap.nred == 0:
                res, vec = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,plane,coord,N1,N2,0,1)
                self.mask_blue[plane] = res

            elif self.trap.nblue == 0 and self.trap.nred == 2:
                res, vec = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,plane,coord,N1,N2,0,1)
                self.mask_red[plane] = res

            elif self.trap.nblue ==2 and self.trap.nred == 1:
                res1, vec1 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,plane,coord,N1,N2,0,1)
                res2, vec1 = self.getE_convert_and_compute(state,f,mf_shift,plane,coord,N1,N2,2)
                self.mask_blue[plane] = res1
                self.mask_red[plane] = res2

            elif self.trap.nblue == 1 and self.trap.nred == 2:
                res1, vec1 = self.getE_convert_and_compute(state,f,mf_shift,plane,coord,N1,N2,0)
                res2, vec2 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,plane,coord,N1,N2,1,2)
                self.mask_blue[plane] = res1
                self.mask_red[plane] = res2

            elif self.trap.nblue ==2 and self.trap.nred == 2 :
                res1, vec1 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,plane,coord,N1,N2,0,1)
                res2, vec2 = self.getE_convert_and_compute_contrapropag(state,f,mf_shift,plane,coord,N1,N2,2,3)
                self.mask_blue[plane] = res1
                self.mask_red[plane] = res2
            else :
                raise ValueError("nred and nblue must be integers between 0 and 2")

#            self.x = x
#            self.y = y


            print("Simulation completed, masks available")


#####################################                   Masks in 1D                   ################################
    def getE_convert_and_compute_1D(self,state,f,mf_shift,plane,axe,coord,N1,N2,wavelength_index):
        E,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index][1],self.indices[wavelength_index][0],N1,N2)
        if is_first(axe,plane):
            res = np.zeros((len(x),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(y)))
            for j in range(len(x)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][index,j],E['y'][index,j],E['z'][index,j])
                val, vec = self.system.potential(Ep,Em,E0,state,f,self.approx_wavelengths[wavelength_index])
                res[j] = val[mf_shift]
        elif is_second(axe,plane):
            res = np.zeros((len(y),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(x)))
            for i in range(len(y)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,index],E['y'][i,index],E['z'][i,index])
                val, vec = self.system.potential(Ep,Em,E0,state,f,self.approx_wavelengths[wavelength_index])
                res[i] = val[mf_shift]

        return res, vec

    def getE_convert_and_compute_contrapropag_1D(self,state,f,mf_shift,plane,axe,coord,N1,N2,wavelength_index_1,wavelength_index_2):
        E1,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index_1][1],self.indices[wavelength_index_1][0],N1,N2)
        E2,x,y = self.structure.get_fields("e",plane,coord,self.indices[wavelength_index_2][1],self.indices[wavelength_index_2][0],N1,N2)
        E2['x'] = -E2['x']
        E = dict_sum(E1,E2)
        if is_first(axe,plane):
            res = np.zeros((len(x),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(y)))
            for j in range(len(x)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][index,j],E['y'][index,j],E['z'][index,j])
                val, vec = self.system.potential_contrapropag(Ep,Em,E0,state,f,self.approx_wavelengths[wavelength_index_1],self.approx_wavelengths[wavelength_index_2])
                res[j] = val[mf_shift]
        elif is_second(axe,plane):
            res = np.zeros((len(y),len(mf_shift)),dtype = "complex")
            index = np.argmin(np.abs(np.array(x)))
            for i in range(len(y)):
                E0, Ep, Em = convert_fields_to_spherical_basis(E['x'][i,index],E['y'][i,index],E['z'][i,index])
                val, vec = self.system.potential_contrapropag(Ep,Em,E0,state,f,self.approx_wavelengths[wavelength_index_1],self.approx_wavelengths[wavelength_index_2])
                res[i] = val[mf_shift]

        return res, vec

    def set_masks1D(self,state,f,mf,plane,axe,coord,N1,N2):
        mf_shift = mf + f
        try :
            self.mask_red1D[axe]
            self.mask_blue1D[axe]
            pass
        except KeyError:

            if self.trap.nblue == 0 and self.trap.nred == 0 :
                raise "Nothing to simulate"

            elif self.trap.nblue == 1 and self.trap.nred == 0 :
                res, self.vec = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0)
                self.mask_blue1D[axe] = res

            elif self.trap.nblue == 0 and self.trap.nred == 1 :
                res, self.vec = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0)
                self.mask_red1D[axe] = res

            elif self.trap.nblue == 1 and self.trap.nred == 1:
                res1, self.vec1 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0)
                res2, self.vec2 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord,N1,N2,1)
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            elif self.trap.nblue == 2 and self.trap.nred == 0:
                res, self.vec = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0,1)
                self.mask_blue1D[axe] = res

            elif self.trap.nblue == 0 and self.trap.nred == 2:
                res, self.vec = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0,1)
                self.mask_red1D[axe] = res

            elif self.trap.nblue ==2 and self.trap.nred == 1:
                res1, self.vec1 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0,1)
                res2, self.vec2 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord,N1,N2,2)
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            elif self.trap.nblue == 1 and self.trap.nred == 2:
                res1, self.vec1 = self.getE_convert_and_compute_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0)
                res2, self.vec2 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord,N1,N2,1,2)
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            elif self.trap.nblue ==2 and self.trap.nred == 2 :
                res1, self.vec1 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord,N1,N2,0,1)
                res2, self.vec2 = self.getE_convert_and_compute_contrapropag_1D(state,f,mf_shift,plane,axe,coord,N1,N2,2,3)
                self.mask_blue1D[axe] = res1
                self.mask_red1D[axe] = res2

            else :
                raise ValueError("nred and nblue must be integers between 0 and 2")
            print("Simulation completed, masks available")


#####################################                   Simulation                    ################################
    def simulate(self,state,f,mf,plane,coord,N1,N2):
        mf = np.array(mf)
        if mf.any() > f or mf.any() < -f:
            raise ValueError("m_F should be in the interval [-F,F]")
        else:
            self.mf = mf
            self.f = f
        self.set_masks(state,self.f,self.mf,plane,coord,N1,N2)
        if self.trap.nblue == 0:
            self.trap_2D[plane] = self.trap.Pred*self.mask_red[plane]
        elif self.trap.nred == 0:
            self.trap_2D[plane] = self.trap.Pblue*self.mask_blue[plane]
        else:
            self.trap_2D[plane] = self.trap.Pblue*self.mask_blue[plane] + self.trap.Pred*self.mask_red[plane]

        return self.trap_2D[plane]


    def simulate1D(self,state,f,mf,plane,axe,coord,N1,N2):
        """ mf has to be a list ! """
        mf = np.array(mf)
        if mf.any() > f or mf.any() < -f:
            raise ValueError("m_F should be in the interval [-F,F]")
        else:
            self.mf = mf
            self.f = f

        self.set_masks1D(state,f,mf,plane,axe,coord,N1,N2)
        if self.trap.nblue == 0:
            self.trap_1D[axe] = self.trap.Pred*self.mask_red1D[axe]
        elif self.trap.nred == 0:
            self.trap_1D[axe] = self.trap.Pblue*self.mask_blue1D[axe]
        else:
            self.trap_1D[axe] = self.trap.Pblue*self.mask_blue1D[axe] + self.trap.Pred*self.mask_red1D[axe]

        return self.trap_1D[axe]

    def get_trapdepth(self,plane,axe,mat_hand = None):
        """
        Computes trap depth for a trap along the specified axis. Checks whereas a 2D
        trap has been computed or just a 1D quick one. Returns position of minimum and trap depth in mK

        Args:
            plane (str): Either 'XY', 'YZ' or 'XZ'
            axe (str): 'X', 'Y' or 'Z'. Has to be in the plane.
        """
        if hasattr(self.structure,"material"):
            C3 = self.system.get_C3(self.structure.material,self.system.groundstate)
        else:
            if mat_hand is None:
                raise ValueError("Please give a material string to the function")
            else:
                C3 = self.system.get_C3(mat_hand,self.system.groundstate)

#        C3 = self.structure.material.C3["Rb"]
        if is_first(axe,plane):
            try :
                self.structure.x_edge
                pass
            except AttributeError:
                self.structure.x_edge = 0
            x_indborder = np.argmin(abs(self.x-self.structure.x_edge))
            if self.x[x_indborder] < self.structure.x_edge :
                x_indborder += 1
            x_outside = self.x[x_indborder:] - self.structure.x_edge
            CP = [-C3/(r**3)/pu.kB/pu.mK for r in x_outside]
            index = np.argmin(np.abs(np.array(self.y)))
            try :
                self.trap_2D[plane]
                trap_outside = np.real(self.trap_2D[plane][x_indborder:,index]/(pu.kB*5e12*pu.mK) + CP)
                pass
            except KeyError:
                trap_outside = self.trap_1D[axe][x_indborder:]/(pu.kB*5e12*pu.mK) + CP
            c_outside = x_outside

        if is_second(axe,plane):
            try :
                self.structure.y_edge
#                self.structure.y_edge = self.structure.y_edge*self.structure.period
                pass
            except AttributeError:
                self.structure.y_edge = 0
            y_indborder = np.argmin(abs(self.y-self.structure.y_edge*self.structure.period))
            if self.y[y_indborder] < self.structure.y_edge*self.structure.period:
                y_indborder += 1
            y_outside = self.y[y_indborder:] - self.structure.y_edge*self.structure.period
            CP = [-C3/(r**3)/pu.kB/pu.mK for r in y_outside]
            index = np.argmin(np.abs(np.array(self.x)))
            try :
                self.trap_2D[plane]
                trap_outside = np.real(self.trap_2D[plane][index,y_indborder:]/(pu.kB*5e12*pu.mK) + CP)
                pass
            except KeyError:
                trap_outside = self.trap_1D[axe][x_indborder:]/(pu.kB*5e12*pu.mK) + CP
            c_outside = y_outside
        local_minima = find_peaks(-trap_outside,distance = 10, prominence = 5e-4)

        if len(local_minima[0])==0:
            trap_pos,trap_depth = 0,0
        elif len(local_minima[0])==1 and local_minima[0][0] > 5:
            if trap_outside[local_minima[1]["left_bases"][0]] >= 0:
                trap_pos,trap_depth = c_outside[local_minima[0]], -local_minima[1]["prominences"][0]
            else :
                trap_pos,trap_depth = 0,0
        elif len(local_minima[0])==1 and local_minima[0][0] <= 5:
            trap_pos,trap_depth = 0,0
        elif len(local_minima[0])>1:
            arg = np.argmin(np.real(trap_outside[local_minima[0]]))
            if trap_outside[local_minima[1]["left_bases"][arg]] >= 0:
                trap_pos,trap_depth = c_outside[local_minima[0]], -local_minima[1]["prominences"][arg]
            else :
                trap_pos,trap_depth = 0,0

        return c_outside, trap_outside, trap_pos, trap_depth

    def get_trapfreq(self,c_outside,trap_pos,trap_1D):
        fit = np.polyfit(c_outside, trap_1D, 40)
        p = np.poly1d(fit)
        der_fit = np.real(np.gradient(p(c_outside),c_outside))
        der2_fit = np.gradient(der_fit,c_outside)
        index_min = np.argmin(np.abs(c_outside-trap_pos))
        moment2 = der2_fit[index_min]
        trap_freq = np.sqrt(moment2*pu.kB*1e-3*1e18)/(self.system.atom.mass)*(1e-3)/(2*np.pi)
        return trap_freq

    def get_trapdepth_and_freq(self,plane,axe):
        c_outside, trap1D, pos, depth = self.get_trapdepth(plane, axe)
        if depth != 0:
            trap_freq = self.get_trapfreq(c_outside, pos, trap1D)
        else:
            print("Atoms are not trapped in the %s direction" %(axe))
            trap_freq = 0
        return trap_freq
    
    def showLevelMixing(self,state,f,lmbda):
        vec = self.trap.Pblue*self.vec1 + self.trap.Pred*self.vec2
        zeemanstates = np.arange(-f,f+1,1)
        fig, ax = plt.subplots()
        im = ax.imshow(abs(vec)**2,cmap="viridis")

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(zeemanstates)))
        ax.set_yticks(np.arange(len(zeemanstates)))
        ax.set_xlim(-0.5,len(zeemanstates)-0.5)
        ax.set_ylim(-0.5,len(zeemanstates)-0.5)
        # ... and label them with the respective list entries
        ax.set_xticklabels(zeemanstates)
        ax.set_yticklabels(-zeemanstates)

        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #        rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(zeemanstates)):
            for j in range(len(zeemanstates)):
                text = ax.text(j, i, round((abs(vec)**2)[i, j],2),
                            ha="center", va="center", color="w")

        ax.set_title("Mixing of the Zeeman states")
        fig.colorbar(im)
        fig.tight_layout()
        plt.show()

#    def averaged_trap():
#        for k in range
#        self.set_masks1D(state,f,mf,plane,ax,coord,N1,N2)


    def plot_trap(self,plane,N1,N2):
        """Method for plotting the trap in any 2D plane. Comes with sliders to change the powers
        of the red and blue lasers. Can plot a non-uniform grid.

        Args:
            plane (str):  Either 'XY', 'YZ' or 'XZ'.
            N1 (int): Number of points where the trap is computed along the first direction ('X' if the plane is 'XY').
            N2 (int): Number of points along the second direction ('Y' if 'XY').

        """
        if plane == "XY":
    #           eps_r = legume.viz.eps_ft(self.structure.gme, figsize=2., cbar=True, Nx=N1, Ny=N2)
            x, y, eps_rr = legume.viz.eps_xy(self.structure.structure,self.structure.thickness/2,Ny=N2,Nx=N1)
    #        _, x_plot, y_plot = self.structure.gme.get_field_xy(field="e", kind=0, mind=0, z=self.structure.thickness/2, Nx=N1, Ny=N2)
    #            fig = plt.figure()
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.5, bottom=0.05)
            a = ax.pcolormesh(self.structure.period*x/pu.nm,self.structure.period*y/pu.nm,np.transpose(np.real(self.trap_2D[plane])),shading = "gouraud")

#            a = plt.imshow(np.transpose(np.real(self.trap_2D[plane])),origin = "lower", extent = [self.structure.period*x[0]/pu.nm,self.structure.period*x[-1]/pu.nm,self.structure.period*y[0]/pu.nm,self.structure.period*y[-1]/pu.nm],vmin = -1e-12,vmax = 1e-12)
            cbar = plt.colorbar(a)

    #            plt.clim(-5e-13,5e13)
            y_edge_ind = [i for i in range(len(y)) if y[i] >=np.ceil(10*self.structure.y_edge)/10+0.1]
            index_min = np.unravel_index(self.trap_2D[plane][:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),self.trap_2D[plane][:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
            point, = plt.plot(self.structure.period*x[index_min[0]]/pu.nm,self.structure.period*y[index_min[1]+y_edge_ind[0]]/pu.nm,'ro')
            plt.contour(self.structure.period*x/pu.nm,self.structure.period*y/pu.nm,np.real(eps_rr),[self.structure.material.n])
            plt.xlabel("x (nm)")
            plt.ylabel("y (nm)")
            plt.ylim(-440, 440)

        if plane == "YZ":
            y, z, eps_rr = legume.viz.eps_yz(self.structure.structure,0,Ny=N1,Nz=N2)
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.5, bottom=0.05)

            a = ax.pcolormesh(self.structure.period*y/pu.nm,self.structure.period*z/pu.nm,np.transpose(np.real(self.trap_2D[plane])),shading = "gouraud")
            cbar = plt.colorbar(a)
            y_edge_ind = [i for i in range(len(y)) if y[i] >=np.ceil(10*self.structure.y_edge)/10+0.1]
            index_min = np.unravel_index(self.trap_2D[plane][y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),self.trap_2D[plane][y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
            point, = plt.plot(self.structure.period*y[index_min[0]+y_edge_ind[0]]/pu.nm,self.structure.period*z[index_min[1]]/pu.nm,'ro')
            plt.contour(self.structure.period*y/pu.nm,self.structure.period*z/pu.nm,np.real(eps_rr),[self.structure.material.n])
            plt.xlabel("y (nm)")
            plt.ylabel("z (nm)")
            plt.ylim(-440, 440)

        plt.title("2D plot of trapping potentiel in the %s plane" %(plane))

        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'

        axPblue = plt.axes([0.15, 0.1, 0.03, 0.75], facecolor=axcolor)
        sPblue = Slider(axPblue, 'Pblue', 0, 30.0, valinit=self.trap.Pblue, valstep=0.5,orientation='vertical')

        axPred = plt.axes([0.23, 0.1, 0.03, 0.75], facecolor=axcolor)
        sPred = Slider(axPred, 'Pred', 0, 10.0, valinit=self.trap.Pred, valstep=0.2,orientation='vertical')

        axlmbdablue = plt.axes([0.31, 0.1, 0.03, 0.75], facecolor=axcolor)
        slmbdablue = Slider(axlmbdablue, 'lmbdablue', 700, 780, valinit=self.approx_wavelengths[0]/pu.nm, valstep=5,orientation='vertical')

        axlmbdared = plt.axes([0.39, 0.1, 0.03, 0.75], facecolor=axcolor)
        slmbdared = Slider(axlmbdared, 'lmbdared', 780, 900, valinit=self.approx_wavelengths[-1]/pu.nm, valstep=5,orientation='vertical')

        def update(val):
            Pb = sPblue.val
            Pr = sPred.val
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap_2D = self.simulate(self.system.groundstate,self.f,self.mf,plane,self.structure.thickness/2,N1=N1,N2=N2)
            a.set_array(np.transpose(np.real(trap_2D)).ravel())
#            a.set_clim(vmin=np.min(np.real(trap_2D)),vmax=np.max(np.real(trap_2D)))
#            cbar.draw_all()
#            a.set_data(np.transpose(np.real(self.trap_2D[plane])))
            if plane == "XY":
                argmin_update = np.unravel_index(trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
                point.set_data(self.structure.period*x[argmin_update[0]]/pu.nm,self.structure.period*y[argmin_update[1]+y_edge_ind[0]]/pu.nm)

            elif plane == "YZ":
                argmin_update = np.unravel_index(trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
                point.set_data(self.structure.period*y[argmin_update[0]+y_edge_ind[0]]/pu.nm,self.structure.period*z[argmin_update[1]]/pu.nm)
            # print(argmin_update)
            fig.canvas.draw_idle()

        sPblue.on_changed(update)
        sPred.on_changed(update)
        slmbdared.on_changed(update)
        slmbdablue.on_changed(update)
        plt.show()

    def plot_trap_XY_YZ(self,N1,N2):
        """Method for plotting the trap in both XY and YZ 2D plane. Comes with sliders to change the powers
        of the red and blue lasers. Can plot a non-uniform grid.

        Args:
            N1 (int): Number of points where the trap is computed along the first direction ('X' if the plane is 'XY').
            N2 (int): Number of points along the second direction ('Y' if 'XY').
        """
        x, y, eps_xy = legume.viz.eps_xy(self.structure.structure,self.structure.thickness/2,Ny=N2,Nx=N1)
        y_edge_ind = [i for i in range(len(y)) if y[i] >=np.ceil(10*self.structure.y_edge)/10+0.1]

        fig, ax = plt.subplots(2)
        plt.subplots_adjust(left=0.3, right=0.8, bottom=0.05)

        a = ax[0].pcolormesh(self.structure.period*x/pu.nm,self.structure.period*y/pu.nm,np.transpose(np.real(self.trap_2D["XY"])),shading = "gouraud")
        index_min_xy = np.unravel_index(self.trap_2D["XY"][:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),self.trap_2D["XY"][:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
        point_xy, = ax[0].plot(self.structure.period*x[index_min_xy[0]]/pu.nm,self.structure.period*y[index_min_xy[1]+y_edge_ind[0]]/pu.nm,'ro')
        ax[0].contour(self.structure.period*x/pu.nm,self.structure.period*y/pu.nm,np.real(eps_xy),[self.structure.material.n])
        ax[0].set(xlabel="x (nm)", ylabel="y (nm)")
#        ax[0].set_ylim([-440, 440])

        y, z, eps_yz = legume.viz.eps_yz(self.structure.structure,0,Ny=N2,Nz=N1)

        b = ax[1].pcolormesh(self.structure.period*y/pu.nm,self.structure.period*z/pu.nm,np.transpose(np.real(self.trap_2D["YZ"])),shading = "gouraud")
        index_min_yz = np.unravel_index(self.trap_2D["YZ"][y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),self.trap_2D["YZ"][y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
        point_yz, = ax[1].plot(self.structure.period*y[index_min_yz[0]+y_edge_ind[0]]/pu.nm,self.structure.period*z[index_min_yz[1]]/pu.nm,'ro')
        ax[1].contour(self.structure.period*y/pu.nm,self.structure.period*z/pu.nm,np.real(eps_yz),[self.structure.material.n])
        ax[1].set(xlabel="x (nm)", ylabel="z (nm)")
#        ax[1].set_xlim([-440, 440])

        cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.8])
        fig.colorbar(a, cax=cbar_ax)
#        plt.title("2D plot of trapping potentiel in the %s plane" %(plane))

        ax[0].margins(x=0)
        ax[1].margins(x=0)

        axcolor = 'lightgoldenrodyellow'

        axPblue = fig.add_axes([0.1, 0.1, 0.03, 0.75], facecolor=axcolor)
        sPblue = Slider(axPblue, 'Pblue', 0, 30.0, valinit=self.trap.Pblue, valstep=0.5,orientation='vertical')

        axPred = fig.add_axes([0.18, 0.1, 0.03, 0.75], facecolor=axcolor)
        sPred = Slider(axPred, 'Pred', 0, 10.0, valinit=self.trap.Pred, valstep=0.2,orientation='vertical')

        def update(val):
            Pb = sPblue.val
            Pr = sPred.val
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap_2D_XY = self.simulate(self.system.groundstate,self.f,self.mf,"XY",self.structure.thickness/2,N1=N1,N2=N2)
            trap_2D_YZ = self.simulate(self.system.groundstate,self.f,self.mf,"YZ",0,N1=N2,N2=N1)
            a.set_array(np.transpose(np.real(trap_2D_XY)).ravel())
            b.set_array(np.transpose(np.real(trap_2D_YZ)).ravel())
            a.set_clim(vmin=np.min(np.real(trap_2D_XY)),vmax=np.max(np.real(trap_2D_XY)))
            argmin_update_xy = np.unravel_index(trap_2D_XY[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D_XY[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
            point_xy.set_data(self.structure.period*x[argmin_update_xy[0]]/pu.nm,self.structure.period*y[argmin_update_xy[1]+y_edge_ind[0]]/pu.nm)
            argmin_update_yz = np.unravel_index(trap_2D_YZ[y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),trap_2D_YZ[y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
            point_yz.set_data(self.structure.period*y[argmin_update_yz[0]+y_edge_ind[0]]/pu.nm,self.structure.period*z[argmin_update_yz[1]]/pu.nm)
            fig.canvas.draw_idle()

        sPblue.on_changed(update)
        sPred.on_changed(update)
        plt.show()

    def plot_trap1D(self,plane,axe, coord, N1,N2):       
        fig = plt.figure()
#        fig, ax = plt.subplots(1,2)
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.27)
        jet = cm = plt.get_cmap('Greys')
        cNorm  = colors.Normalize(vmin=-1, vmax=len(self.mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        a = []
        for k in range(len(self.mf)):
            colorVal = scalarMap.to_rgba(k)
            a = a + plt.plot(np.real(self.trap_1D[axe])[:,k], color = colorVal, label = "m$_f$ = %s" %(self.mf[k]))

        if len(self.mf) == 1:
            b, = plt.plot(np.real(self.trap.Pblue*self.mask_blue1D[axe]), color = 'blue')
            r, = plt.plot(np.real(self.trap.Pred*self.mask_red1D[axe]), color = 'red')

        plt.legend()
        plt.xlabel(axe+" (nm)")
        plt.ylabel("E (mK)")
        plt.ylim(-5e-14,5e-14)
        plt.title("1D plot of trapping potential along %s " %(axe))

        axcolor = 'lightgoldenrodyellow'
        axPblue = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
        sPblue = Slider(axPblue, 'Pblue', 0, 30.0, valinit=1, valstep=0.5)

        axPred = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
        sPred = Slider(axPred, 'Pred', 0, 10.0, valinit=1, valstep=0.1)

        def update(val):
            Pb = sPblue.val
            #print(Pb)
            Pr = sPred.val
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap = self.simulate1D(self.system.groundstate,self.f,self.mf,plane,axe,coord,N1=N1,N2=N2)
            plt.subplot(111)
            plt.ylim(trap.min(),trap.max())

            for k in range(len(self.mf)):
                a[k].set_ydata(np.real(self.trap_1D[axe])[:,k])

            if len(self.mf) == 1:
                b.set_ydata(np.real(self.trap.Pblue*self.mask_blue1D[axe]))
                r.set_ydata(np.real(self.trap.Pred*self.mask_red1D[axe]))

    #            argmin_update = np.unravel_index(trap_1D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
    #            point.set_data(self.structure.period*x[argmin_update[0]]/nm,self.structure.period*y[argmin_update[1]+y_edge_ind[0]]/nm)
            fig.canvas.draw_idle()

        sPblue.on_changed(update)
        sPred.on_changed(update)
        plt.show()
        
    def plot_trap1D_with_splitting(self,plane,axe, coord, N1,N2):
        if self.trap.nblue >= 1 and self.trap.nred >= 1:
            vec = self.trap.Pblue*self.vec1 + self.trap.Pred*self.vec2
        elif self.trap.red == 0 :
            vec = self.vec*self.trap.Pblue
        else :
            vec = self.vec*self.trap.Pbred
        
        fig = plt.figure()
#        fig, ax = plt.subplots(1,2)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1]) 
        ax0 = plt.subplot(gs[0])
#        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.27)
        
        """Line plots for the 1st subplot showing the potential"""
        
        jet = cm = plt.get_cmap('Greys')
        cNorm  = colors.Normalize(vmin=-1, vmax=len(self.mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        a = []
        for k in range(len(self.mf)):
            colorVal = scalarMap.to_rgba(k)
            a = a + ax0.plot(np.real(self.trap_1D[axe])[:,k], color = colorVal, label = "m$_f$ = %s" %(self.mf[k]))

        if len(self.mf) == 1:
            b, = ax0.plot(np.real(self.trap.Pblue*self.mask_blue1D[axe]), color = 'blue')
            r, = ax0.plot(np.real(self.trap.Pred*self.mask_red1D[axe]), color = 'red')

        plt.legend()
        plt.xlabel(axe+" (nm)")
        plt.ylabel("E (mK)")
#        plt.ylim(-5e-14,5e-14)
        plt.title("1D plot of trapping potential along %s " %(axe))

        axcolor = 'lightgoldenrodyellow'
        axPblue = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
        sPblue = Slider(axPblue, 'Pblue', 0, 30.0, valinit=1, valstep=0.5)

        axPred = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
        sPred = Slider(axPred, 'Pred', 0, 10.0, valinit=1, valstep=0.1)

        def update(val):
            Pb = sPblue.val
            #print(Pb)
            Pr = sPred.val
            self.trap.set_powers(Pblue = Pb, Pred = Pr)
            trap = self.simulate1D(self.system.groundstate,self.f,self.mf,plane,axe,coord,N1=N1,N2=N2)
#            plt.subplot(111)
            ax0.set_ylim([trap.min(),trap.max()])
#            plt.ylim(trap.min(),trap.max())

            for k in range(len(self.mf)):
                a[k].set_ydata(np.real(self.trap_1D[axe])[:,k])

            if len(self.mf) == 1:
                b.set_ydata(np.real(self.trap.Pblue*self.mask_blue1D[axe]))
                r.set_ydata(np.real(self.trap.Pred*self.mask_red1D[axe]))

    #            argmin_update = np.unravel_index(trap_1D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
    #            point.set_data(self.structure.period*x[argmin_update[0]]/nm,self.structure.period*y[argmin_update[1]+y_edge_ind[0]]/nm)
            fig.canvas.draw_idle()
                
        sPblue.on_changed(update)
        sPred.on_changed(update)
        
            
        """Matrix plot showing how the eigenvectors are composed in a second subplot"""
        ax1 = plt.subplot(gs[1])
        zeemanstates = np.arange(-self.f,self.f+1,1)
        im = ax1.imshow(abs(vec)**2,cmap="viridis")

        # We want to show all ticks...
        ax1.set_xticks(np.arange(len(zeemanstates)))
        ax1.set_yticks(np.arange(len(zeemanstates)))
        ax1.set_xlim(-0.5,len(zeemanstates)-0.5)
        ax1.set_ylim(-0.5,len(zeemanstates)-0.5)
        # ... and label them with the respective list entries
        ax1.set_xticklabels(zeemanstates)
        ax1.set_yticklabels(-zeemanstates)
        for i in range(len(zeemanstates)):
            for j in range(len(zeemanstates)):
                text = ax1.text(j, i, round((abs(vec)**2)[i, j],2),
                            ha="center", va="center", color="w")

        ax1.set_title("Mixing of the Zeeman states")
        cax = plt.axes([0.95, 0.2, 0.03,0.73 ])
        fig.colorbar(mappable = im, cax = cax)
        fig.tight_layout()
        
        plt.show()


 
# 