import legume
import nanotrappy.utils.physicalunits as pu
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.integrate import simps
#from nanotrappy.utils.materials import *
from scipy.optimize import fsolve, root
from scipy.special import kn, jv, j0, j1, k0, k1, jvp, kvp #using the j0, j1, k0, k1 is faster; jvp, kvp are derivatives
import sys
#import construction as ct
from collections import OrderedDict
sys.path.append(r"C:\\Program Files\\Lumerical\\v202\\api\\python")
try:
    import lumapi
except:
    pass

freqpoints = 200
freqstart = 280e12
freqstop = 500e12

def jp(n,x):
    return 0.5*(jv(n-1,x)-jv(n+1,x))

def kp(n,x):
    return -0.5*(kn(n-1,x)+kn(n+1,x))

class Structure():
    """
    This is the class that implements the nanostructures that will be simulated with GME solver.
    
    Note:
        Structures such as W1 waveguide, Nanobeam or Nanofiber are already implemented. Feel free to add your own!
        Each of these structure is a inherited class from Structure. Don't call Structure directly.
        
    Attributes:
        PreCompute (bool): If True, the band diagram will be simulated upon definition of the structure.
        method (str): "legume" or "Lumerical". Defines the solver that will be used for simulating the fields (deprecated)
        gmax (int): For the GME solver, maximum reciprocal lattice wave-vector length in units of 2pi/a. Increasing it may increase dramatically the computation time. Usually gmax around 4 offers already good accuracy. Default is 3
        k_min (float, [0,0.5]): Starting wavevector for computing the band diagram. 
        k_max (float, [0,0.5]): End wavevector for computing the band diagram.
        N_k (int): Number of sampled wavevector for the band diagram computation
        numeig (int): Number of bands computed at each step. Default is 20.
    """
    def __init__(self,structure,PreCompute,method,gmax, k_min,k_max,N_k,numeig):
        self.structure = structure
        self.method = method
        if method == "legume" :
            self.gme = legume.GuidedModeExp(self.structure, gmax=gmax)
            if PreCompute == True:
                path = self.structure.lattice.bz_path([[k_min*2*np.pi,0], [k_max*2*np.pi,0]], [N_k])
                self.gme.run(kpoints=path["kpoints"], numeig=numeig, compute_im=False)
        elif method == "Lumerical" :
 
            self.structure.addfdtd(x = 0, y = 0, z = 0, x_span = 2000e-9, y_span = 2000e-9, z_span=500e-9, simulation_time = 3000, mesh_accuracy = 5, pml_profile=2, use_early_shutoff=0, x_min_bc = "Anti-Symmetric", y_min_bc = "Symmetric")
                
            propsxy = OrderedDict([("name", "Field XY"),("monitor type", "2D Z-normal"),("x", 0.),("y", 0),("z", 0.),("x span", 1e-6),("y span", 1e-6)])
            indexpropxy = OrderedDict([("name", "Index XY"),("monitor type", "2D Z-normal"),("x", 0.),("y", 0),("z", 0),("x span", 1e-6),("y span", 1e-6)])
            self.structure.addindex(properties = indexpropxy)
            self.structure.addpower(properties = propsxy)
        
            modesource = OrderedDict([("name", "mode source"),("injection axis", "z"),("x", 0.),("y", 0),("z", 0.),("x span", 2.1e-6),("y span", 2.1e-6),("wavelength start", 6e-7),("wavelength stop", 1e-6),("mode selection","fundamental mode"),("frequency dependent profile",1),("number of field profile samples",20)])
            self.structure.addmode(properties = modesource)
            
        # else :
        #     raise ValueError("Supported calculation method are 'legume' (GME) or 'Lumerical' (FDTD) only")
                
        
    def show_structure(self,N1,N2):
        if self.method == "legume":
            legume.viz.structure(self.structure, xz=True, figsize=2, cbar=True, Nx=N1, Ny=N2)
#        elif self.method == "Lumerical":
#            self.structure
            
    def show_gme(self,N1,N2):
        legume.viz.eps_ft(self.gme, figsize=2., cbar=True, Nx=N1, Ny=N2)

    def simulate(self,k_min, k_max, N_k,numeig=20):
        """This method compute the band diagram of the structure (without plotting it). Not needed if 'PreCompute' is True when defining the structure.
                
        Args:
            k_min (float, [0,0.5]): Starting wavevector for computing the band diagram. 
            k_max (float, [0,0.5]): End wavevector for computing the band diagram.
            N_k (int): Number of sampled wavevector for the band diagram computation
            numeig (int): Number of bands computed at each step. Default is 20.
            
        Note:
            Defines a path that can be used next to access some result of the simulations
            
        """
        path = self.structure.lattice.bz_path([[k_min*2*np.pi,0], [k_max*2*np.pi,0]], [N_k])
        self.gme.run(kpoints=path["kpoints"], numeig=numeig, compute_im=False)
        
    def simulate_1k(self,kx,ky,numeig=20):
        self.gme.run(kpoints=np.array([[kx],[ky]]), numeig=numeig, compute_im=False)
        
    def plot_fields(self,field,plane, coord,k_num,mode_num,component,N1=40,N2=40):
        if plane == "XY" :
            fig = legume.viz.field(self.gme, z=coord, field=field, kind=k_num, mind=mode_num, component=str(component), val='abs', N1=N1, N2=N2)
        elif plane == "YZ":
            fig = legume.viz.field(self.gme, x=coord, field=field, kind=k_num, mind=mode_num, component=str(component), val='abs', N1=N1, N2=N2)
        elif plane == "XZ" :
            fig = legume.viz.field(self.gme, y=coord, field=field, kind=k_num, mind=mode_num, component=str(component), val='abs', N1=N1, N2=N2)
        else : 
            raise ValueError("The specified planes should be in ['XY','YZ','XZ']")
        print("k_x = ", self.gme.kpoints[0,k_num]/2/np.pi, "k_y = ", self.gme.kpoints[1,k_num]/2/np.pi, " , f = ", self.gme.freqs[k_num,mode_num])
    
    def get_fields_xy(self,field,coord,k_num,mode_num,N1,N2):
        E, x, y = self.gme.get_field_xy(field=field, kind=k_num, mind=mode_num, z=coord, Nx=N1, Ny=N2)
        return E, x, y
        
    def get_fields_yz(self,field,coord,k_num,mode_num,N1,N2):
        E, y, z = self.gme.get_field_yz(field=field, kind=k_num, mind=mode_num, x=coord, Ny=N1, Nz=N2)
        return E, y, z
    
    def get_fields_xz(self,field,coord,k_num,mode_num,N1,N2):
        E, x, z = self.gme.get_field_xz(field=field, kind=k_num, mind=mode_num, y=coord, Nx=N1, Nz=N2)
        return E, x, z
    
    def get_fields(self,field, plane,coord,k_num,mode_num,N1,N2):
        """This method returns all the components of a specified electric or magnetic mode once the simulation for that mode has been run.
                
        Args:
            field (str): {'e','h'} Field we want to get, electric or magnetic.
            plane (str): {'XY','YZ','XZ'} Only 2D fields can be called. Specifies the plane in which we want it.
            coord (float): Position of the plane in the 3rd axis.
            k_num (int): Wavevector number (defined from 'path') of the mode wanted.
            mode_num (int): Band number of the mode of interest.
            N1 (int): Sampling points in the first axis direction ("x" if plane "XY" is given)
            N2 (int): Sampling points in the second axis direction

        Returns:
            (tuple): tuple containing 
            
                (dict): Electric or magnetic field in the given wavevector, mode and plane. Dictionnary of 2D arrays with keys {'X','Y','Z'} for each component of the field.
                (list): First coordinate axis ("x" if plane "XY" is given)
                (list): Second coordinate axis ("y" if plane "XY" is given)
            
        Note:
            Does not plot the fields.
            
        """
        
        if self.method == 'legume':
            if plane == "XY":
                return self.get_fields_xy(field,coord,k_num,mode_num,N1,N2)
            elif plane == "YZ":
                return self.get_fields_yz(field,coord,k_num,mode_num,N1,N2)
            elif plane == "XZ":
                return self.get_fields_xz(field,coord,k_num,mode_num,N1,N2)
            else :
                raise ValueError("Allowed plane are 'XY','YZ' and 'XZ'")
        elif self.method == 'Lumerical':
            data = self.structure.getresult("mode source","mode profile")
            lmbda = np.squeeze(self.structure.getresult("mode source","neff")["lambda"])
            x = np.squeeze(data["x"])
            y = np.squeeze(data["y"])
            self.z = np.squeeze(data["z"])
            self.Etot = np.squeeze(data["E"]) #4 dimensional matrix (x,y,freq,component)
            lmbda_approx = lmbda[mode_num]
            Etot = self.Etot[:,:,mode_num,:]
            E = {}
            E['x'] = Etot[:,:,0]
            E['y'] = Etot[:,:,1]
            E['z'] = Etot[:,:,2]
            return E, x, y
            
    
    def plot_intensity(self,field,plane,coord,k_num,mode_num,N1,N2):
        E,coord1,coord2 = self.get_fields(field,plane,coord,k_num,mode_num,N1,N2)
        I = abs(E["x"])**2+abs(E["y"])**2+abs(E["z"])**2
        fig = plt.figure()
        plt.imshow(np.transpose(I))
        plt.colorbar()
        plt.title("2D plot of intensity in the %s plane" %(plane))

        
    def band_diagram(self,ymin, ymax,period=None):
        """Plots band diagram if simulation for a few wavevectors has already been run.
        
        Args:
            ymin (float): Min frequency to display in THz. Default is 250.
            ymax (float): Min frequency to display in THz. Default is 475.
                
        """
        if period is None and self.__class__ == Nanobeam:
           period = self.width
           
        if self.method == 'legume':
           
            fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6, 6)) # ax.plot([0, 1], [0, 3e5/period])
            ax.fill_betweenx([0, sc.c/period*1e-12], [0, 1], color=[.7, .7, .7])
            ax.plot(self.gme.kpoints[0, :]/2/np.pi, self.gme.freqs*sc.c/period*1e-12, 'b')
            ax.set_ylim([ymin, ymax])
            ax.set_xlim([self.gme.kpoints[0, 0]/2/np.pi, self.gme.kpoints[0, -1]/2/np.pi])
            ax.set_xlabel('$k_x (a/2\pi)$')
            ax.set_ylabel('Frequency (THz)')
            plt.show()
        
        if self.method == 'Lumerical':
            self.f = np.squeeze(self.structure.getresult("mode source","neff")["f"])
            self.lmbda = np.squeeze(self.structure.getresult("mode source","neff")["lambda"])
            self.neff = self.structure.getresult("mode source","neff")["neff"]
            omega = 2*np.pi*self.f
            grad = np.gradient(self.neff,omega,edge_order = 2)
            self.n_g = self.neff + omega*grad
            fig, ax = plt.subplots()
            plt.plot(self.f,self.n_g)
            plt.grid()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Group index")
            
    def denormalize(self,coord, k_num,mode_num,N1,N2):
        E, y, z = self.get_fields('e', "YZ",coord,k_num,mode_num,N1,N2)
        H, y, z = self.get_fields('h', "YZ",coord,k_num,mode_num,N1,N2)
        Ex = E['x']
        Ey = E['y']
        Ez = E['z']
        E = np.stack((Ex,Ey,Ez))
        Hx = H['x']
        Hy = H['y']
        Hz = H['z']
        H = np.stack((Hx,Hy,Hz))
        Cross = np.cross(E,np.conjugate(H),axis = 0)
        Poynt = 1/2*np.real(Cross[2,:,:])
        tot_int = simps([simps(zz_x,y) for zz_x in Poynt],z)
        return y,z,Poynt
        
        
        
    def _print(self, text, flush=False, end='\n'):
        if flush==False:
            print(text, end=end)
        else:
            sys.stdout.write("\r" + text)
            sys.stdout.flush()
            
            
          
class PeriodicStructure(Structure):
    def __init__(self,structure,period,PreCompute,method,gmax, k_min,k_max,N_k,numeig):
        self.period = period
        super().__init__(structure,PreCompute,method,gmax, k_min,k_max,N_k,numeig)
        
    def band_diagram(self, ymin = 250, ymax = 475):
        super().band_diagram(ymin, ymax,self.period)
        
    def find_guided_modes(self,N1,N2,maxy,threshold = 0.3):
        """Because of the periodic boundary conditions they impose, GME solvers can find spurious modes in the band gap that are not guided. This method looks at the field maps of the modes and looks if the most part of the intensity is actually around the center.
        
        Args:
            N1 (int): Sampling points in the first axis direction when getting the fields. Keeping it low makes the calculation faster. As it is a classification problem, it will be obvious if they are too low.
            N2 (int): Sampling points in the second axis direction.
            maxy (float): Span around the center where we want the fields to be localised (in units of the period)
            threshold (float): Ratio of the intensities in the center over the one on the whole surface for which we consider the filed is guided. Default is 0.3    
            
        Returns:
            array: Array of size (num bands, num wavevectors) with value NaN if not guided, the value of the frequency of the mode if guided.
        """
        guided_freq = np.zeros((np.shape(self.gme.freqs)[1],np.shape(self.gme.freqs)[0]))
        for m in range(np.shape(self.gme.freqs)[1]):
            self._print("Running over k_points for mode %s out of %s" %(m,np.shape(self.gme.freqs)[1]-1), flush=True)  
            for k in range(np.shape(self.gme.freqs)[0]-1,-1,-1):
#                print("k = ", k)
                E, x, y = self.gme.get_field_xy(field="e", kind=k, mind=m, z=self.thickness/2, Nx=N1, Ny=N2)
                I = abs(E["x"])**2+abs(E["y"])**2+abs(E["z"])**2
                yind_center = np.array([i for i in range(len(y)) if y[i] >=-maxy+self.y_edge and y[i]<=maxy+self.y_edge])
                y_center = y[yind_center[0]:yind_center[-1]+1]
                center_int = simps([simps(zz_x,x) for zz_x in I[yind_center[0]:yind_center[-1]+1,:]],y_center)
                tot_int = simps([simps(zz_x,x) for zz_x in I],y)
#                self._print("Running over k_points for mode %s" % (m)), flush=True)  
#                print("integral ratio between", -maxy, "and", maxy, "is ",  center_int/tot_int)
                if center_int >= threshold*tot_int:
                    guided_freq[m,k] = self.gme.freqs[k,m]
                elif (center_int < threshold*tot_int and k!=(np.shape(self.gme.freqs)[0]-1)) : 
                    if m!=0:
                        E, x, y = self.gme.get_field_xy(field="e", kind=k, mind=m-1, z=self.thickness/2, Nx=N1, Ny=N2)
                        I = abs(E["x"])**2+abs(E["y"])**2+abs(E["z"])**2
                        center_int_m1 = simps([simps(zz_x,x) for zz_x in I[yind_center[0]:yind_center[-1]+1,:]],y_center)
                        tot_int_m1 = simps([simps(zz_x,x) for zz_x in I],y)
#                        print("ratio at m-1 is " , center_int_m1/tot_int_m1)
                    else :
                        break
                    if center_int_m1 >= threshold*tot_int_m1 and guided_freq[m-1,k] == 0 :
                        guided_freq[m-1,k] = self.gme.freqs[k,m-1]
                    else:                        
                        break
                else:
                    break
        guided_freq[guided_freq == 0] = 'nan'
        fig=plt.figure()
        
        for m in range(len(guided_freq)):
                plt.plot(self.gme.kpoints[0]/2/np.pi,np.transpose(guided_freq[m,:])*sc.c/self.period*1e-12,'x-')
                plt.text(0.5,np.transpose(guided_freq[m,-1])*sc.c/self.period*1e-12,m)
#        ax = plt.gca()
#        ax.fill_betweenx([0, sc.c/self.period*1e-12], [0, 1], color=[.7, .7, .7])
        plt.ylabel("Frequency (THz)")
        plt.xlabel("k$_x$")
        plt.title("Band diagram with only the guided modes")
        plt.show(block = False)
#        mask = np.all(np.isnan(guided_freq), axis=1)
#        guided_freq = guided_freq[~mask]
        self.guided_freq = guided_freq
        return guided_freq
    
    
    def n_g(self,band_number = 13):
        ''' Function to plot group index for optimization of the design'''
        if not hasattr(self,"guided_freq"):
            self.find_guided_modes(20,40,1.5,threshold = 0.3)
            
        grad = np.gradient(self.guided_freq[band_number]*sc.c/self.period,self.gme.kpoints[0]/self.period)
        lambda_guided = [sc.c/x for x in self.guided_freq*sc.c/self.period]
        ng = [-sc.c/(2*np.pi*x) for x in grad]
        self.absng = [abs(x) for x in ng]
        fig, ax = plt.subplots()
#        plt.plot(self.gme.kpoints[0]/2/np.pi,self.absng)
        plt.plot(self.guided_freq[band_number]*sc.c/self.period,self.absng)
        plt.ylabel("n$_g$")
        plt.xlabel("$\lambda$")
        return self.absng
    
    
class HalfW1_waveguide(PeriodicStructure):
    """
    Class that implements a Half-W1 waveguide (see Zang, X. et al. (2016) Phys. Rev. Appl., 5(2), 024003).
    

    Attributes:
        material (material object): Material of the structure, from utils.materials
        PreCompute (bool): If True, the band diagram will be simulated upon definition of the structure.
        method (str): "legume" or "Lumerical". Defines the solver that will be used for simulating the fields (deprecated)
        gmax (int): For the GME solver, maximum reciprocal lattice wave-vector length in units of 2pi/a. Increasing it may increase dramatically the computation time. Usually gmax around 4 offers already good accuracy. Default is 4.
        k_min (float, [0,0.5]): Starting wavevector for computing the band diagram. 
        k_max (float, [0,0.5]): End wavevector for computing the band diagram.
        N_k (int): Number of sampled wavevector for the band diagram computation.
        numeig (int): Number of bands computed at each step. Default is 20.
        period (float): Periodicity of the cristal (in m)
        L (float): Width of the cristal with no hole, close to the edge (in m)
        radius (float): Radius of the holes (in units of the period)
        thickness (float): Slab thickness (in m)
        s1 (float): Shift of the first row of holes (for dispersion engineering)
        s2 (float): Shift of the second row of holes.
        
    Note:
        Inherits from PeriodicStructure class
    """
    def __init__(self,material,PreCompute = True, method = "legume", gmax = 4, period=230e-9,L=382e-9,radius=0.3,thickness = 150e-9, s1=25e-9,s2=15e-9,k_min = 0.2,k_max = 0.5,N_k = 5,numeig = 20):
        self.material = material
        if method == "legume":
            self.L = L/period
            self.radius = radius
            self.thickness = thickness/period
            
            N_layers = 10
            
            self.s1 = s1/period
            self.s2= s2/period
            
            L_air = 9
            L_bottom = np.sqrt(3)/2
            self.L_sim = (N_layers-1)*np.sqrt(3)/2 + L_air + self.L + L_bottom
            self.lattice = legume.Lattice([1,0], [0,self.L_sim]) #inside are the primitive vectors
            structure = legume.PhotCryst(self.lattice, eps_l=1, eps_u=1)
            structure.add_layer(d=self.thickness, eps_b=self.material.n**2)
            
            for i in range(N_layers):
                x = (i%2)/2
                if i == 0 :
                    y = self.L_sim/2-L_air-self.L-i*np.sqrt(3)/2+self.s1
                elif i == 1 :
                    y = self.L_sim/2-L_air-self.L-i*np.sqrt(3)/2+self.s2
                else: 
                    y = self.L_sim/2-L_air-self.L-i*np.sqrt(3)/2
                circle = legume.Circle(eps=1., x_cent=x, y_cent=y, r=radius)
                structure.add_shape(circle)
                            
            x_edges = [1/2, 1/2, -1/2, -1/2]
            y_edges = [self.L_sim/2-L_air, self.L_sim/2, self.L_sim/2, self.L_sim/2-L_air]
            rect = legume.Poly(eps=1, x_edges=x_edges, y_edges=y_edges)
            self.y_edge = self.L_sim/2-L_air
            structure.add_shape(rect) #PeriodicStructure.__init__(self,period)
        if method == "Lumerical":
            structure = lumapi.FDTD()
            self.thickness = thickness
            self.L = L
            n_rows = 22
            n_cols = 5
            even_flag = 0
            structure.addgroup(name="halfW1")
            for i in range(round(-n_rows/2),0):
                for j in range(round(-n_cols/2),round(n_cols/2)):
                    if (i!=0 and even_flag==0): 
                        if i == 1 or i == -1 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period+period/2 , y = (i)*(L-s1) , z = 0 , z_span = thickness , material = "etch")
                        elif i == 2 or i == -2 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period+period/2 , y = (i/2)*(L-s2)+i/2*period*np.sqrt(3)/2 , z = 0 , z_span = thickness , material = "etch")
                        else : 
                            structure.addcircle(radius = radius*period, x = (j-1)*period+period/2 , y = -L+(i+1)*period*np.sqrt(3)/2, z = 0 , z_span = thickness , material = "etch")
                    elif (i!=0 and even_flag==1):
                        if i == 1 or i == -1 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period , y = (i)*(L-s1) , z = 0 , z_span = thickness , material = "etch")
                        elif i == 2 or i == -2 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period , y = (i/2)*(L-s2)+i/2*period*np.sqrt(3)/2 , z = 0 , z_span = thickness , material = "etch")
                        else : 
                            structure.addcircle(radius = radius*period, x = (j-1)*period , y = -L+(i+1)*period*np.sqrt(3)/2 , z = 0 , z_span = thickness , material = "etch")
                    structure.addtogroup("halfW1")
                if even_flag==0:
                    even_flag=1
                else:
                    even_flag=0
            rectangle = structure.addrect(x = -period/2, y = -n_rows*period/4, z = 0.0, x_span = (n_cols+2)*period, y_span = n_rows*period/2, z_span = thickness, index = self.material.n)      
        super().__init__(structure,period,PreCompute,method,gmax, k_min,k_max,N_k,numeig)
            

class W1_waveguide(PeriodicStructure):
    """
    Class that implements a W1 waveguide (see Li, J. et al. (2008) Opt. express, 16(9), 6227).
    

    Attributes:
        material (material object): Material of the structure, from utils.materials.
        PreCompute (bool): If True, the band diagram will be simulated upon definition of the structure.
        method (str): "legume" or "Lumerical". Defines the solver that will be used for simulating the fields (deprecated)
        gmax (int): For the GME solver, maximum reciprocal lattice wave-vector length in units of 2pi/a. Increasing it may increase dramatically the computation time. Usually gmax around 4 offers already good accuracy. Default is 4
        k_min (float, [0,0.5]): Starting wavevector for computing the band diagram. 
        k_max (float, [0,0.5]): End wavevector for computing the band diagram.
        N_k (int): Number of sampled wavevector for the band diagram computation.
        numeig (int): Number of bands computed at each step. Default is 20.
        period (float): Periodicity of the cristal (in m)
        radius (float): Radius of the holes (in units of the period)
        thickness (float): Slab thickness (in m)
        s1 (float): Shift of the first row of holes (for dispersion engineering)
        s2 (float): Shift of the second row of holes.
        
    Note:
        Inherits from PeriodicStructure class
    """
    def __init__(self,material,PreCompute,method = "legume",gmax = 4, period=230e-9,radius=0.3,thickness = 150e-9,s1=25e-9,s2=15e-9,k_min = 0.2,k_max = 0.5,N_k = 5,numeig = 20):
        self.material = material
        if method == "legume":
            self.radius = radius
            self.thickness = thickness/period
            
            N_layers = 10

            self.s1 = s1/period
            self.s2= s2/period

            L_extra = np.sqrt(3)/2
            self.L_sim = 2*(N_layers-1)*np.sqrt(3)/2 + 2*L_extra
            self.lattice = legume.Lattice([1,0], [0,self.L_sim]) #inside are the primitive vectors
            structure = legume.PhotCryst(self.lattice, eps_l=1, eps_u=1)
            structure.add_layer(d=self.thickness, eps_b=self.material.n**2)
            self.y_edge = 0
            
            for i in range(N_layers):
                x = (i%2)/2
                if i == 0 :
                    y = -(i+1)*np.sqrt(3)/2+s1
                elif i == 1 :
                    y = -(i+1)*np.sqrt(3)/2+s2
                else: 
                    y = -(i+1)*np.sqrt(3)/2
                circle1 = legume.Circle(eps=1., x_cent=x, y_cent=y, r=radius)
                circle2 = legume.Circle(eps=1., x_cent=x, y_cent=-y, r=radius)
                structure.add_shape(circle1)
                structure.add_shape(circle2)
                
        if method == "Lumerical":
            structure = lumapi.FDTD()
            self.thickness = thickness
            self.L = 0
            n_rows = 22
            n_cols = 5
            even_flag = 0
            structure.addgroup(name="W1")
            for i in range(round(-n_rows/2),round(n_rows/2)):
                for j in range(round(-n_cols/2),round(n_cols/2)):
                    if (i!=0 and even_flag==0): 
                        if i == 1 or i == -1 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period+period/2 , y = (i)*period*np.sqrt(3)/2-i*s1 , z = 0 , z_span = thickness , material = "etch")
                        elif i == 2 or i == -2 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period+period/2 , y = (i)*period*np.sqrt(3)/2-i/2*s2 , z = 0 , z_span = thickness , material = "etch")
                        else : 
                            structure.addcircle(radius = radius*period, x = (j-1)*period+period/2 , y = (i)*period*np.sqrt(3)/2, z = 0 , z_span = thickness , material = "etch")
                    elif (i!=0 and even_flag==1):
                        if i == 1 or i == -1 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period , y = (i)*period*np.sqrt(3)/2-i*s1 , z = 0 , z_span = thickness , material = "etch")
                        elif i == 2 or i == -2 :
                            structure.addcircle(radius = radius*period, x = (j-1)*period , y = (i)*period*np.sqrt(3)/2-i/2*s2 , z = 0 , z_span = thickness , material = "etch")
                        else : 
                            structure.addcircle(radius = radius*period, x = (j-1)*period , y = (i)*period*np.sqrt(3)/2, z = 0 , z_span = thickness , material = "etch")
                    structure.addtogroup("W1")
                if even_flag==0:
                    even_flag=1
                else:
                    even_flag=0
            rectangle = structure.addrect(x = -period/2, y =0, z = 0.0, x_span = (n_cols+2)*period, y_span = n_rows*period, z_span = thickness, index = self.material.n)      

        super().__init__(structure,period,PreCompute,method,gmax,k_min,k_max,N_k,numeig)

        
class Nanobeam(Structure):
    """
    Class that implements a Nanobeam waveguide 

    Attributes:
        material (material object): Material of the structure, from utils.materials
        PreCompute (bool): If True, the band diagram will be simulated upon definition of the structure.
        method (str): "legume" or "Lumerical". Defines the solver that will be used for simulating the fields (deprecated)
        gmax (int): For the GME solver, maximum reciprocal lattice wave-vector length in units of 2pi/a. Increasing it may increase dramatically the computation time. Usually gmax around 4 offers already good accuracy. Default is 4
        k_min (float, [0,0.5]): Starting wavevector for computing the band diagram. 
        k_max (float, [0,0.5]): End wavevector for computing the band diagram.
        N_k (int): Number of sampled wavevector for the band diagram computation
        numeig (int): Number of bands computed at each step. Default is 20.
        width (float): Width of the nanobeam (in m)
        thickness (float): Slab thickness (in m)
        L_air (float): Size of air region around the waveguide (in units of the width)
        
    Note:
        Inherits from Structure class
    """
    def __init__(self,material,PreCompute,method = "legume",gmax = 4,width = 150e-9, thickness = 150e-9, L_air = 400e-9,k_min = 0.2,k_max = 0.5,N_k = 5,numeig = 20):
        self.material = material
        if method == "legume":
            self.width = width
            self.thickness = thickness/self.width
            L_air = L_air/self.width
            self.L_sim = 1 + 2*L_air
            
            lattice = legume.Lattice([1,0], [0,self.L_sim]) #inside are the primitive vectors
            structure = legume.PhotCryst(lattice, eps_l=1, eps_u=1)
            structure.add_layer(d=self.thickness, eps_b=material.n**2)
            
            x_edges = [1/2, 1/2, -1/2, -1/2]
            y_edges = [self.L_sim/2-L_air, self.L_sim/2, self.L_sim/2, self.L_sim/2-L_air]
            rect = legume.Poly(eps=1, x_edges=x_edges, y_edges=y_edges)
            structure.add_shape(rect)
            
            x_edges = [1/2, 1/2, -1/2, -1/2]
            y_edges = [-self.L_sim/2, -self.L_sim/2+L_air, -self.L_sim/2+L_air, -self.L_sim/2]
            rect = legume.Poly(eps=1, x_edges=x_edges, y_edges=y_edges)
            structure.add_shape(rect)
        if method == "Lumerical":
            structure = lumapi.FDTD()
            self.thickness = thickness
            self.L = 0
            rectangle = structure.addrect(x = 0, y = 0, z = 0, x_span = 2e-6, y_span = width, z_span = thickness, index = self.material.n)

        super().__init__(structure,PreCompute,method,k_min,k_max,N_k,numeig)
        

class Nanofiber(Structure):
    def __init__(self,material,cladding,PreCompute,method = "Theory",gmax = 4, radius = 200e-9, k_min = 0.2,k_max = 0.5,N_k = 5,numeig = 20):
        self.material = material
        self.cladding = cladding
        self.radius = radius
        self.width = 2*radius
        self.thickness = 2*radius
        self.L = 0
        
        structure = 0
        super().__init__(structure,PreCompute,method,gmax,k_min,k_max,N_k,numeig)
    
    def compute_beta_joos(self,lmbda,method='lm',start=1.4,maxrec=10,tolerance=1e-4,**kwargs):
        a = self.radius
        n1 = self.material.n
        n2 = self.cladding.n

        k00 = 2*np.pi/lmbda
        
        h = lambda beta: np.sqrt((k00**2)*(n1**2)-beta**2)
        q = lambda beta: np.sqrt(beta**2-(k00**2)*(n2**2))
        
        
        func = lambda beta: (jp(1,h(beta)*a)/(h(beta)*a*j1(h(beta)*a)) + kp(1,q(beta)*a)/(q(beta)*a*k1(q(beta)*a))) * (n1**2*jp(1,h(beta)*a)/(h(beta)*a*j1(h(beta)*a)) + n2**2*kp(1,q(beta)*a)/(q(beta)*a*k1(q(beta)*a))) - (beta**2/k00**2)*(1/(h(beta)*a)**2 + 1/(q(beta)*a)**2)**2

        for i in range(maxrec):
            beta_full_solution = root(func, (1+(start-1)/(2**i))*k00,method=method,**kwargs)
            if abs(beta_full_solution.fun[0]) < tolerance :
                self.beta = beta_full_solution.x[0]
                return self.beta/k00,func,beta_full_solution
            else:
                continue
        self.beta = k00
        return self.beta/k00,func,beta_full_solution

    def electric_field_circular(self,x,y,z,lmbda,Ptot,sign):
        a = self.radius
        n1 = self.material.n
        n2 = self.cladding.n

        k00 = 2*np.pi/lmbda
        omega = pu.cc * k00
        beta,_,_ = self.compute_beta_joos(lmbda)
        beta = beta*k00

        ## Coordinates
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        ### Function definition
        h = np.sqrt((k00**2)*(n1**2)-beta**2)
        q = np.sqrt(beta**2-(k00**2)*(n2**2))
        s = (1/(h*a)**2+1/(q*a)**2)/(jp(1,h*a)/(h*a*j1(h*a))+kp(1,q*a)/(q*a*k1(q*a)))
        x = beta**2 /(2*h*pu.μ0*omega)
        y = beta**2 /(2*q*pu.μ0*omega)


        ### Normalization
        Tin = (1+s)*(1+(beta**2/h**2)*(1+s))*(jv(2,h*a)**2-j1(h*a)*jv(3,h*a)) + (1-s)*(1+(beta**2/h**2)*(1-s))*(j0(h*a)**2 + j1(h*a)**2)
        Tout = (1+s)*(1-(beta**2/q**2)*(1+s))*(kn(2,q*a)**2-k1(q*a)*kn(3,q*a)) + (1-s)*(1-(beta**2/q**2)*(1-s))*(k0(q*a)**2 - k1(q*a)**2)
        C = np.sqrt(4*omega*pu.μ0*Ptot/(np.pi*beta*a**2))*(Tout+ Tin*(k1(q*a)/j1(h*a))**2)**(-1/2)

        ### Fields inside the core
        if 0< r < a:
            Ez = C*j1(h*r)*(k1(q*a)/j1(h*a))*np.exp(sign*1j*theta)*np.exp(-1j*beta*z)
            Er = C*(k1(q*a)/j1(h*a))*(1j*beta/(2*h))*(jv(2,h*r)*(1+s)-j0(h*r)*(1-s))*np.exp(sign*1j*theta)*np.exp(-1j*beta*z)
            Etheta = sign*C*(k1(q*a)/j1(h*a))*(beta/(2*h))*(jv(2,h*r)*(1+s)+j0(h*r)*(1-s))*np.exp(sign*1j*theta)*np.exp(-1j*beta*z)

        ### Fields outside the core
        elif r >= a:
            Ez = C*k1(q*r)*np.exp(sign*1j*theta)*np.exp(-1j*beta*z)
            Er = -C*(1j*beta/(2*q))*(kn(2,q*r)*(1+s)+k0(q*r)*(1-s))*np.exp(sign*1j*theta)*np.exp(-1j*beta*z)
            Etheta = sign*C*(beta/(2*q))*(-kn(2,q*r)*(1+s)+k0(q*r)*(1-s))*np.exp(sign*1j*theta)*np.exp(-1j*beta*z)
        
        return Ez, Er, Etheta
    
    def convert_to_cartesian(self,Ez,Er,Etheta):
        return 

    def electric_field_linear(self,x,y,z,lmbda,Ptot,axis):
        a = self.radius
        n1 = self.material.n
        n2 = self.cladding.n

        k00 = 2*np.pi/lmbda
        omega = pu.cc * k00
        if hasattr(self,'beta'):
            beta = self.beta
        else:
            beta,_,_ = self.compute_beta_joos(lmbda)*k00

        ## Coordinates
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        ### Function definition
        h = np.sqrt((k00**2)*(n1**2)-beta**2)
        q = np.sqrt(beta**2-(k00**2)*(n2**2))
        s = (1/(h*a)**2+1/(q*a)**2)/(jp(1,h*a)/(h*a*j1(h*a))+kp(1,q*a)/(q*a*k1(q*a)))
        x = beta**2 /(2*h*pu.μ0*omega)
        y = beta**2 /(2*q*pu.μ0*omega)


        ### Normalization
        Tin = (1+s)*(1+(beta**2/h**2)*(1+s))*(jv(2,h*a)**2-j1(h*a)*jv(3,h*a)) + (1-s)*(1+(beta**2/h**2)*(1-s))*(j0(h*a)**2 + j1(h*a)**2)
        Tout = (1+s)*(1-(beta**2/q**2)*(1+s))*(kn(2,q*a)**2-k1(q*a)*kn(3,q*a)) + (1-s)*(1-(beta**2/q**2)*(1-s))*(k0(q*a)**2 - k1(q*a)**2)
        C = np.sqrt(4*omega*pu.μ0*Ptot/(np.pi*beta*a**2))*(Tout+ Tin*(k1(q*a)/j1(h*a))**2)**(-1/2)

        ### Fields inside the core
        if 0 <= r < a:
            Ex = C*(1j*beta/(h*np.sqrt(2)))*(k1(q*a)/j1(h*a))*(jv(2,h*r)*(1+s)*np.cos(2*theta-axis)-j0(h*r)*(1-s)*np.cos(axis))*np.exp(-1j*beta*z)
            Ey = C*(1j*beta/(h*np.sqrt(2)))*(k1(q*a)/j1(h*a))*(jv(2,h*r)*(1+s)*np.sin(2*theta-axis)-j0(h*r)*(1-s)*np.sin(axis))*np.exp(-1j*beta*z)
            Ez = np.sqrt(2)*C*j1(h*r)*(k1(q*a)/j1(h*a))*np.cos(theta-axis)*np.exp(-1j*beta*z)


        ### Fields outside the core
        elif r >= a:
            Ex = -C*(1j*beta/(q*np.sqrt(2)))*(kn(2,q*r)*(1+s)*np.cos(2*theta-axis)+k0(q*r)*(1-s)*np.cos(axis))*np.exp(-1j*beta*z)
            Ey = -C*(1j*beta/(q*np.sqrt(2)))*(kn(2,q*r)*(1+s)*np.sin(2*theta-axis)+k0(q*r)*(1-s)*np.sin(axis))*np.exp(-1j*beta*z)
            Ez = np.sqrt(2)*C*k1(q*r)*np.cos(theta-axis)*np.exp(-1j*beta*z)
        
        return Ex,Ey,Ez