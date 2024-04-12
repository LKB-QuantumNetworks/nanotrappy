'''
Script with functions for simulating Laguerre-Gaussian beams and their propagation,
useful to compute tweezer traps'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from numpy import fft
import scipy.constants as sc
from scipy.special import eval_genlaguerre, genlaguerre

def w_z(z, w0, zR):
    return w0*np.sqrt(1+(z/zR)**2)
    
def n_theta(theta, phi):
    return np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), - np.sin(theta)])

def n_phi(phi):
    return np.array([-np.sin(phi), np.cos(phi), phi*0 ])

def n_rho(phi):
    return np.array([np.cos(phi), np.sin(phi), phi*0]) 

'''Class that defines a Laguerre-Gaussian beam and stores its parameters
Has multiple methods to get the beam field in any plane and phase'''
class LG_Beam():
    def __init__(self, p, l, lmbda, w0, power = 1, center = [0,0]):
        self.lmbda = lmbda
        self.w0 = w0
        self.zR = np.pi*self.w0**2/self.lmbda
        self.k = 2*np.pi/self.lmbda
        self.z0 = 0
        self.p = p
        self.l = l
        self.x0 = center[0]
        self.y0 = center[1]
        self.power = power #in Watts

    def waist(self, z):
        return self.w0*np.sqrt(1+(z/self.zR)**2)
    
    def Gouy(self,z):
        return np.exp(1J*(2*self.p+abs(self.l)+1)*np.arctan(z/self.zR))
    
    def field(self,x,y,z):
        rho_sq = (y-self.y0)**2+(x-self.x0)**2
        return np.sqrt(2*self.power/(self.w0**2*sc.c*sc.epsilon_0))*np.sqrt(2*np.math.factorial(self.p)/(np.pi*np.math.factorial(self.p+abs(self.l))))*(self.w0/self.waist(z))*(np.sqrt(rho_sq*2)/self.waist(z))**abs(self.l)*np.exp(-rho_sq/(self.waist(z)**2))*np.exp(-1J*self.k*rho_sq*z/(2*(z**2+self.zR**2)))*eval_genlaguerre(self.p,self.l,2*rho_sq/self.waist(z)**2)*np.exp(-1J*self.l*np.arctan2(y-self.y0,x-self.x0))*self.Gouy(z)
        #See Wikipedia page for the field expression
        
    def field_SLM(self, x,y,z):    
        rho_sq = (y-self.y0)**2+(x-self.x0)**2
        return np.sqrt(self.power/(self.w0**2))*np.sqrt(2*np.math.factorial(self.p)/(np.pi*np.math.factorial(self.p+abs(self.l))))*(self.w0/self.waist(z))*(np.sqrt(rho_sq*2)/self.waist(z))**abs(self.l)*np.exp(-1J*self.k*rho_sq*z/(2*(z**2+self.zR**2)))*eval_genlaguerre(self.p,self.l,2*rho_sq/self.waist(z)**2)*np.exp(-1J*self.l*np.arctan2(y-self.y0,x-self.x0))*self.Gouy(z)
    
    def field_norm(self,x,y,z): #normalized field, Imax = 1
        rho_sq = (y-self.y0)**2+(x-self.x0)**2
        return (self.w0/self.waist(z))*(np.sqrt(rho_sq*2)/self.waist(z))**abs(self.l)*np.exp(-rho_sq/(self.waist(z)**2))*np.exp(-1J*self.k*rho_sq*z/(2*(z**2+self.zR**2)))*eval_genlaguerre(self.p,self.l,2*rho_sq/self.waist(z)**2)*np.exp(-1J*self.l*np.arctan2(y-self.y0,x-self.x0))*self.Gouy(z)
    
    def intensity_plane(self, plane, coord1range, coord2range, out_of_plane_coord, samples):
        fig, ax= plt.subplots()
        coord1 = np.linspace(coord1range[0],coord1range[1],samples)
        coord2 = np.linspace(coord2range[0],coord2range[1],samples)
        I = np.zeros((len(coord1),len(coord2)))
        if plane == "XY":
            for k in range(len(coord1)):
                for i in range(len(coord2)):
                    I[k,i] = np.abs(self.field(coord1[k],coord2[i],out_of_plane_coord))**2
        
        plt.imshow(I, extent = [coord1[0],coord1[-1],coord2[0],coord2[-1]])
        plt.colorbar() 
        
    def amplitude_plane(self, plane, coord1, coord2, out_of_plane_coord):
        self.amplitude = np.zeros((len(coord1),len(coord2)),dtype = "complex128")
        if plane == "XY":
            coord1v, coord2v = np.meshgrid(coord1, coord2)
            self.amplitude = self.field(coord1v,coord2v,out_of_plane_coord)
        if plane == "YZ":
            for k in range(len(coord1)):
                for i in range(len(coord2)):
                    self.amplitude[k,i] = self.field(out_of_plane_coord,coord1[k],coord2[i])
        return self.amplitude
        
    def phase_plane(self, plane, coord1range, coord2range, out_of_plane_coord, samples):
        fig, ax= plt.subplots()
        coord1 = np.linspace(coord1range[0],coord1range[1],samples)
        coord2 = np.linspace(coord2range[0],coord2range[1],samples)
        phi = np.zeros((len(coord1),len(coord2)))
        if plane == "XY":
            for k in range(len(coord1)):
                for i in range(len(coord2)):
                    phi[k,i] = np.angle(self.field(coord1[k],coord2[i],out_of_plane_coord))
        
        plt.imshow(phi, extent = [coord1[0],coord1[-1],coord2[0],coord2[-1]])
     
'''Class that takes a field amplitude in input and propagates it to the focus of a given 
focusing object with focal distance f and numerical aperture NA. The calculation is done 
in the Debye-Wolf formalism, considering paraxial approximation breaks down.'''
class DebyeWolfPropagator:
    def __init__(self,x,y,z,E,lmbda, zout = 0):
        self.E_in = E
        self.x_in = x
        self.y_in = y
        self.z_in = z 
        self.z_out = zout
        
        self.lmbda = lmbda
        self.k = 2*np.pi/self.lmbda
        
        self.rho_max  = min(np.max(np.sqrt(self.x_in**2)),np.max(np.sqrt(self.y_in**2)))
        self.E_inc_xy = RegularGridInterpolator((self.x_in, self.y_in), E[0,:,:,0], method = "linear", fill_value = 0)

    def set_focusing_object(self, f, NA):
        self.f = f 
        self.NA = NA
        self.theta_max = np.arcsin(self.rho_max/self.f)
        theta_max_NA = np.arcsin(self.NA)
        if self.theta_max >= theta_max_NA:
            self.theta_max = theta_max_NA
            
    def E_inc(self, theta, phi):
        # Einc doesn't need to be axisymmetric anymore (contrary to the resolution chosen in DebyeWolfI0I1 and DebyeWolfArbitrary)
        rho = self.f*np.sin(theta)
        # Rho, Phi = np.meshgrid(rho, phi)
        Ei = self.E_inc_xy((rho*np.cos(phi), rho*np.sin(phi)))
        return Ei
        
    def E_infini(self,theta, phi):
        return np.tile(self.E_inc(theta, phi)*np.sqrt(np.cos(theta)),(3,1,1))*[np.cos(phi)*n_theta(theta, phi) - np.sin(phi)*n_phi(phi)]
    
    def set_integration_params(self, M):
        self.M = M #Number of sampling points for k in the pupil aperture
        self.Delta_K = self.k*self.NA/self.M
        kxlist = np.linspace(-self.M*self.Delta_K, self.M*self.Delta_K, 2*self.M+1)
        kylist = np.linspace(-self.M*self.Delta_K, self.M*self.Delta_K, 2*self.M+1)
        KX, KY = np.meshgrid(kxlist, kylist)
        self.theta_array = np.nan_to_num(np.arcsin(self.Delta_K/self.k*np.sqrt((KX/self.Delta_K)**2 + (KY/self.Delta_K)**2)), nan = np.pi/2)
        self.condition = (self.theta_array > self.theta_max)
        #Create masked arrays to not go above the interpolation range for E
        ta = np.ma.masked_where(self.condition, self.theta_array)
        self.phi_array = np.arctan2(KY,KX)
        pa = np.ma.masked_array(self.phi_array, ta.mask)
        self.ta = ta.filled(fill_value = 0)
        self.pa = pa.filled(fill_value = 0) #Converts them into arrays with 0 in the masked regions
        self.kz_array = self.k*np.cos(self.theta_array)
        
    def propagate(self, n_pad):
        E_infini_array = self.E_infini(self.ta, self.pa) 
        Einfx = np.ma.masked_where(self.condition, E_infini_array[0,0,:,:])
        Einfy = np.ma.masked_where(self.condition, E_infini_array[0,1,:,:])
        Einfz = np.ma.masked_where(self.condition, E_infini_array[0,2,:,:])

        Einfx, Einfy, Einfz = Einfx.filled(fill_value = 0), Einfy.filled(0),Einfz.filled(0)
        Einf_tot = np.stack((Einfx, Einfy, Einfz))

        integrandfft_array = np.exp(1J*self.kz_array*self.z_out)*Einf_tot/self.kz_array

        integrandfft_array_pad = np.stack((np.pad(integrandfft_array[0],n_pad),np.pad(integrandfft_array[1],n_pad),np.pad(integrandfft_array[2],n_pad)))

        pad_dim = integrandfft_array_pad.shape[1]
        kxlist_pad = np.linspace(-(pad_dim-1)/2*self.Delta_K, (pad_dim-1)/2*self.Delta_K, pad_dim)
        kylist_pad = np.linspace(-(pad_dim-1)/2*self.Delta_K, (pad_dim-1)/2*self.Delta_K, pad_dim)

        self.xf = np.fft.fftshift(np.fft.fftfreq(len(kxlist_pad),self.Delta_K))*(2*np.pi)
        self.yf = np.fft.fftshift(np.fft.fftfreq(len(kylist_pad),self.Delta_K))*(2*np.pi)

        self.fftarray = fft.fftshift(fft.fft2(fft.ifftshift(integrandfft_array_pad)))*self.Delta_K**2
        self.fftarray = self.fftarray/5**4
        #Empirical normalization to recover the fields in the paraxial case: doesn't seem to depend on any simulation parameter
        return self.fftarray
    
    def Itot(self):
        self.Itot = np.abs(self.fftarray[0])**2+np.abs(self.fftarray[1])**2+np.abs(self.fftarray[2])**2
        return self.Itot
        
    def plot_intensities(self, f0, axmin = -3, axmax = 3):
        
        self.Itot = np.abs(self.fftarray[0])**2+np.abs(self.fftarray[1])**2+np.abs(self.fftarray[2])**2

        self.fig, ax = plt.subplots(2,2)
        fx = ax[0,0].pcolormesh(self.xf/self.lmbda, self.yf/self.lmbda, np.abs(self.fftarray[0,:,:])**2, shading = "auto")
        plt.colorbar(fx, ax=ax[0,0])
        fy = ax[0,1].pcolormesh(self.xf/self.lmbda, self.yf/self.lmbda, np.abs(self.fftarray[1,:,:])**2, shading = "auto")
        plt.colorbar(fy, ax=ax[0,1])
        fz = ax[1,0].pcolormesh(self.xf/self.lmbda, self.yf/self.lmbda, np.abs(self.fftarray[2,:,:])**2, shading = "auto")
        plt.colorbar(fz, ax=ax[1,0])
        I = ax[1,1].pcolormesh(self.xf/self.lmbda, self.yf/self.lmbda, self.Itot, shading = "auto") #, cmap="seismic_r")
        plt.colorbar(I, ax=ax[1,1])

        custom_xlim = (axmin, axmax)
        custom_ylim = (axmin, axmax)
        lims = plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        plt.suptitle("Focused beam with NA = %s, filling fraction f0 = %s, \n at a distance z = %s mm from focus" %(self.NA, f0, np.round(self.z_out*1e3,3)))
    

if __name__ == "__main__" :
    lmbda = 1064e-9
    k = 2*np.pi/lmbda
    
    #Define parameters for our setup
    NA = 0.7 #Numerical aperture
    f = 2e-3 #focal length, 
    f0 = 1 #Filling factor= waist of the input beam/aperture radius of the focusing object
    
    w0 = f0*f*NA #input waist
    zR_1 = (np.pi*w0**2)/lmbda #input zR
    w0_2 = f*lmbda/(np.pi*w0) #output waist
    zR_2 = (np.pi*w0_2**2)/lmbda #output zR
    
    #Can play on it if results not satisfying/resolution too low
    points = 601
    x_extrema = 8000 
    # n_pad = 2000
    n_pad = 1000
    
    #Create beam to be focused
    LG20 = LG_Beam(0,0,lmbda,w0, power = 1)
    LGs = [LG20]
    yi = np.linspace(-x_extrema*lmbda, x_extrema*lmbda, points)
    xi = np.linspace(-x_extrema*lmbda, x_extrema*lmbda, points)
    zi = [0] # z position on the input plane (0 means we have a perfect collimated beam)
    z = zi[0] 
    #Set position we want to look at at the output plane
    # z_out = 5*zR_2
    z_out = 0
    xm, ym, zm = np.meshgrid(xi,yi,zi, indexing ='ij')
    
    #Create input beam polarized along x
    Ex_red = LGs[0].field(xm,ym,zm)
    Ey_red = np.zeros((len(xi),len(yi),len(zi)))
    Ez_red = np.zeros((len(xi),len(yi),len(zi)))
    E_red = np.stack((Ex_red,Ey_red,Ez_red))
    
    #%% Propagate beam to the focus
    DBP = DebyeWolfPropagator(xi, yi, zi, E_red, lmbda, z_out)
    DBP.set_focusing_object(f, NA) #set characteristics of focusing object
    DBP.set_integration_params(M = 10) #M is the number of samples to compute the FFT. If the beam is big (for high z), this should be increased
    #This value determines the extension of the xf,yf plane
    fftarray = DBP.propagate(n_pad)
    # DBP.plot_intensities(f0, -7,7)
    # print(DBP.xf[-1])
    
    fig, ax = plt.subplots()
    I = ax.pcolormesh(DBP.xf, DBP.yf, DBP.Itot(), cmap = "viridis")
    cbar = plt.colorbar(I, ax=ax)
    cbar.set_label("Intensity [a.u.]", labelpad = 2)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    # DBP.plot_intensities(f0, -2*w_z(z_out, w0_2, zR_2)/lmbda,2*w_z(z_out, w0_2, zR_2)/lmbda) 
    #plots the beam with a range that depends on z, the values are all in units of lambda
    
    #%% Line plot
    fig, ax = plt.subplots()
    plt.plot(DBP.xf/DBP.lmbda, DBP.Itot[:,DBP.Itot.shape[1]//2]/np.max(DBP.Itot), label = "along x")
    plt.plot(DBP.yf/DBP.lmbda, DBP.Itot[DBP.Itot.shape[0]//2,:]/np.max(DBP.Itot), label = "along y")
    ax.legend()