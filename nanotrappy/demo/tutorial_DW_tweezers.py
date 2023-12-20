# -*- coding: utf-8 -*-
import nanotrappy as nt
from nanotrappy.utils.physicalunits import *
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(dir_path)
sys.path.insert(0, parentdir) 
import SLM_scripts as slms
import os

subfolder = r"\testfolder_FFT_2Dxy"
dir_path = os.path.dirname(os.path.realpath(__file__))

f0 = 0.4
lmbda = 1064e-9
NA = 0.4
f = 10e-3 
k = 2*np.pi/lmbda
#%% Define focusing parameters
w0 = f0*f*NA #input waist
zR_1 = (np.pi*w0**2)/lmbda #input zR
w0_2 = f*lmbda/(np.pi*w0) #output waist
zR_2 = (np.pi*w0_2**2)/lmbda #output zR

#Can play on it if results not satisfying/resolution too low
points = 401
x_extrema = 8000 
# n_pad = 1400
n_pad = 800

#Define the region over which we want to simulate the beams. I recommend to do either 1D or 2D maps and not 3D
#Better to make 2 files, 1 for 2D and 1 for 1D
yi = np.linspace(-x_extrema*lmbda, x_extrema*lmbda, points)
xi = np.linspace(-x_extrema*lmbda, x_extrema*lmbda, points)
zi = [0] # z position on the input plane (0 means we have a perfect collimated beam)
z = zi[0] 
#Set position we want to look at at the output plane
z_out = 0
xm, ym, zm = np.meshgrid(xi,yi,zi, indexing ='ij')

plist = [0,2,4]
for i in range(len(plist)):
    p = plist[i]
    LG = slms.LG_Beam(p,0,lmbda,w0, power = 1)
    LGs = [LG]
    #Create input beam polarized along x
    Ex_red = LGs[0].field(xm,ym,zm)
    Ey_red = np.zeros((len(xi),len(yi),len(zi)))
    Ez_red = np.zeros((len(xi),len(yi),len(zi)))
    E_red = np.stack((Ex_red,Ey_red,Ez_red))

    # Propagate beam to the focus
    DBP = slms.DebyeWolfPropagator(xi, yi, zi, E_red, lmbda, z_out)
    DBP.set_focusing_object(f, NA) #set characteristics of focusing object
    DBP.set_integration_params(M = 25) #M is the number of samples to compute the FFT. If the beam is big (for high z), this should be increased
    fftarray = DBP.propagate(n_pad)
    fftarray = fftarray.astype(np.complex64)


    arr = [lmbda, DBP.xf, DBP.yf, [z_out], np.expand_dims(fftarray,-1)]
    save_arr = np.empty(len(arr), object)
    save_arr[:] = arr  
    try:
        np.save(dir_path + subfolder + "\\LG" + str(p) + "0_1064_f0=" + str(np.round(f0,2)).ljust(5, '0') + ".npy", save_arr)
    except FileNotFoundError:
        os.mkdir(dir_path + subfolder)
        np.save(dir_path+ subfolder + "\\LG" + str(p) + "0_1064_f0=" + str(np.round(f0,2)).ljust(5, '0') + ".npy", save_arr)

#%% Then analyze the fields to compute the tweezers
datafolder = os.path.dirname(os.path.abspath(__file__)) + subfolder
P_range = np.linspace(0,4e-3,51)

#Single gaussian tweezer
beam = nt.Beam(1064e-9,"f",3*mW, 0)

#LG sum tweezer, with the same intensities
beam_sum = nt.BeamSum([1064e-9,1064e-9,1064e-9], [1*mW,1*mW,1*mW],indices = [0, 1, 2])

trap = nt.Trap_beams(beam)
syst = nt.atomicsystem(nt.Rubidium87(), "5S1/2", f2)
Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder)

#%%
Simul.geometry = nt.AxisX(coordinates=(0, 0))
Simul.compute()
Simul.save()
#%%
viz = nt.Viz(Simul, trapping_axis="X")
fig, ax, slider_ax = viz.plot_trap(mf=[0], Pranges=[30], increments=[0.01])


