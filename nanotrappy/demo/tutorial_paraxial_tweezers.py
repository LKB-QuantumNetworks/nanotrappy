# -*- coding: utf-8 -*-
# import nanotrappy as nt
# from nanotrappy.utils.physicalunits import *
import os
import numpy as np
import matplotlib.pyplot as plt
from beam_prop import DebyeWolfPropagator, LG_Beam
from datetime import datetime

#Create a folder for storing the simulated fields
dir_path = os.path.dirname(os.path.realpath(__file__))
current_date = datetime.now().strftime("%y%m%d")
subfolder = r"\testfolder_paraxial_"+ current_date

#%% Paraxial case

#Define LG parameters
p, l = 0, 0
lmbda = 1064e-9
w0 = 1.5e-6
LG = LG_Beam(p,l,lmbda,w0, power = 1)
    
points = 301
x_extrema = 3
z_extrema = 10

# y = np.linspace(-x_extrema*lmbda, x_extrema*lmbda, points)
# x = np.linspace(-x_extrema*lmbda, x_extrema*lmbda, points)
x = [0]
y = [0]
z = np.linspace(-z_extrema*lmbda, z_extrema*lmbda, points)
xm, ym, zm = np.meshgrid(x,y,z, indexing ='ij')

Ex_red = LG.field(xm,ym,zm)
Ey_red = np.zeros((len(x),len(y),len(z)))
Ez_red = np.zeros((len(x),len(y),len(z)))
E_red = np.stack((Ex_red,Ey_red,Ez_red)).astype(object)
I_red = np.sum(np.abs(E_red)**2,0)

arr = [lmbda, x, y, z, E_red]
save_arr = np.empty(len(arr), object)
save_arr[:] = arr  
os.mkdir(dir_path + subfolder)
np.save(dir_path + subfolder + "\\LG" + str(p) + "0_" + str(lmbda*1e9) + ".npy", save_arr)
        
#%% Trap simulation as in the other files
datafolder = os.path.dirname(os.path.abspath(__file__)) + subfolder
P_range = np.linspace(0,4e-3,51)

#Single gaussian tweezer
beam = nt.Beam(1064e-9,"f",3*mW, 0)

#LG sum tweezer, with the same intensities
beam_sum = nt.BeamSum([1064e-9,1064e-9,1064e-9], [1*mW,1*mW,1*mW],indices = [0, 1, 2])

trap = nt.Trap_beams(beam) #choose here between the two different strategies
syst = nt.atomicsystem(nt.Rubidium87(), "5S1/2", f2)
Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder)

#%%
Simul.geometry = nt.AxisX(coordinates=(0, 0)) #cannot look along z with this technique
Simul.compute()
Simul.save()
#%%
viz = nt.Viz(Simul, trapping_axis="X")
fig, ax, slider_ax = viz.plot_trap(mf=[0], Pranges=[30], increments=[0.01])


