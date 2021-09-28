import nanotrappy as nt
from nanotrappy.utils.physicalunits import *
import numpy as np
import time, os
import matplotlib.pyplot as plt

"""
The electric field around a Nanofiber can be computed in situ by the package 
and saved with the right formatting
"""
nanof = nt.Nanofiber(nt.SiO2(), nt.air(), radius=250e-9)
wavelength_red = 937e-9
wavelength_blue = 685.5e-9
wavelength_blue2 = 685.6e-9
y = np.linspace(0, 800e-9, 201)
x = np.linspace(0, 800e-9, 201)
z = np.array([0])
P = 1 #in Watts
theta = 0 #radians
E_red = nanof.compute_E_linear(x,y,z,wavelength_red,P,theta)
E_blue_fwd = nanof.compute_E_linear(x,y,z,wavelength_blue,P,theta)
E_blue_bwd = nanof.compute_E_linear(x,y,z,wavelength_blue2,P,theta)
np.save("./testfolder/modeblue685.npy", [wavelength_blue, x, y, z, E_blue_fwd])
np.save("./testfolder/modeblue2685.npy", [wavelength_blue2, x, y, z, E_blue_bwd])
np.save("./testfolder/modered937.npy", [wavelength_red, x, y, z, E_red])

"""
Then the .npy file containing the electric field maps can be loaded
"""
datafolder = os.path.dirname(os.path.abspath(__file__)) + "/testfolder"

red_beam = nt.BeamPair(937e-9, 0.3 * mW, 937e-9, 0.3 * mW)
blue_beam = nt.BeamPair(685.5e-9, 4 * mW, 685.6e-9, 4 * mW)
# red_beam = nt.Beam(937e-9, "f", 0.6 * mW)
# blue_beam = nt.Beam(685.5e-9, "f", 4 * mW)
trap = nt.Trap_beams(blue_beam, red_beam, propagation_axis="Z")

#Set surface, here a nanofiber with 250nm radius along Z
surface = nt.CylindricalSurface(axis = nt.AxisZ(), radius = 250e-9)
#We want to trap ground state cesium atoms
syst = nt.atomicsystem(nt.Caesium(), "6S1/2", f4)

#Create simulation object containing all of this
Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder, surface)

"""
If you want to parallelize the computation by using multiple cores at once, for speedup,
you can create a ParallelSimulator object specifying the number of cores used and set it. 
Otherwise, a SequentialSimulator using only one core will be created. 
"""
#Simul.simulator = ParallelSimulator(max_workers = 4)


t0 = time.time()
#Computes 2D total potential in the XY plane (transverse here) including CP 
trap2D = Simul.compute()
print("Simulation time :", time.time() - t0, "s")
#Save so if you run the simulation again you won't have to wait
Simul.save()
viz = nt.Viz(Simul, trapping_axis="X")
fig, ax, slider_ax = viz.plot_trap(mf=0, Pranges=[30, 30])
