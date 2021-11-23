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
wavelength_blue_fwd = 685.5e-9
wavelength_blue_bwd = 685.6e-9
y = np.linspace(-800e-9, 800e-9, 201)
x = np.linspace(-800e-9, 800e-9, 201)
z = np.array([0])
P = 1  # in Watts
theta = 0  # radians
E_red = nanof.compute_E_linear(x, y, z, wavelength_red, P, theta)
E_blue_fwd = nanof.compute_E_linear(x, y, z, wavelength_blue_fwd, P, theta)
E_blue_bwd = nanof.compute_E_linear(x, y, z, wavelength_blue_bwd, P, theta)
np.save("./testfolder/modeblue685.npy", [wavelength_blue_fwd, x, y, z, E_blue_fwd])
np.save("./testfolder/modeblue2685.npy", [wavelength_blue_bwd, x, y, z, E_blue_bwd])
np.save("./testfolder/modered937.npy", [wavelength_red, x, y, z, E_red])
