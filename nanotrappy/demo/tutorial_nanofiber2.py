# -*- coding: utf-8 -*-
import nanotrappy as nt
import time, os
import matplotlib.pyplot as plt
from nanotrappy.utils.physicalunits import *

if __name__ == "__main__":
    #Choose the folder where you electric fields are stored
    datafolder = os.path.dirname(os.path.abspath(__file__)) + "\\testfolder"

    #Define the red and blue beams (counterpropagating in this case)
    red_beam = nt.BeamPair(937e-9, 0.95 * mW, 937e-9, 0.95 * mW)
    blue_beam = nt.BeamPair(685.5e-9, 16 * mW, 685.6e-9, 16 * mW)
    
    #Or simple propagation
    red_beam = nt.Beam(937e-9, 'f', 0.95 * mW)
    blue_beam = nt.Beam(685.5e-9, 'f', 16 * mW)
    
    trap = nt.Trap_beams(blue_beam, red_beam)
    
    #trap Cesium atoms in their ground state
    syst = nt.atomicsystem(nt.Caesium(), "6S1/2", f4)
    
    #We define a cylinder of radius 250e-9 to include the CP interactions
    surface = nt.CylindricalSurface(axis=nt.AxisZ(), radius=250e-9)
    Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder, surface)
    # Simul.simulator = ParallelSimulator(max_workers=None)
    
#%% Plot in 2D
    Simul.geometry = nt.PlaneXY(normal_coord=0)
    Simul.compute()
    Simul.save()
    viz = nt.Viz(Simul, trapping_axis="X")
    fig, ax, slider_ax = viz.plot_trap(mf=0, Pranges=[30, 10], increments=[0.01, 0.01])

#%% Plot along the x axis, can visualize many mf states at once
    Simul.geometry = nt.AxisX(coordinates=(0, 0))
    Simul.compute()
    Simul.save()
    viz = nt.Viz(Simul, trapping_axis="X")
    fig, ax, slider_ax = viz.plot_trap(mf=[-2,-1,0,1,2], Pranges=[30, 10], increments=[0.01, 0.01])
