#%%
# -*- coding: utf-8 -*-
from nanotrappy.utils.vdw2 import CylindricalSurface, PlaneSurface
from nanotrappy.trapping.geometry import AxisX, AxisZ, PlaneXY
from nanotrappy.trapping.simulator2 import SequentialSimulator, ParallelSimulator
from nanotrappy.trapping.simulation2 import Simulation
from nanotrappy.trapping.beam import *
from nanotrappy.trapping.trap import Trap_beams
from nanotrappy.trapping.atomicsystem import atomicsystem
from nanotrappy.utils.viz2 import Viz
from nanotrappy import Caesium, SiO2
from nanotrappy.utils.physicalunits import *
import time, os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    datafolder = os.path.dirname(os.path.abspath(__file__)) + "/testfolder"

    # red_beam = BeamPair(937e-9, 0.3 * mW, 937e-9, 0.3 * mW)
    # blue_beam = BeamPair(685.5e-9, 4 * mW, 685.6e-9, 4 * mW)
    # red_beam = Beam(937e-9, "f", 0.6 * mW)
    # blue_beam = Beam(685.5e-9, "f", 4 * mW)
    red_beam = BeamPair(1064e-9, 2.2 * mW, 1064e-9, 2.2 * mW)
    blue_beam = Beam(780e-9, "f", 25 * mW)
    trap = Trap_beams(blue_beam, red_beam, propagation_axis="Z")

    # surface = nt.Cylinder((0, 0, 0), 250e-9, "Z")
    syst = atomicsystem(Caesium(), "6P3/2", f4)

    surface = CylindricalSurface(axis=AxisZ(), radius=250e-9)
    Simul = Simulation(syst, SiO2(), trap, datafolder, surface)
    Simul.simulator = ParallelSimulator(max_workers=None)
    Simul.geometry = PlaneXY(normal_coord=0)
    Simul.geometry = AxisX(coordinates=(0, 0))

    Simul.compute()
    Simul.save()

    viz = Viz(Simul, trapping_axis="X")
    # fig, ax, slider_ax = viz.plot_trap(mf=range(-f4, f4 + 1), Pranges=[30, 10], increments=[0.1, 0.1])
    fig, ax, slider_ax = viz.plot_trap(mf=0, Pranges=[10, 2], increments=[0.01, 0.01])
    plt.show()


# %%