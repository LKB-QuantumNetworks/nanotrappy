#%%
# -*- coding: utf-8 -*-
import nanotrappy as nt
from nanotrappy.utils.physicalunits import *
import time, os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    datafolder = os.path.dirname(__file__) + "\\testfolder"

    # red_beam = nt.Beam(1064e-9,"f",0.8*mW)
    # blue_beam = nt.Beam(780e-9,"f",4.5*mW)
    # trap = nt.Trap_beams(blue_beam,red_beam,propagation_axis="Z")

    red_beam = nt.BeamPair(937e-9, 0.3 * mW, 937e-9, 0.3 * mW)
    blue_beam = nt.BeamPair(685.5e-9, 4 * mW, 685.6e-9, 4 * mW)
    # red_beam = nt.Beam(937e-9, "f", 0.6 * mW)
    # blue_beam = nt.Beam(685.5e-9, "f", 4 * mW)
    trap = nt.Trap_beams(blue_beam, red_beam, propagation_axis="Z")

    # standing_wave_red = Nb.BeamPair(1064e-9, 1*mW, 1064e-9, 1*mW)
    # running_wave_blue = Nb.BeamPair(780e-9, 2.2*mW, 781e-9, 2.2*mW)
    # trap = Nt.Trap_beams(running_wave_blue,standing_wave_red,propagation_axis="Z")

    surface = nt.Cylinder((0, 0, 0), 250e-9, "Z")
    syst = nt.atomicsystem(nt.Caesium(), "6S1/2", f4)
    Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder, surface)

    # t0 = time.time()
    # trap2D = Simul.compute_potential("XY", 0)
    # print("Simulation time :", time.time() - t0, "s")
    # Simul.save()
    # viz = nt.Viz(Simul, trapping_axis="X")
    # fig, ax, slider_ax = viz.plot_trap("XY", mf=0, Pranges=[30, 30])

    trap1D = Simul.compute_potential_1D("X", 0, 0)
    # Simul.save()
    viz = nt.Viz(Simul, trapping_axis="X")
    # fig, ax, slider_ax = viz.plot_trap1D("X", mf=range(-f4, f4 + 1), Pranges=[30, 30])

    params = {
        "ymin": 251.0e-9,
        "Pmin1": 0.001 * mW,
        "Pmax1": 10.0 * mW,
        "Pstep1": 0.05 * mW,
        "Pmin2": 0.001 * mW,
        "Pmax2": 1.1 * mW,
        "Pstep2": 0.01 * mW,
    }

    viz.optimize_and_show(**params)
