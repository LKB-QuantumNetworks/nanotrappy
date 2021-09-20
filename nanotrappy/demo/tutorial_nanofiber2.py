# -*- coding: utf-8 -*-
import nanotrappy as nt
from nanotrappy.utils.physicalunits import *
from arc import *
import time

#%%

if __name__ == "__main__":
    datafolder = os.path.dirname(__file__) + "\\testfolder"

    blue_beam = nt.Beam(937e-9, "f", 25 * mW)
    red_beam = nt.Beam(685.5e-9, "f", 7 * mW)
    # standing_wave = Nb.BeamPair(1070e-9, 2.2*mW, 1070e-9, 2.2*mW)
    trap = nt.Trap_beams(blue_beam, red_beam, propagation_axis="Z")

    surface = nt.Cylinder((0, 0), 250e-9, "Z")

    syst = nt.atomicsystem(Caesium(), nt.atomiclevel(6, P, 3 / 2), f4)

    Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder, surface)

    #%%
    t0 = time.time()
    trap2D = Simul.compute_potential("XY", 0)
    print("Simulation time :", time.time() - t0, "s")
    Simul.save()
    #%%
    viz = nt.Viz(Simul, trapping_axis="X")
    fig, ax, slider_ax = viz.plot_trap("XY")
    #%%
    # trap1D = Simul.compute_potential_1D('X',0,0)
    # Simul.save()
    # #%%
    # fig, ax, slider_ax = Nv.plot_trap1D(Simul,"X",[-4,-3,-2,-1,0,1,2,3,4],[10,30])

    #%%
    # Nv.get_min_trap(Simul, 'X',0)
    # #%%
    # Nv.get_trapfreq(Simul, 'X',0)

    # trap1D = Simul.total_potential()[:,49]
    # fig, ax = plt.subplots()
    # plt.plot(Simul.x, trap1D)
    # plt.plot(Simul.x, Simul.potentials[0,:,49], color = "blue")
    # plt.plot(Simul.x, Simul.potentials[1,:,49], color = "red")
