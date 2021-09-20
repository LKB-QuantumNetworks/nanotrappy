import nanotrappy.trapping.atomicsystem as Na
import nanotrappy.trapping.beam as Nb
import nanotrappy.trapping.trap as Nt
import nanotrappy.trapping.simulation as Ns
import nanotrappy.utils.materials as Nm
import nanotrappy.utils.viz as Nv
import nanotrappy.utils.vdw as vdw
from nanotrappy.utils.physicalunits import *
from arc import *
import time
#%%

datafolder = r'C:/Users/Adrien/Documents/1. These LKB/1. Simulations waveguide/Microtoroids/fields_nanotrappy/smaller_ranges'
#%%
blue_beam = Nb.Beam(8.499614171321733e-7,"f",1*mW)
red_beam = Nb.Beam(9.000675507801087e-7,"f",1*mW)
standing_wave = Nb.BeamPair(9.000675507801087e-7, 2.2*mW, 9.000675507801087e-7, 2.2*mW)
trap = Nt.Trap_beams(blue_beam,standing_wave,propagation_axis="Z")
#%%

surface = vdw.Cylinder((10.5e-6,0,0),1.5e-6,"Z")

syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,P,3/2),f4)
#%%
Simul = Ns.Simulation(syst,Nm.SiO2(),trap,datafolder,surface)

#%%
t0 = time.time()
trap2D = Simul.compute_potential("XY",0)
print("Simulation time :", time.time() - t0, "s")
Simul.save()

#%%
trap1D = Simul.compute_potential_1D('X',0,0)
Simul.save()
#%%
viz = Nv.Viz(Simul, 'X')
#%%
fig, ax, slider_ax = viz.plot_trap("XY",0,[100,100])

#%%
fig, ax, slider_ax = viz.plot_trap1D("X",[-4,-3,-2,-1,0,1,2,3,4],[100,100])
#%%
fig, ax, slider_ax = viz.plot_3axis(0, 0)
#%%
mf_index, edge, y_outside, trap_1D_Y = viz.get_coord_trap_outside_structure(viz.trapping_axis, 
                                                                             0, 0)
omegay = viz.get_trapfreq(y_outside, trap_1D_Y)
print("Trapping frequency =", omegay, " Hz")

    