# -*- coding: utf-8 -*-
#%%
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

datafolder = os.path.dirname(__file__)+"\\testfolder"

#%%

# red_beam = Nb.Beam(1064e-9,"f",0.8*mW)
# blue_beam = Nb.Beam(780e-9,"f",4.5*mW)

#red_beam = Nb.BeamPair(852.5252e-9,0.002*mW,852.5252e-9,0.002*mW)
#blue_beam = Nb.BeamPair(685.5e-9,4*mW,685.6e-9, 4*mW)
blue_beam = Nb.Beam(685.5e-9,"f",5*mW)
#trap = Nt.Trap_beams(blue_beam,red_beam,propagation_axis="Z")
trap = Nt.Trap_beams(blue_beam,propagation_axis="Z")

# standing_wave_red = Nb.BeamPair(1064e-9, 1*mW, 1064e-9, 1*mW)
# running_wave_blue = Nb.BeamPair(780e-9, 2.2*mW, 781e-9, 2.2*mW)
# trap = Nt.Trap_beams(running_wave_blue,standing_wave_red,propagation_axis="Z")

surface = vdw.Cylinder((0,0,0),250e-9,"Z")
surface = vdw.NoSurface()

#syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,S,1/2),f4)
f = f4
syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,P,3/2),f)

Simul = Ns.Simulation(syst,Nm.SiO2(),trap,datafolder,surface)

#%%
trap2D = Simul.compute_potential("XY",0)
#Simul.save()
viz = Nv.Viz(Simul, trapping_axis= 'X')
fig, ax, slider_ax = viz.plot_trap("XY", mf = 0, Pranges = [30])
#%%
trap1D = Simul.compute_potential_1D('X',0,0,contrib="tensor")
#Simul.save()
viz = Nv.Viz(Simul, trapping_axis = 'X')
fig, ax, slider_ax = viz.plot_trap1D("X", mf = range(-f,f+1),Pranges = [30])
#%%
#fig, ax, slider_ax = viz.plot_3axis(0, 0)

Simul.total_potential()[60,:]
#%%
#min = viz.get_min_trap(Simul, 'X',0)
# mf_index, edge, y_outside, trap_1D_Y_outside = viz.get_coord_trap_outside_structure(viz.trapping_axis, 0, 0, 0,edge_no_surface=None)
# min = viz.get_min_trap(y_outside, trap_1D_Y_outside)
# print(min)
#%%
print("Range N : ",Simul.atomicsystem.rangeN)
print("Alpha scalar : ", Simul.atomicsystem.alpha0/AMU)
print("Alpha vector : ", Simul.atomicsystem.alpha1/AMU)
print("Alpha tensor : ", Simul.atomicsystem.alpha2/AMU)

plt.plot(Simul.total_potential()[60],'.')
plt.show()
        
# %%
