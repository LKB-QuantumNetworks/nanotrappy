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

datafolder = os.path.dirname(__file__)+r"/testfolder_HW1"
datafolder = "C:/Users/Adrien/Documents/1. These LKB/1. Simulations waveguide/legume/Aligning lower band to D2 Rb line/a = 230/E_fields"
datafolder = "C:/Users/Adrien/Documents/1. These LKB/1. Simulations waveguide/HalfW1/Efield for trapping/a=230_L=265_r=0.3"
datafolder = "C:/Users/Adrien/Documents/1. These LKB/1. Simulations waveguide/HalfW1/Efield for trapping/a=230_L=353_r=0.3_systematic"
print(datafolder)
#%%

blue_beam = Nb.Beam(7.205163862718708e-07,"f",25*mW)
red_beam = Nb.Beam(8.161957450024562e-7,"b",7*mW)

blue_beam = Nb.Beam(7.3329e-07,"f",25*mW)
red_beam = Nb.Beam(8.04543013e-07,"f",25*mW)
# blue_beam = Nb.Beam(676.74e-09,"f",25*mW)
# red_beam = Nb.Beam(984.831175e-09,"f",7*mW)

standing_wave = Nb.BeamPair(8.161957450024562e-7, 2.2*mW, 8.161957450024562e-7, 2.2*mW)
standing_wave_blue = Nb.BeamPair(7.1982260e-07, 2.2*mW, 7.20597426e-07, 2.2*mW)

standing_wave_red = Nb.BeamPair(8.04543013e-07, 2.2*mW, 8.00353804e-07, 2.2*mW)
standing_wave_blue = Nb.BeamPair(7.3329e-07, 2.2*mW, 7.33829109e-07, 2.2*mW)


trap = Nt.Trap_beams(standing_wave_blue,standing_wave_red,propagation_axis="X")
# trap = Nt.Trap_beams(blue_beam,red_beam,propagation_axis="X")

surface = vdw.Plane(np.array([0,1e-9,0]),np.array([0,0,0]))

# trap = Nt.Trap_beams(blue_beam,red_beam,propagation_axis="Z")

syst = Na.atomicsystem(Rubidium(),Na.atomiclevel(5,S,1/2),f2)

Simul = Ns.Simulation(syst,Nm.GaInP(),trap,datafolder,surface)
#%%
t0 = time.time()
trap2D = Simul.compute_potential_avg("XY",0)
print(time.time() - t0)
Simul.save()
#%%
viz = Nv.Viz(Simul,"Y")
#%%
fig, ax, slider_ax = viz.plot_trap("XY", mf = 0, Pranges = [50,50])
#%%
trap1D = Simul.compute_potential_1D('Y',-115e-9,0)
Simul.save()
#%%
fig, ax, slider_ax = viz.plot_trap1D("Y",[-2,-1,0,1,2],[4,1])

#%%
fig, ax, slider_ax = viz.plot_3axis(-115e-9, 0)
