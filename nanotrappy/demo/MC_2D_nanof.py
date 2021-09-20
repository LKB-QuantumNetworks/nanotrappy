from sympy import *
import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
from scipy.misc import derivative
from scipy.interpolate import interp1d, interp2d

from matplotlib import colors

import progressbar
import time
import arc
import os
import nanotrappy.trapping.atomicsystem as Na
import nanotrappy.trapping.beam as Nb
import nanotrappy.trapping.trap as Nt
import nanotrappy.trapping.simulation as Ns
import nanotrappy.utils.materials as Nm
import nanotrappy.utils.viz as Nv
import nanotrappy.utils.vdw as vdw
from nanotrappy.utils.utils import *
from nanotrappy.utils.physicalunits import *
from arc import *
import time

g = 9.81

wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
lambda_a = 852.34727582e-9 #m #wavelength
gamma = 2*np.pi*5.234e6 #Hz #decay rate
Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)

# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
# U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

#%%

datafolder = os.path.dirname(__file__)+"\\testfolder"
#%%
datafolder = r"C:/Users/Adrien/Documents/1. These LKB/1. Simulations waveguide/Nanofiber trap/E_fields_MC"

# red_beam = Nb.Beam(1064e-9,"f",5*mW)
blue_beam = Nb.Beam(780e-9,"f",26*mW)
standing_wave = Nb.BeamPair(1064e-9, 5*mW, 1064e-9, 5*mW)
trap = Nt.Trap_beams(blue_beam,standing_wave,propagation_axis="Z")

surface = vdw.Cylinder((0,0),250e-9,"Z")

syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,S,1/2),f4)

Simul = Ns.Simulation(syst,Nm.SiO2(),trap,datafolder,surface)
viz = Nv.Viz(Simul, 'X')
mf_index, edge, y_outside, trap_1D_Y_outside = viz.get_coord_trap_outside_structure(viz.trapping_axis, 0, 0)
ymin_ind, y_min, trap_depth, trap_prominence = viz.get_min_trap(y_outside, trap_1D_Y_outside)
fig, ax, slider_ax = viz.plot_trap1D("X",0,[30,30])

#%%
plane = "XZ"
trap2D = np.real(Simul.compute_potential(plane,0)[:,:,4])
fig, ax, slider_ax = viz.plot_trap(plane,0,[30,30])
Simul.save()

# trap1D = np.real(Simul.compute_potential_1D('X',0,0)[:,4])
#%%

def force(potential):
    potential = kB*potential/1000
    # force = -np.gradient(potential,dx,axis = axis_number)
    grad_x, grad_z = np.gradient(potential,Simul.x,Simul.z)
    forcex, forcez = -np.array(grad_x), -np.array(grad_z)
    print(forcex.shape)
    forcex = interp2d(Simul.x, Simul.z, np.transpose(forcex))
    forcez = interp2d(Simul.x, Simul.z, np.transpose(forcez))
    # func = interp1d(x,force)
    return forcex, forcez
    
diffsx = np.array([x - Simul.x[i - 1] for i, x in enumerate(Simul.x)][1:])
grad_x = -np.gradient(trap2D[1:,1:],diffsx,axis=0)

diffsz = [x - Simul.z[i - 1] for i, x in enumerate(Simul.z)][1:]
grad_z = -np.gradient(trap2D[1:,1:],diffsz,axis=1)

# x_grid,z_grid = np.meshgrid(Simul.x,Simul.z)
forcex, forcez = force(trap2D)
#%%
def af2(t,y):
    #y is a list here ?
    x = y[1]
    if x < Simul.x[0]:
        x = Simul.x[0]
    if x >= Simul.x[-1]:
        x = Simul.x[-1]
    z = y[3]
    if z < Simul.z[0]:
        z = Simul.z[0]
    if z >= Simul.z[-1]:
        z = Simul.z[-1]
    
    vx = y[0]
    vz = y[2]
    return [1/mass*forcex(x,z), vx, 1/mass*forcez(x,z), vz]

#%%

start = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

print("start Dorman-Prince 5 algo:")

T0 = 1e-4
nb = 5 # nb particles
vxr = np.random.normal(loc=0.0,scale=np.sqrt(kB*T0/mass),size=nb)
vzr = np.random.normal(loc=0.0,scale=np.sqrt(kB*T0/mass),size=nb)
# scale a = sqrt(kT/m) => and v in units of vL = sqrt(2U0/mass) => vL^2/a^2 = 2U0/kT => a = VL sqrt(kT/2U0)
# but I have chosen before U0 = kb* 1mK so a = sqrt(T/(2)) with T in mK

sol_arr=[]

tstart = 0.0
tend = 1e-5 # !!! time is in unit of tL
dt = 1e-8 # 5% period in tL unit

pbar = progressbar.ProgressBar(maxval=len(vxr)).start()
t0 = time.time()
for i,vi in enumerate(vxr):
    vxi = vi
    vzi = vzr[i]
    pbar.update(i)

    xi = np.random.normal(y_min+edge,40e-9)
    zi = np.random.normal(500e-9,40e-9)
    if xi < Simul.x[0]:
        xi = Simul.x[0]
    if xi >= Simul.x[-1]:
        xi = Simul.x[-1]
    if zi < Simul.z[0]:
        zi = Simul.z[0]
    if zi >= Simul.z[-1]:
        zi = Simul.z[-1]

    #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
    
    sol = solve_ivp(af2, (tstart,tend), (vxi,xi,vzi,zi), t_eval = np.linspace(tstart,tend,int((tend-tstart)/dt),endpoint=True),method=BESTMETHOD,atol=1e-6,rtol=1e-10)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr.append(sol)
    
print("total time = ", time.time()-t0)

#%%
coord1, coord2 = set_axis_from_plane(plane, viz.simul)
trap = np.real(viz.simul.total_potential())[:,:,4]
trap_noCP = np.real(viz.simul.total_potential_noCP[:,:,4])
fig, ax = plt.subplots()
for k in range(len(vxr)):
    plt.plot(sol_arr[k]['y'][1],sol_arr[k]['y'][3],zorder = 1)
    print(k)
a = plt.pcolormesh(coord1,coord2,np.transpose(trap),shading = "gouraud",
                   norm = colors.TwoSlopeNorm(vmin=min(np.min(trap_noCP)/2,-0.001), 
                   vcenter=0, vmax=max(np.max(trap_noCP)*2,0.001)), cmap = "seismic_r")