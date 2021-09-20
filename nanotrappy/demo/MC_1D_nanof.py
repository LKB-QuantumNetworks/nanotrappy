import nanotrappy
from sympy import *
import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
from scipy.misc import derivative
from scipy.interpolate import interp1d

import progressbar
import time
import arc
import os
from nanotrappy.utils.physicalunits import *

# -*- coding: utf-8 -*-
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

g = 9.81

wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
lambda_a = 852.34727582e-9 #m #wavelength
gamma = 2*np.pi*5.234e6 #Hz #decay rate
Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)

# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

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
trap1D = np.real(Simul.compute_potential_1D('X',0,0)[:,4])

def forcex(x,potential):
    potential = kB*potential/1000
    force = -np.gradient(potential,x)
    func = interp1d(x,force)
    return func

force = forcex(Simul.x,trap1D)

viz = Nv.Viz(Simul, 'X')
mf, edge, x_trap, trap = viz.get_coord_trap_outside_structure(Simul.axis_name,Simul.plane_coord1,Simul.plane_coord2)
ymind, ymin, trap_depth, prominence =  viz.get_min_trap(x_trap,trap)

#%%
def af2(t,y):
    #y is a list here ?
    potential = trap1D*np.cos(2*np.pi*t*1e9)
    force = forcex(Simul.x,potential)
    
    x = y[1]
    if x < 250e-9:
        x = 500e-9
    if x >= 1000e-9:
        x = 500e-9
    vx = y[0]
    if x > 1e-6:
        print("x", x)
        print(1/mass*force(x))
        print(vx)
    
    return [1/mass*force(x), vx]

T0 = 1e-4
start = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

print("start Dorman-Prince 5 algo:")

nb = 2 # nb particles
vxr = np.random.normal(loc=0.0,scale=np.sqrt(kB*T0/mass),size=nb)
# scale a = sqrt(kT/m) => and v in units of vL = sqrt(2U0/mass) => vL^2/a^2 = 2U0/kT => a = VL sqrt(kT/2U0)
# but I have chosen before U0 = kb* 1mK so a = sqrt(T/(2)) with T in mK

sol_arr=[]

tstart = 0.0
tend = 1e-4 # !!! time is in unit of tL
dt = 5e-8 # 5% period in tL unit

pbar = progressbar.ProgressBar(maxval=len(vxr)).start()

for i,vi in enumerate(vxr):
    vxi = vi
    pbar.update(i)

    xi = np.random.normal(ymin+edge,50e-9)
    if xi < Simul.axis[0]:
        xi = Simul.axis[0]
    if xi >= Simul.axis[-1]:
        xi = Simul.axis[-1]

    #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
    
    sol = solve_ivp(af2, (tstart,tend), (vxi,xi), t_eval = np.linspace(tstart,tend,int((tend-tstart)/dt),endpoint=True),method=BESTMETHOD,atol=1e-6,rtol=1e-10)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr.append(sol)
    
end = time.time()
pbar.finish()
print('finished in (seconds):', end - start)

print('--- rk45 (DP5) ----')
print('message:', sol.message)
print('sol status:', sol.status)
if sol.status ==0:
    print('SUCCESS, reached end.')
print('sol arry', len(sol_arr))
print('sol shape:',sol.y.shape)

print('--saving traj data to .npz file--')

#%%
fig,ax = plt.subplots()
for k in range(len(vxr)):
    plt.plot(sol_arr[k]['t'],sol_arr[k]['y'][1])
    print(k)

