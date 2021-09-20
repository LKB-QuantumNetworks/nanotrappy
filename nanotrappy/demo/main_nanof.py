import nanotrappy.trapping.atomicsystem as Na
import nanotrappy.trapping.beam as Nb
import nanotrappy.trapping.trap as Nt
import nanotrappy.trapping.simulation as Ns
import nanotrappy.trapping.structures as Nss
import nanotrappy.utils.materials as Nm
import nanotrappy.utils.viz as Nv
import nanotrappy.utils.vdw as vdw
from nanotrappy.utils.physicalunits import *
from arc import *
import numpy as np
import time


struct_nf = Nss.Nanofiber(Nm.SiO2(),Nm.air(), False, radius = 250e-9)
# wavelength_red = 937e-9
# wavelength_blue = 686.1e-9
wavelength_red = 1065e-9
wavelength_blue = 785e-9
# x = np.linspace(0,1000e-9,500)
# y = np.array([0])
# z = np.linspace(-500e-9,500e-9,100)

x = np.linspace(-1000e-9,1000e-9,400)
y = np.linspace(-1000e-9,1000e-9,400)
z = np.array([0])
#%%
beta_red = struct_nf.compute_beta_joos(wavelength_red)

E_red = np.zeros((3,len(x),len(y),1),dtype = "complex")
for k in range(len(x)):
    for i in range(len(y)):
       E_red[:,k,i,0] = struct_nf.electric_field_linear(x[k],y[i],z[0],wavelength_red,1,0)
       
#%%      
beta_blue = struct_nf.compute_beta_joos(wavelength_blue)
E_blue = np.zeros((3,len(x),len(y),1),dtype = "complex")
for k in range(len(x)):
    for i in range(len(y)):
       E_blue[:,k,i,0] = struct_nf.electric_field_linear(x[k],y[i],z[0],wavelength_blue,1,np.pi/2)
#%%
t0 = time.time()
E_red_dic = {}
E_red_dic["x"] = E_red[:,:,0]
E_red_dic["y"] = E_red[:,:,1]
E_red_dic["z"] = E_red[:,:,2]

E_blue_dic ={}
E_blue_dic["x"] = E_blue[:,:,0]
E_blue_dic["y"] = E_blue[:,:,1]
E_blue_dic["z"] = E_blue[:,:,2]

lambdas = {'lambda_blue_fwd': [wavelength_blue],'lambda_red_fwd': [wavelength_red]}
trap = Nt.Trap(nblue = 1,nred = 1,Pblue = 25e-3 ,Pred = 4.4e-3,**lambdas)
syst = Na.atomicsystem_dico(Caesium(), Na.atomiclevel(6,pu.S,1/2), Na.atomiclevel(6,pu.P,3/2))
Simul_Nanof = Nss.Outside_Simulation(syst,trap,[[E_blue_dic],[E_red_dic]],x,y,z,Nm.SiO2())
trap_2D_Y_new = Simul_Nanof.simulate(Simul_Nanof.system.groundstate,pu.f4,[0],"XY",0,0,0)
print(time.time() - t0)