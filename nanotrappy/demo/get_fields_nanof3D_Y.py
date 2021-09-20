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
wavelength_red = 852.5252e-9
wavelength_blue = 685.5e-9
wavelength_blue2 = 685.6e-9
# x = np.linspace(0,1000e-9,500)
# y = np.array([0])
# z = np.linspace(-500e-9,500e-9,100)

# y = np.linspace(-1000e-9,1000e-9,101)
y = np.linspace(0,800e-9,101)
x = np.linspace(0,800e-9,101)
z = np.array([0])
#z = np.linspace(0,1000e-9,101)
#%%
beta_red = struct_nf.compute_beta_joos(wavelength_red)

E_red = np.zeros((3,len(x),len(y),len(z)),dtype = "complex")
for k in range(len(x)):
    for i in range(len(y)):
        for j in range(len(z)):
            E_red[:,k,i,j] = struct_nf.electric_field_linear(x[k],y[i],z[j],wavelength_red,1,0)
       
#%%      
beta_blue = struct_nf.compute_beta_joos(wavelength_blue)
E_blue = np.zeros((3,len(x),len(y),len(z)),dtype = "complex")
for k in range(len(x)):
    for i in range(len(y)):
        for j in range(len(z)):
            E_blue[:,k,i,j] = struct_nf.electric_field_linear(x[k],z[j],-y[i],wavelength_blue,1,0)


E_blue_Y = np.zeros((3,len(x),len(y),len(z)),dtype = "complex")
E_blue_Y[0,:,:,:] = E_blue[0,:,:,:]
E_blue_Y[1,:,:,:] = E_blue[2,:,:,:]
E_blue_Y[2,:,:,:] = -E_blue[1,:,:,:]

#%%      
beta_blue2 = struct_nf.compute_beta_joos(wavelength_blue2)
E_blue2 = np.zeros((3,len(x),len(y),len(z)),dtype = "complex")
for k in range(len(x)):
    for i in range(len(y)):
        for j in range(len(z)):
            E_blue2[:,k,i,j] = struct_nf.electric_field_linear(x[k],y[i],z[j],wavelength_blue2,1,0)






np.save("./testfolder/modeblue685_Y.npy",[wavelength_blue,x,y,z,E_blue])
# np.save("./testfolder/modeblue2685.npy",[wavelength_blue2,x,y,z,E_blue2])
# np.save("./testfolder/modered852.npy",[wavelength_red,x,y,z,E_red])