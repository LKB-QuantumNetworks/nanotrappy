import numpy as np

hbar = 1.054571628*10**-34 #(* Planck constant*)
cc = 299792458.0 #(*Speed of light*)
kB = 1.3806504*10**-23 #(*Boltzmann constant*)
NA = 6.02214179*10**23 #(* Abogadro number *)
ee = 1.602176487*10**-19 #(*Electron Charge*)
me = 9.10938215*10**-31 #(* Electron Mass *)
ϵ0 = 8.854187817*10**-12 #(*Vacuum permittivity*)
μ0 = 1.2566370614*10**-6 #(* vacuum permeability*)
a0 = 52.9177*10**-12 #(*Bohr radius*)
mCs = 2.20695*10**-25 #(*Cs mass (kg)*)
hP = hbar*2*np.pi
aB = 5.291772108*10**-11
au = ee*aB
μB = (ee*hbar)/(2*me)
EH = (me*ee**4)/(4*np.pi*ϵ0*hbar)**2 #(*Hartree Energy*)
AMU = 1.6487772754*10**-41 


## Conversion relations
MHz = 10**6
GHz = 10**9 
THz = 10**12
nm = 10**-9
mm = 10**-3 
μm = 10**-6
cm = 10**-2
mW = 10**-3
kHz = 10**3
μW = 10**-6
μK = 10**-6 
mK = 10**-3
μs = 10**-6 
ns = 10**-9
G = 10**-4
pi = np.pi

### Zeeman states
f0 = 0
f1 = 1
f2 = 2
f3 = 3
f4 = 4
f5 = 5


# Fine stucture nomenclature
S=0
P=1
D=2
F=3
G=4

#Trap useful constants
wavelength_equality_tolerance =  5*nm