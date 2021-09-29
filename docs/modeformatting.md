---
template: main.html
title: Formatting of the computed modes of the electric field
---

In order to be used by `nanotrappy`, the fields in the data folder have to have a specific formatting.

The names of the files don't matter nor their order.

!!! attention
The files must be numpy arrays stored in .npy files. The array must be done as follows

    [![mode.npy = lmbda, x, y, z, E][1]][1]

    where E is a 4-dimensional np.array of size **(3,len(x),len(y),len(z))**.

    The first dimension corresponds to Ex, Ey and Ez.

The program will check all the .npy files in the specified folder, open them and store the first element (the wavelength) in a list.
It will then compare it to the specified wavelengths for trapping to check is the modes are available.

## Saving electric field simulations from nanotrappy pre-implemented structure

Create a Nanofiber with the right material, cladding and radius:

```python linenums="5"
nanof = nt.Nanofiber(nt.SiO2(), nt.air(), radius=250e-9)
```

Define wavelengths used for trapping: we will have a red standing wave so only one is needed for the red-detuned beam and two blue-detuned beams slightly detuned from each other.

```python linenums="6"
wavelength_red, wavelength_blue_fwd, wavelength_blue_bwd = 937e-9, 685.5e-9, 685.6e-9
y = np.linspace(0, 800e-9, 201)
x = np.linspace(0, 800e-9, 201)
z = np.array([0])
P = 1 #in Watts
theta = 0 #radians
```

Then compute the actual fields in 3D and save them. Be careful, we should always save fields in 3D even if we computed them in a 2D plane.
In that case the 3rd dimension should be set as an array of length 1 (see z here).

```python linenums="12"
E_red = nanof.compute_E_linear(x,y,z,wavelength_red,P,theta)
E_blue_fwd = nanof.compute_E_linear(x,y,z,wavelength_blue_fwd,P,theta)
E_blue_bwd = nanof.compute_E_linear(x,y,z,wavelength_blue_bwd,P,theta)
np.save("./testfolder/modeblue685.npy", [wavelength_blue_fwd, x, y, z, E_blue_fwd])
np.save("./testfolder/modeblue2_685.npy", [wavelength_blue_bwd, x, y, z, E_blue_bwd])
np.save("./testfolder/modered937.npy", [wavelength_red, x, y, z, E_red])
```

[1]: ./images/mode_formatting.PNG
