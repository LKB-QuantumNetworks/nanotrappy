---
template: main.html
title: Setting up a simulation
---

# State insensitive trap around a nanofiber

```python linenums="1"
import nanotrappy as nt
from nanotrappy.utils.physicalunits import *
import time, os
import matplotlib.pyplot as plt
```

In this tutorial we will reproduce the trapping schemes around a nanofiber discussed in Lacroûte et al. "[A state-insensitive, compensated nanofiber trap][1]" New J. Phys. 14 (2012).

## Specifying the atom we want to trap

First we have to specify for which atom the trapping potentials will be computed.
This part is based on the [Alkali Rydberg Calculator][3] library.
Originally conceived for studying Rydberg atoms, we have included the hyperfine levels of such Alkali atoms.

```python linenums="5"
syst = Na.atomicsystem(Caesium(), "6S1/2",f=4)
```

An alkali atom has to be chosen, an atomic level from the fine structure (here the ground state of Cesium 6S1/2) and the F of the hyperfine level studied.
Indeed the dipole force felt by the atoms depends on the hyperfine level they are in.
Note that the state can be written as a string.

## Creating the beams and trapping scheme

There are many different designs that can be used for trapping atoms with evanescent fields.
We have then to specify how many beams we want to use, at which wavelengths and powers.

If the structures are pre-implemented in `nanotrappy`, the right modes will then be computed.
If not, a folder containing the modes will have to be specified, consult the [mode formatting][2] section to see how the modes have to be saved in order to be used by `nanotrappy`.

Beam objects are the basic building blocks of your trap in `nanotrappy`.

```python linenums="5"
blue_beam = nt.Beam(1064e-9,direction = "f", power = 25*mW)
red_beam = nt.Beam(780e-9, direction = "f", power = 7*mW)
```

Here we define two beams, blue- and red-detuned from the Cesium D2 line respectively.
"f" means the beam is propagating forward, along the propagation axis of the waveguide. "b" stands for backwards.
This is important as chiral effects can arise when coupling tightly focused light to quantum emitters for example.

Similarly, if your trap consists of a red standing wave and a blue slightly detuned running wave as in [1][1],

```python linenums="5"
red_beam = nt.BeamPair(937e-9, 0.3 * mW, 937e-9, 0.3 * mW) #Standing wave
blue_beam = nt.BeamPair(685.5e-9, 4 * mW, 685.6e-9, 4 * mW) #Running wave for cancellation of the vector shift
```

These beams are bundled into a trap object that will then be used in the simulation.

```python linenums="7"
trap = nt.Trap_beams(blue_beam,red_beam)
```

### Optional : Set surfaces for the Van der Waals interactions

See [the dedicated page][4] for more information. The nanofiber is a cylinder with axis of symmetry the propagation axis Z.
We define then a cylinder centered at the origin of the grid, with 250 nm radius and with "Z" revolution axis.

```python linenums="9"
surface = nt.CylindricalSurface(radius=250e-9, axis = AxisZ(coordinates = (0,0)))
```

##Define the Simulation object

The simulation object will then be a bundle of everything needed to compute the trapping potentials.
The right modes corresponding to the trapping scheme, the atomic level and the surface and material (for CP interactions).
This class will be useful mostly for keeping the results of the simulation and visualisation purposes further on.

```python linenums="7"
Simul = nt.Simulation(syst,SiO2(),trap,datafolder,surface)
Simul.geometry = PlaneXY(normal_coord=0)
```

We also defined a simulation geometry, meaning on which plane or along which axis we want to run the computation for future observation.

##Run the simulation in 2 dimensions

```python linenums="9"
trap2D = Simul.compute()
```

`compute` then takes no argument as everything has been defined earlier.
Careful that the simulation in 2D can be time consuming as the potential is computed point by point on the given grid.
As an example, for a 100x100 grid, times of the order of 30s are expected. See the [parallelizing][5] section for more information on speeding up your calculation.

##Visualizing the results

The [viz][6] submodules has a few methods to visualise and analyze your results.
You have first to create a Vizualizer object to access these methods. The trapping axis corresponds to the axis that is going away from the main surfaces.
For the nanofiber it can be either X or Y, depending on your experimental parameters.

The easiest is to plot the 2D potential with power sliders as follows :

```python linenums="11"
viz = Viz(Simul, trapping_axis="X")
fig, ax, slider_ax = viz.plot_trap(mf=0, Pranges=[10, 2], increments=[0.01, 0.01])
```

Pranges and increments are in mW.
It is important to return the image and sliders as this will ensure the sliders do not get stuck.

[![Nanofiber plot][7]{: style="height:600px;width:600px"}][7]

We see here 2 trapping sites, one on each side of the fiber (around 230 nm from the surface each).

If you are satisfied, you can save your simulation with `Simul.save()`.
This will create a new folder in your mode folder and save a JSON file with the list of parameters of the simulation as well as a NPY with the potentials (not summed) and the eigenvectors in the m<sub>f</sub> basis.

When you run a simulation, `nanotrappy` looks in the results folder if a simulation with the same parameters has already been saved. If so, it just reloads it without running it again.

A 1D simulation can also be run to look at how the different m<sub>f</sub> states are affected by the differential vector shift coming from the local circular polarisation of the light acting as a fictitious magnetic field.
To do so, you just have to modifiy the `geometry` of the simulation and run the computation again:

```python linenums="10"
Simul.geometry = AxisX(coordinates = (0,0))
trap1D = Simul.compute()
fig, ax, slider_ax = Nv.plot_trap(mf = [-4,-3,-2,-1,0,1,2,3,4], Pranges=[10, 2], increments=[0.01, 0.01])
Simul.save()
```

[![1D trap around a nanofiber][8]][8]

[1]: https://doi.org/10.1088/1367-2630/14/2/023056
[2]: modeformatting.md
[3]: https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/
[4]: casimirpolder.md
[5]: parallelizing.md
[6]: reference/viz.md
[7]: ./images/nanofiber_plot.png
[8]: ./images/nanofiber_1D_splitting.png
