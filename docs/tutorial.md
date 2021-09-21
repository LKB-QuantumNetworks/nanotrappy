---
template: main.html
title: Setting up a simulation
---

# State insensitive trap around a nanofiber

```python linenums="1"
import nanotrappy
import nanotrappy.trapping.atomicsystem as Na
import nanotrappy.trapping.beam as Nb
import nanotrappy.trapping.trap as Nt
import nanotrappy.trapping.simulation as Ns
import nanotrappy.utils.materials as Nm
import nanotrappy.utils.viz as Nv
import nanotrappy.utils.vdw as vdw
from nanotrappy.utils.physicalunits import *
```

In this tutorial we will reproduce the trapping schemes around a nanofiber discussed in Lacroûte et al. "A state-insensitive, compensated nanofiber trap" New J. Phys. 14 (2012).

## Use the pre-implemented nanofiber field calculations

## Creating the beams and trapping scheme

There are many different designs that can be used for trapping atoms with evanescent fields.
We have then to specify how many beams we want to use, at which wavelengths and powers.

If the structures are pre-implemented in `nanotrappy`, the right modes will then be computed.
If not, a folder containing the modes will have to be specified, consult the [mode formatting][1] section to see how the modes have to be saved in order to be used by `nanotrappy`.

Beam objects are the basic building blocks of your trap in `nanotrappy`.

```python linenums="2"
blue_beam = Nb.Beam(1064e-9,"f",25*mW)
red_beam = Nb.Beam(780e-9,"f",7*mW)
```

Here we define two beams, blue- and red-detuned from the Cesium D2 line respectively.
"f" means the beam is propagating forward, along the propagation axis of the waveguide that will be specified later. "b" stands for backwards.
This is important as chiral effects can arise when coupling tightly focused light to quantum emitters for example.

These beams are bundled into a trap object, where a propagation axis is specified.

```python linenums="4"
trap = Nt.Trap_beams(blue_beam,red_beam,propagation_axis="Z")
```

## Specifying the atom we want to trap

This part is based on the [Alkali Rydberg Calculator][2] library. Originally conceived for studying Rydberg atoms, we have included the hyperfine levels of such Alkali.

```python linenums="5"
syst = Na.atomicsystem(Caesium(), Na.atomiclevel(6,S,1/2),f=4)
```

An alkali atom has to be chosen, an atomic level from the fine structure (here the ground state of Cesium 6S1/2) and the F of the hyperfine level studied.
Indeed the dipole force felt by the atoms depends on the hyperfine level they are in.

### Optional : Set surfaces for the Van der Waals interactions

See [the dedicated page][3] for more information. The nanofiber is a cylinder with axis of symmetry the propagation axis.
We define then a cylinder centered at the origin of the grid, with 250 nm radius and with "Z" revolution axis.

```python linenums="6"
surface = vdw.Cylinder((0,0),250e-9,"Z")
```

##Define the Simulation object

The simulation object will then be a bundle of everything needed to compute the trapping potentials.
The right modes corresponding to the trapping scheme, the atomic level and the surface and material (for CP interactions).
This class will be useful mostly for keeping the results of the simulation and visualisation purposes further on.

```python linenums="7"
Simul = Ns.Simulation(syst,Nm.SiO2(),trap,datafolder,surface)
```

##Run the simulation in 2 dimensions

```python linenums="8"
trap2D = Simul.compute_potential("XY",0)
```

Careful that the simulation in 2D can be time consuming as the potential is computed point by point on the given grid.
As an example, for a 100x100 grid, times of the order of 30s are expected.

##Visualizing the results

The [viz][5] submodules has a few methods to visualise and analyze your results.
The easiest is to plot the 2D potential with power sliders as follows :

```python linenums="9"
fig, ax, slider_ax = Nv.plot_trap(Simul,"XY")
```

It is important to return the image and sliders as this will ensure the sliders do not get stuck.

[![Nanofiber plot][6]{: style="height:600px;width:600px"}][6]

We see here 2 trapping sites, one on each side of the fiber (around 230 nm from the surface each).
We can also look at a cut in 1D for easier analysis.

If you are satisfied, you can save your simulation with `Simul.save()`.
This will create a new folder in your mode folder and save a JSON file with the list of parameters of the simulation as well as a NPY with the potentials (not summed).

When you run a simulation `nanotrappy` looks in the results folder if a simulation with the same parameters has been already saved. If so, it just opens it without running the simulation again.

A 1D simulation can also be run to look at how the different mf states are affected by the differential vector shift coming from the local circular polarisation of the light acting as a fictitious magnetic field.

```python linenums="10"
trap1D = Simul.compute_potential_1D('X',0,0)
fig, ax, slider_ax = Nv.plot_trap1D(Simul,"X",[-4,-3,-2,-1,0,1,2,3,4])
Simul.save()
```

[![1D trap around a nanofiber][7]][7]

[1]: modeformatting.md
[2]: https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/
[3]: casimirpolder.md
[4]: troubleshooting.md
[5]: reference/viz.md
[6]: ./images/nanofiber_plot.png
[7]: ./images/nanofiber_1D_splitting.png
