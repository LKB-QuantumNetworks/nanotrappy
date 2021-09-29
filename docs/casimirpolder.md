---
template: main.html
title: On Casimir-Polder interactions and surfaces implemented in nanotrappy
---

Van der Waals interactions (also called Casimir-Polder forces) tend to bring the atoms closer to the surface. They depend on the atom, its atomic level and on the material they are approaching.

Their action is only on very short distances (usually on the order of 100nm) but when looking at atoms close to surface with high refractive index, they sometimes have to be taken into account.
Our approach is to set surfaces that will be added to the simulation.
For now only CylindricalSurfaces and PlaneSurfaces have been implemented but together they cover most of the waveguides of interest.

Distance from these surfaces are computed and the CP potential is set to its first order approximation : U<sub>CP</sub> = - C<sub>3</sub>/d<sup>3</sup>.

The C<sub>3</sub> coefficient depends on both atom and material of the structure. A routine in nanotrappy allows computation of C<sub>3</sub> for {atom, material} pairs that are not yet implemented.
See the main text of the [companion article][1] for more information.

[1]: https://arxiv.org/
