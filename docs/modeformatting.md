---
template: overrides/main.html
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

[1]: ./images/mode_formatting.png
