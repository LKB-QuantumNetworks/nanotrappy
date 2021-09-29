---
template: main.html
title: Setting up a simulation
---

# Parallelizing your calculations

For precise simulations on big grids, the computation can take a few minutes, especially if designs with counterpropagating beams are used.
In order to speed up the optimisation process, we implemented a native way for dispatching the computation on various cores of the processor.
This is done thanks to the `ParallelSimulator` class, which takes advantage of the Python `multiprocessing` package.

After setting the simulation and before running `Simul.compute`, you can set up a `ParallelSimulator` that will handle the parallelization of the computation:

```python linenums="1"
   Simul = Simulation(syst, SiO2(), trap, datafolder,surface)
   Simul.simulator = ParallelSimulator(max_workers=None)
```

`max_workers` specify the number of maximum cores that will be used for the simulation. When set to None, all the possible cores will be used.
By default an object `SequentialSimulator` is created, avoiding any issue that could arise when using `multiprocessing`.

On Linux, the htop command allows to see the usage of the microprocessor cores.
