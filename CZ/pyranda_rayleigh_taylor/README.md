# Rayleigh Taylor Modeling with pyranda

pyranda is an mpi parallel high order finited difference solver for arbitrary hyperbolic PDE systems [pyranda](https://github.com/LLNL/pyranda).  This set of examples works through simulations of the Rayleigh-Taylor problem,
which is an instability between interfaces of two fluids acting under non-impulsive accelerations (see Richtmyer-Meshkov for the impulsively accelerated version).

## Nominal behaviors

The first set of models demonstrate the phenomena and explore the effects of different fluid densities via the non-dimensional Atwood number that expresses the light/heavy fluid
density ratios.  There are a variety of regimes that can be probed, but we'll focus on the configuration of multimode initial interface perturbations with miscible fluids.  In this
setup, the mixing width grows with a form of ~ alpha*A*g*t, where A = atwood number, g = accleration (often gravity), t = time, and alpha is a ~constant factor.  There are some caveats,
such as low wavenumber content in the initial condition (or large wavelength) tends to dominate and grow faster.  Thus this scaling law breaks down a bit in the presence of a lot of
low wavenumber content.  The intial study will show some of these effects with a caveat that doing this in 2D can't quite give the right answer owing to the significant 3D effects in
such problems.

### Use Maestro To Run Experiments with pyranda

The `rayleigh_taylor.yaml` file is a Maestro study specification that can run parameter sweeps of our pyranda rayleigh taylor model, allowing variance of the Atwood number and
the seed fed into a randomized velocity perturbation applied to the fluid interface.

## UQ

TBD
