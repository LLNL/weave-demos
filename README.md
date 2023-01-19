Overview
========

This is a repository for demos of WEAVE functionality. Demo subdirectories contain more information (including instructions) for each.


Ball Bounce
-----------

A demonstration of Sina and Maestro being used together to perform and analyze runs of a toy code that simulates a ball bouncing around in a 3D box. Maestro is used to launch suites of runs, each sharing a ball starting position but having a randomized velocity vector. The toy code outputs DSV, which is converted to Sina's format, ingested, and can then be explored in the included Jupyter notebooks.

Once the data is explorable, Trata samples parameter points that are used by IBIS to infer parameter uncertainties in the bouncing ball simulations using IBIS' default Markov chain Monte Carlo (MCMC) method. 