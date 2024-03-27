# Overview

This is a repository for demos of WEAVE functionality. Demo subdirectories contain more information (including instructions) for each.

## Ball Bounce

A demonstration of Sina and Maestro being used together to perform and analyze runs of a toy code that simulates a ball bouncing around in a 3D box. Maestro is used to launch suites of runs, each sharing a ball starting position but having a randomized velocity vector. The toy code outputs DSV, which is converted to Sina's format, ingested, and can then be explored in the included Jupyter notebooks.

## Ball Bounce VVUQ

An extension of the Ball Bounce demo that generates ensembles of runs in order to perform Verification, Validation, and Uncertainty & Quantification. Trata also samples parameter points that are used by IBIS to infer parameter uncertainties in the bouncing ball simulations using IBIS' default Markov chain Monte Carlo (MCMC) method.

## Ball Bounce LSTM

An extension of the Ball Bounce demo that generates ensembles of runs in order to train a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) to predict the transient path of the bouncing ball.

## Encore Optimization

An Encore Workflow that uses SciPy to optimize a single Quantity of Interest with a single parameter or multiple parameters. The workflow has been generalized such that a user can pass in their own simulation.