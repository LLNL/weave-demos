# Encore Optimization using Scipy

This is a generic tool using Encore and Scipy to find the minimum of a function/simulation/simulations.

## How to run

1. Run `setup.sh` in the top directory to create a virtual environment with all necessary dependencies and install the jupyter kernel.

2. Run `source cz_tutorials_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it) and `cd` back into this directory.

3. Follow the steps in the sections below.

## Structure

### Initial Guess and Bounds YAML file

The `initial_guess_and_bounds.yml` file contains the parameters, their initial guesses, and their bounds for the optimization study. It is in a standard yaml format. This is passed to the Encore Template using the variable `INITIAL_GUESS_AND_BOUNDS_PATH`.

```yaml
env:
    variables:
      INITIAL_GUESS_AND_BOUNDS_PATH: /g/g20/moreno45/Projects/WEAVE/weave_demos/CZ/encore_optimization/initial_guess_and_bounds_single_parameter.yml
```

`initial_guess_and_bounds.yml`
```yaml
x:
  initial_guess: 9.0
  bounds: [-10,10]
y:
  initial_guess: -4.0
  bounds: [-5,5]
```

### Two additional steps needed in Encore Template

There are two additional steps needed in the Encore Template for the optimization study to work.

#### sim_end_res

The step `sim_end_res` is the step that does the **FINAL** post-processing in order to acquire the quantity of interest (QoI) you are trying to minimize. This QoI or **simulation end result (sim_end_res)** can be anything you want but it **MUST** be outputted to a csv file named `sim_end_res.csv` with only one column titled 'sim_end_res' and its corresponding value to the current `sim_end_res` directory.

```python
with open('sim_end_res.csv', 'w') as stream:
    stream.write('sim_end_res') # Column header is always simulation end result 'sim_end_res'
    stream.write('\n') # Need a new line to separate header from value
    stream.write(f'{z}') # Value for 'sim_end_res'
```

`sim_end_res.csv`
```
sim_end_res
0.003439124634386416
```

#### decide

The step `decide` is the step that actually does the optimiazation by calling the `check.py` script. This script uses the `scipy.optimize.minimize()` method which can be used with a single parameter or multiple parameters (this is taken care of behind the scenes, you just need to add your parameters to `initial_guess_and_bounds.yml`). The current minimization method that is being used is `method='Nelder-Mead'` but that can be changed in `check.py`. **WARNING: Changing anything else in the script will most likely break the script. Everything else is automated so there should be no need to change it.** The `check.py` script outputs `overall_file.csv` (file containing all of our iteration values) and `next_guess.csv` (file containing the next iteration guess) at the top level of the encore study.

`overall_file.csv`
```csv
iteration,x,y,sim_end_res
1,9.0,-4.0,97.0
2,9.45,-4.0,105.30249999999998
3,9.0,-4.2,98.64
4,8.549999999999999,-4.199999999999999,90.74249999999998
5,8.099999999999998,-4.299999999999999,84.09999999999997
6,8.099999999999998,-4.099999999999999,82.41999999999996
7,7.649999999999999,-4.049999999999999,74.92499999999997
8,6.7499999999999964,-4.349999999999998,64.48499999999993
9,5.624999999999993,-4.524999999999997,52.11624999999989
10,5.174999999999994,-4.274999999999997,45.05624999999991
```

`next_guess.csv`
```
x,y
0.02264385223389618,-0.054096031188950414
```

Instead of wrapping all of the simulations within the `scipy.optimize.minimize()` method which calls a function `f`, we use the function `f` to read the `overall_file.csv` file with all of our iteration values in order for `check.py` to be generic. Once the function `f` doesn't see a matching `iteration` with its corresponding `sim_end_res`, it outputs a `next_guess.csv` which gets read in for the next iteration by the custom `pgen.py`

The Encore Optimization study will stop once the `scipy.optimize.minimize()` method reaches a tolerance defined by the `method` used. The `check.py` script will output a yaml file `encore.yaml` which gets read by Encore to decide if another iteration is needed. This yaml file has a simple key value pair of `'is_done: True` for converged or `'is_done: False` for not converged. If the study converged, it will output a `converged_results.txt` file with `scipy.optimize.minimize()` output at the top level of the encore study.

**The step must have this command:**

```
python $(SPECROOT)/check.py $(sim_end_res.workspace)/sim_end_res.csv $(OUTPUT_PATH) $(INITIAL_GUESS_AND_BOUNDS_PATH)
```

## Running a study

You can run a study with the commands below. Choose a low `--limit` of iterations just to make sure everything is running properly.

### Single Parameter

`encore run encore_optimizer_single_parameter.yml --limit 3 --pgen pgen.py`

### Multiple Parameters

`encore run encore_optimizer_multiple_parameters.yml --limit 3 --pgen pgen.py`

## Continuing a Study

What happens if the `--limit` was not enough or the HPCs shut down over the weekend? Luckily there is a built in feature that allows the user to continue the study. This works because we didn't wrap our simulations within `scipy.optimize.minimize()` and instead we are just reading in `overall_file.csv`. The only additional variables needed in the Encore Optimization template are `CONTINUE_PATH` (the path of the `overall_file.csv` file) and `NEXT_GUESS_PATH` (the path of the `next_guess.csv` file). **BOTH** of these paths are needed for a continuation of a study.

```yaml
env:
    variables:
      INITIAL_GUESS_AND_BOUNDS_PATH: /g/g20/moreno45/Projects/WEAVE/weave_demos/CZ/encore_optimization/initial_guess_and_bounds_multiple_parameters.yml
      CONTINUE_PATH: /g/g20/moreno45/Projects/WEAVE/weave_demos/CZ/encore_optimization/overall_file.csv
      NEXT_GUESS_PATH: /g/g20/moreno45/Projects/WEAVE/weave_demos/CZ/encore_optimization/next_guess.csv
```

The `decide` step also needs to be updated so that `check.py` gets passed the `CONTINUE_PATH`.

```
python $(SPECROOT)/check.py $(sim_end_res.workspace)/sim_end_res.csv $(OUTPUT_PATH) $(INITIAL_GUESS_AND_BOUNDS_PATH) $(CONTINUE_PATH)
```

If only some of your simulations completed when the Encore Optimization study crashed, you can just finish running them in that dir and run the slurm scripts for the `sim_end_res` and `decide` steps for that specific iteration. This will allow you to get the latest and greatest `overall_file.csv` and `next_guess.csv` files and pass those in to the variables mentioned above.

If the directory for `decide` was not created, you can run the command below **from this repo** to get the `overall_file.csv` and `next_guess.csv` files. Note that the second argument `<path/to/encore_optimization_DATETIME/encore_optimization_DATETIME_10>` **should not** have the end slash `/` and should end on the iteration #.

```
python check.py <path/to/encore_optimization_DATETIME/encore_optimization_DATETIME_10>/sim_end_res/sim_end_res.csv  <path/to/encore_optimization_DATETIME/encore_optimization_DATETIME_10> initial_guess_and_bounds_multiple_parameters.yml
```

### Multiple Parameters Continuation

`encore run encore_optimizer_multiple_parameters_continuation.yml --limit 3 --pgen pgen.py`
