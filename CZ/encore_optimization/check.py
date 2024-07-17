import sys
import os
from scipy.optimize import minimize, minimize_scalar
import pandas as pd
import yaml
import shutil

sim_end_res_path = sys.argv[1]
output_path = sys.argv[2]
initial_guess_and_bounds_path = sys.argv[3]
current_iter = int(output_path.split('_')[-1])
try:
    continue_path = sys.argv[4]
    original_iter = current_iter
    df_continue= pd.read_csv(continue_path)
    current_iter = int(df_continue.iloc[-1]['iteration']) + original_iter
except:
    continue_path = None
    original_iter = None

sim_dict_overall = {}
sim_dict_current = {}

print(f'Output path: {output_path}')
print(f'Current iteration: {current_iter}')

######################################################
##### Check sim_end_res/ dir for sim_end_res.csv #####
######################################################
df_current = pd.read_csv(sim_end_res_path)
sim_dict_current = df_current.to_dict('list')

parameter_path = os.path.join(output_path,'meta','parameters.yaml')
with open(parameter_path, "r") as stream:
    parameters = yaml.load(stream ,Loader=yaml.Loader) # Can't use safeload since it has ordered dict constructor
    for key, val in parameters.items():
        for param_key, param_val in val['labels'].items():
            param_name = param_key.replace('$(','').replace('.label)','').lower()
            param_float = [float(param_val.split('.',1)[1])]  # has to be in a list to later append
            sim_dict_current[param_name] = param_float

sim_dict_current['iteration'] = [current_iter]
sim_dict_overall = sim_dict_current

print(f'Current simulation/ dictionary:\n{sim_dict_current}\n')

############################################################################
##### Check ../overall_file.csv for all the different iteration values #####
############################################################################
if continue_path is not None and original_iter==1:
    sim_dict_overall = df_continue.to_dict('list')
    for k,v in sim_dict_overall.items():
        sim_dict_overall[k].extend(sim_dict_current[k]) # adding current simulation values to overall values
elif current_iter != 1: # Not the first iteration
    file_path = os.path.join(output_path, '../', 'overall_file.csv')
    df_overall = pd.read_csv(file_path)
    sim_dict_overall = df_overall.to_dict('list')
    for k,v in sim_dict_overall.items():
        sim_dict_overall[k].extend(sim_dict_current[k]) # adding current simulation values to overall values

df = pd.DataFrame(data=sim_dict_overall)

# Rearrange so 'iteration' is first column
cols = df.columns.tolist()
# First iteration adds iteration column after values
if cols[0] != 'iteration':
    cols = [cols[-1]] + cols[1:-1] + [cols[0]]
df = df[cols]

# Copy dataframe to current iteration dir and overall encore study dir
current_iter_file = os.path.join(output_path,"file.csv")
df.to_csv(current_iter_file, sep=',',index=False)
shutil.copy(current_iter_file, os.path.join(output_path, '../', 'overall_file.csv'))

print(f'Overall simulations dictionary:\n{sim_dict_overall}\n')
print(f'Overall simulations dataframe:\n{df}\n')

###############################################################################################
##### "function" that optimizer calls to iterate, just grabs simulation end result values #####
###############################################################################################

def f(x,*args):
    iteration = args[0]
    internal_iter = args[1]['internal_iter']

    print('-------------------- New x --------------------')
    print(f'Current Iteration: {iteration}')
    print(f'Internal Iteration: {internal_iter}')
    print(f'Internal Iteration parameters:\n{x}')
    print(f'Internal Iteration search:\n{df.loc[df["iteration"]==internal_iter]}')

    sim_end_res = df.loc[df['iteration']==internal_iter]['sim_end_res'].values

    # Internal iteration still in dataframe
    if sim_end_res.size>0 and int(internal_iter) <= int(iteration):
        print(f'Simulation values found for {internal_iter}:\n{sim_end_res}')
        args[1]['internal_iter'] += 1 # Update internal iteration but need to use 'args' to update
        return sim_end_res[0] # Return simulation end result

    # Internal iteration not in dataframe
    else:
        print(f'No matching iteration found for {internal_iter}')
        print(f'Next guesses will be:\n{x}')

        with open(os.path.join(output_path, '../','next_guess.csv'), 'w') as stream:
            stream.write(','.join(str(header) for header in cols[1:-1])) # don't write iteration header or sim_end_res
            stream.write('\n')
            stream.write(','.join(str(val) for val in x))
        sys.exit()

#########################################
##### Single or Multiple Parameters #####
#########################################
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

# Getting initial guess and bounds
x0=[]
bounds=[]
with open(initial_guess_and_bounds_path, "r") as stream:
    initial_guess = yaml.safe_load(stream)
    # Don't grab 'iteration' or 'sim_end_res' columns
    # Initial guesses and bounds have to be in the same order as dictionaries
    for col in cols[1:-1]:
        x0.append(initial_guess[col]['initial_guess'])
        bounds.append(initial_guess[col]['bounds'])

print(f'Initial guess and bounds:\n{initial_guess}\n')
print(f'Initial guess array:\n{x0}\n')
print(f'Initial bounds matrix:\n{bounds}\n')

results = {'is_done': False}
with open(os.path.join(output_path,'encore.yaml'), 'w') as _file:
    yaml.dump(results, _file)

internal_iter = {'internal_iter':1}
res = minimize(f,x0,bounds=bounds,
               args=(current_iter, internal_iter),
               method='Nelder-Mead')

# Will only get here if converged
print(f'CONVERGED!\n{res}')
with open(os.path.join(output_path,'../','converged_results.txt'), 'w') as stream:
    stream.write(str(res))

results = {'is_done': True}
with open(os.path.join(output_path,'encore.yaml'), 'w') as _file:
    yaml.dump(results, _file)


