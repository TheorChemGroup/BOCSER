### Bayesian optimization for conformational search

Here you can find a source code for Bayesian optimization with Gaussian Process Conformational Search method. It allows one to perform conformational search on any level of theory (available in ORCA) via Bayessian Optimization. 

## Installation

1. Clone repo
```
git clone https://github.com/TheorChemGroup/BOCSER.git
```
2. Install dependecies
```
conda create -c bocser python=3.10 trieste rdkit pyyaml
conda activate bocser
``` 
Also method requiers installed ORCA to perform calculations and Slurm workload manager to manage them

## Usage

Conformational search configures from config.yaml file, that should be placed nearby conf_search.py. It has several properties:

* exp_name : str - name of experiment. It should be prefix for all files, associated with current evaluation of method
* mol_file_name : str - name of .mol file of molecule for conformational search
* charge : int - charge of a molecule
* spin_multiplicity : int - spin multiplicity of molecule 
* orca_exec_command : str - path to the orca binary file with which ORCA will be executed
* num_of_procs : str - number of procs for ORCA calculations
* broken_struct_energy : int - default energy, that will be returned if optimiztion didn't finished successful
* bond_length_threshold : float - minimum bond length to accept structure as correct
* rolling_window_size : int - number of steps to take into account when calculating termination criteria
* rolling_std_threshold : float - maximum standard deviation of acquisition function values in rolling window to stop the search
* rolling_mean_threshold : float - maximum acquisition function values mean in rolling window to stop the search
* num_initial_points : int - number of randomly selecting initial points for method
* max_steps : int - maximum number of steps of Bayesian optimization
* load_ensemble : str - path to .xyz file with already calculated ensemble if you want to refine ensemble
* acquisition_function : str - defines acquisition function to use. If "ei" is passed - ExpectedImprovement is used, if "evm" - our acquisition function, named Explorational Variance Minimizer

