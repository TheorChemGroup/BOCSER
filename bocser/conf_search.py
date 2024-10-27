from transform_kernel import TransformKernel
from coef_from_grid import pes, pes_tf, pes_tf_grad
from calc import (
    calc_energy, 
    load_last_optimized_structure_xyz_block,
    change_dihedrals,
    parse_points_from_trj,
    load_params_from_config,
    increase_structure_id
)
from coef_calc import CoefCalculator
from db_connector import LocalConnector
from ensemble_processor import EnsembleProcessor 
from evm import ExplorationalVarianceMinimizer
from dbscan import DBSCAN
from default_vals import ConfSearchConfig

from dataclasses import fields
import trieste
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.data import Dataset
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from scipy.special import erf
import sys
import os
import yaml
import json
import argparse

from trieste.acquisition.function import ExpectedImprovement

from rdkit import Chem

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

MOL_FILE_NAME = None
NORM_ENERGY = 0.

DIHEDRAL_IDS = []

CUR_ADD_POINTS = []

global_degrees = []

structures_path = ""
exp_name = ""

ASKED_POINTS = []

model_chk = None
current_minima = 1e9
acq_vals_log = []

LAST_OPT_OK = True

MINIMA = []

def degrees_to_potentials(
    degrees : np.ndarray,
    mean_func_coefs : np.ndarray
) -> np.ndarray:
    return [
        [pes(np.array([[degree[i]]]), *coefs[i])[0] for i in range(len(degree))]
        for degree in degrees
    ]  

# defines a functions, that will convert args for mean_function
def parse_args_to_mean_func(inp):
    """
    Convert input data from [inp_dim, 7] to [7, inp_dim]
    """
    return tf.transpose(inp)

def calc(dihedrals : list[float]) -> float:
    """
        Perofrms calculating of energy with current dihedral angels
    """
    
    def dump_status_hook(
        dumping_value : bool,
        filename : str = exp_name+"_last_opt_status.json"
    ) -> None:
        with open(filename, 'w') as file:
            json.dump({
                "LAST_OPT_OK" : dumping_value
            }, file)

    global LAST_OPT_OK

    if model_chk:
        print(f"Checkpoint is not null, calculating previous acq. func. max!")
        dihedrals_tf = tf.constant(dihedrals, dtype=tf.float64)
        if len(dihedrals_tf.shape) == 1:
            dihedrals_tf = tf.reshape(dihedrals_tf, [1, dihedrals_tf.shape[0]])
        print(f"Cur dihedrals_tf: {dihedrals_tf}")
        print(f"Current minima: {current_minima}")
        mean, variance = model_chk.predict_f(dihedrals_tf)
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        tau = current_minima + 3.
        acq_val = normal.cdf(tau) * (((tau - mean)**2) * (1 - normal.cdf(tau)) + variance) + tf.sqrt(variance) * normal.prob(tau) *\
                (tau - mean) * (1 - 2*normal.cdf(tau)) - variance * (normal.prob(tau)**2)
        print(f"Previous acq. val: {acq_val}")
        acq_vals_log.append(acq_val.numpy().flatten()[0])        

    if tf.is_tensor(dihedrals):
        dihedrals = list(dihedrals.numpy())

    ASKED_POINTS.append(dihedrals)
    
    print(f"Point: {dihedrals}")

    #Pre-opt
    print('Optimizing constrained struct')
    en, preopt_status = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, True, constrained_opt=True)
    LAST_OPT_OK = preopt_status
    print(f"Status of preopt: {preopt_status}; LAST_OPT_OK: {LAST_OPT_OK}")
    if not preopt_status:
        dump_status_hook(dumping_value=LAST_OPT_OK)
        skipped_structure_id = increase_structure_id()
        print(f"Preopt finished with error! Structure with number {skipped_structure_id} will be skipped!")
        return en + np.random.randn()
    print('Optimized!\nLoading xyz from preopt')
    xyz_from_constrained = load_last_optimized_structure_xyz_block(MOL_FILE_NAME)
    print('Loaded!\nFull opt')
    en, opt_status = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, True, force_xyz_block=xyz_from_constrained)
    LAST_OPT_OK = opt_status
    print(f"Status of opt: {opt_status}; LAST_OPT_OK: {LAST_OPT_OK}")
    print(f'Optimized! En = {en}')
    dump_status_hook(dumping_value=LAST_OPT_OK)

    if not opt_status:
        skipped_structure_id = increase_structure_id() 
        print(f"Opt finished with error! Structure with number {skipped_structure_id} will be skipped!")

    return en + ((not opt_status) * np.random.randn())

def max_comp_dist(x1, x2, period : float = 2 * np.pi):
    """
        Returns dist between two points:
        d(x1, x2) = max(min(|x_1 - x_2|, T - |x_1 - x_2|))), 
        where T is period
    """
    
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)
    if not isinstance(x2, np.ndarray):
        x2 = np.array(x2)
        
    return np.max(np.min((np.abs(x1 - x2), period - np.abs(x1 - x2)), axis=0))

# defines a function that will be predicted
# cur - input data 'tensor' [n, inp_dim], n - num of points, inp_dim - num of dimensions
def func(cur): 
    return tf.map_fn(fn = lambda x : np.array([calc(x)]), elems = cur)

def upd_points(dataset : Dataset, model : gpflow.models.gpr.GPR) -> tuple[Dataset, gpflow.models.gpr.GPR]:
    """
        update dataset and model from CUR_ADD_POINTS
    """

    degrees, energies = [], []
    for cur in CUR_ADD_POINTS:
        d, e = zip(*cur)
        degrees.extend(d)
        energies.extend(e)
    dataset += Dataset(tf.constant(list(degrees), dtype="double"), tf.constant(list(energies), dtype="double").reshape(len(energies), 1))
    model.update(dataset)
    model.optimize(dataset)

    return dataset, model

def upd_dataset_from_trj(
    trj_filename : str, 
    dataset : Dataset
) -> Dataset:
    """
        Return dataset that consists of old points
        add points from trj
    """
    print(f"Input dataset is: {dataset}") 
    parsed_data, last_point = parse_points_from_trj(
        trj_file_name=trj_filename, 
        dihedrals=DIHEDRAL_IDS, 
        norm_en=NORM_ENERGY, 
        save_structs=True, 
        structures_path=structures_path, 
        return_minima=True,
    )

    with open(f"{exp_name}_minima/{len(MINIMA)}.xyz", "w") as minima_xyz_writer:
        minima_xyz_writer.write(last_point["xyz_block"])

    MINIMA.append((last_point["coords"], last_point["rel_en"]))

    print(f"Parsed data: {parsed_data}")
    degrees, energies = zip(*parsed_data)
    print(f"Degrees: {degrees}\nEnergies: {energies}")

    global_degrees.extend(degrees) 

    add_part = Dataset(tf.constant(list(degrees), dtype="double"), tf.constant(list(energies), dtype="double").reshape(len(degrees), 1))

    if not dataset:
        return add_part
    else:
        return dataset + add_part

def erase_last_from_dataset(dataset : Dataset, n : int = 1):
    """
        Deletes last n points from trj
    """
    
    query_points = tf.slice(dataset.query_points, [0, 0], [dataset.query_points.shape[0] - n, dataset.query_points.shape[1]])
    observations = tf.slice(dataset.observations, [0, 0], [dataset.observations.shape[0] - n, dataset.observations.shape[1]])

    return Dataset(query_points, observations)

#TODO: Rewrite in tf way
class PotentialFunction():
    def __init__(self, mean_func_coefs) -> None:
        self.mean_func_coefs = mean_func_coefs

    @tf.function
    def __call__(self, X : tf.Tensor) -> tf.Tensor:
        return tf.stack(
                    [
                        pes_tf(X[:, dim], *self.mean_func_coefs[dim]) for dim in range(len(self.mean_func_coefs))
                    ],
                    axis=1
                )
    @tf.function
    def grad(self, X : tf.Tensor) -> tf.Tensor:
        return tf.stack(
                    [
                        pes_tf_grad(X[:, dim], *self.mean_func_coefs[dim]) for dim in range(len(self.mean_func_coefs))
                    ],
                    axis=1
                )

parser = argparse.ArgumentParser(
    prog="bo_confsearch",
    description="Bayesian optimization for conformational search",
)

parser.add_argument('--config', default='config.yaml')

args = parser.parse_args()

print(f"Reading config {args.config}")

raw_config = {}

try:
    with open(args.config, 'r') as file:
        raw_config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"No config file {args.config}!\nFinishing!")
    exit(0)
except Exception:
    print("Something went wrong!\nFinishing!")
    exit(0)

config = ConfSearchConfig(**raw_config)

MOL_FILE_NAME = config.mol_file_name
structures_path = config.exp_name + "/"
exp_name = config.exp_name

if not os.path.exists(structures_path):
    os.makedirs(structures_path)
    os.makedirs(structures_path[:-1]+"_minima"+"/")

if config.acquisition_function not in {"ei", "evm"}:
    print(f"Acquisition function should be one of the following: 'ei', 'evm', got {config.acquisition_function}; Continue with default: 'evm'")
    config.acquisition_function = "evm"

print(f"Performing conf. search with config: {config}")

load_params_from_config({field.name : getattr(config, field.name) for field in fields(config)}) # TODO: rewrite in better way

print("Coef calculator creatring")

coef_matrix = CoefCalculator(
    mol=Chem.RemoveHs(Chem.MolFromMolFile(MOL_FILE_NAME)),
    config=config, 
    dir_for_inps=f"{exp_name}_scans/", 
    db_connector=LocalConnector('dihedral_logs.db')
).coef_matrix()

print("Coef calculator created!")

mean_func_coefs = []

for ids, coefs in coef_matrix:
    DIHEDRAL_IDS.append(ids)
    mean_func_coefs.append(coefs)

print("Dihedral ids", DIHEDRAL_IDS)
print("Mean func coefs", mean_func_coefs)

search_dim = len(DIHEDRAL_IDS)

print("Cur search dim is", search_dim)

amps = np.array([
    np.abs(mean_func_coefs[i][:3]).sum() for i in range(len(mean_func_coefs))
])

potential_func = PotentialFunction(mean_func_coefs)

kernel = gpflow.kernels.White(0.001) + gpflow.kernels.Periodic(gpflow.kernels.RBF(variance=0.07, lengthscales=0.005, active_dims=[i for i in range(search_dim)]), period=[2*np.pi for _ in range(search_dim)]) + TransformKernel(potential_func, gpflow.kernels.RBF(variance=0.12, lengthscales=0.005, active_dims=[i for i in range(search_dim)])) # ls 0.005 var 0.3 -> 0.15

kernel.kernels[1].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64))
kernel.kernels[2].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64))

search_space = Box([0. for _ in range(search_dim)], [2 * np.pi for _ in range(search_dim)])  # define the search space directly

#Calc normalizing energy
#in kcal/mol!

NORM_ENERGY, _ = calc_energy(MOL_FILE_NAME, dihedrals=[], norm_energy=0.)#-367398.19960427243

print(f"Norm energy: {NORM_ENERGY}")

observer = trieste.objectives.utils.mk_observer(func) # defines a observer of our 'func'

# calculating initial points
dataset = None

if config.load_ensemble:
    print("Loading init points from given ensemble!")
    dataset = Dataset(
        *EnsembleProcessor(
            config.load_ensemble,
            dihedral_idxs=DIHEDRAL_IDS
        ).normalize_energy(NORM_ENERGY).get_tf_data()
    )
    print(f"Init dataset collected!\n{dataset}")
else:

    for idx in range(config.num_initial_points):
        initial_query_points = search_space.sample_sobol(1)
        observed_point = observer(initial_query_points)
        if not LAST_OPT_OK:
            print(f"Optimization didn't finished well. Continue only with broken_struct_energy in required point: {observed_point}")
            dataset = observed_point if not dataset else dataset + observed_point
        else:
            dataset = upd_dataset_from_trj(f"{MOL_FILE_NAME[:-4]}_trj.xyz", dataset)
    
        print(f"Initial dataset observed! {config.num_initial_points} minima observed, total {dataset.query_points.shape[0]} points has been collected!")

gpr = gpflow.models.GPR(
    dataset.astuple(), 
    kernel
)

gpflow.set_trainable(gpr.likelihood, False)
gpflow.set_trainable(gpr.kernel.kernels[0].variance, False)
gpflow.set_trainable(gpr.kernel.kernels[1].period, False)
model = GaussianProcessRegression(gpr, num_kernel_samples=100)

# Starting Bayesian optimization
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

print(f"Initital data:\n{dataset}")

model.optimize(dataset)

model_chk = gpflow.utilities.deepcopy(model.model)
current_minima = tf.reduce_min(dataset.observations).numpy()

rule = None

match config.acquisition_function:
    case "evm":
        print("Continue with Explorational Variance Minimizer acquisition function!")
        rule = EfficientGlobalOptimization(ExplorationalVarianceMinimizer(threshold=3))
    case "ei": 
        print("Continue with ExpectedImprovement acquisition function!")
        rule = EfficientGlobalOptimization(ExpectedImprovement())
    case _:
        raise ValueError(f"Unknown acquisition function {config.acquisition_function}")

#rule = EfficientGlobalOptimization(ExpectedImprovement())

deepest_minima = []

early_termination_flag = False

print(f"MINIMA: {MINIMA}")

for step in range(1, config.max_steps+1):
    print(f"Previous last_opt_ok: {LAST_OPT_OK}")
    print(f"Step number {step}")

    try:
        result = bo.optimize(1, dataset, model, rule, fit_initial_model=False)
        print(f"Optimization step {step} succeed!")
    except Exception:
        print("Optimization failed")
        print(result.astuple()[1][-1].dataset)
    
    print(f"After step: {LAST_OPT_OK}")

    last_opt_status = None
    with open(exp_name+"_last_opt_status.json", "r") as file:
        last_opt_status = json.load(file)
    print(last_opt_status)

    dataset = result.try_get_final_dataset()
    model = result.try_get_final_model()
    print(f"Last asked point was {ASKED_POINTS[-1]}")

    deepest_minima.append(tf.reduce_min(dataset.observations).numpy())    
    
    logs = {
        'acq_vals' : acq_vals_log,
        'deepest_minima' : deepest_minima,
        'norm_en' : NORM_ENERGY
    }

    with open(f"{exp_name}_logs.json", 'w') as file:
        json.dump(logs, file)

    print(f"Eta is {rule._acquisition_function._eta}")    
    if LAST_OPT_OK:
        dataset = erase_last_from_dataset(dataset, 1)
        dataset = upd_dataset_from_trj(f"{MOL_FILE_NAME[:-4]}_trj.xyz", dataset)
    else:
        print(f"Last optimization finished with error, skipping trj parsing!")
    model.update(dataset)
    model.optimize(dataset)

    print("Updating model checkpoint!")
    model_chk = gpflow.utilities.deepcopy(model.model)
    current_minima = rule._acquisition_function._eta.numpy()[0]#tf.reduce_min(dataset.observations).numpy()
    print("Updated!")

    print(f"Step {step} complited! Current dataset is:\n{dataset}")
    
    with open(f"{exp_name}_all_minima.json", "w") as json_minima_writer:
        json.dump(MINIMA, json_minima_writer)
    
    if step < config.rolling_window_size:
        continue
    
    print(f"Checking termination criterion!")
    print(f"Acq vals in window: {logs['acq_vals'][max(0, step-config.rolling_window_size):step]}")
     
    rolling_mean = np.mean(logs['acq_vals'][max(0, step-config.rolling_window_size):step])
    rolling_std = np.std(logs['acq_vals'][max(0, step-config.rolling_window_size):step])

    print(f"After step {step}:")
    print(f"Current rolling mean of acqusition function maximum is: {rolling_mean}, threshold is {config.rolling_mean_threshold}")
    print(f"Current rolling std of acqusition function maximum is: {rolling_std}, threshold is {config.rolling_std_threshold}")
    if step >= config.rolling_window_size and rolling_std < config.rolling_std_threshold and rolling_mean < config.rolling_mean_threshold:
        print(f"Termination criterion reached on step {step}! Terminating search!")
        early_termination_flag = True
        break

if not early_termination_flag:
    print("Max number of steps has been reached!")

print(f"MINIMA: {MINIMA}")

# printing results
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

dbscan_labels = DBSCAN(
    eps=np.pi/12,
    min_pts=1,
).fit_predict(np.asarray([cur[0] for cur in MINIMA]))

res = {int(label) : (1e9, -1) for label in np.unique(dbscan_labels)}

for i in range(len(MINIMA)):
    cluster_id = dbscan_labels[i]
    if MINIMA[i][1] < res[cluster_id][0]:
        res[cluster_id] = MINIMA[i][1], i

print(f"Results of clustering: {res}\nThere are relative energy and number of structure for each cluster. Saved in `{exp_name}_clustering_results.json`")
json.dump(res, open(f'{exp_name}_clustering_results.json', 'w'))

print(f"Saving final ensemble into `{exp_name}_final_ensemble.xyz`")
ens_xyz_str = ""
for _, structure_id in res.values():
    cur_xyz = ""
    with open(f"{exp_name}_minima/{structure_id}.xyz", "r") as cur_xyz_reader:
        cur_xyz = "".join([line for line in cur_xyz_reader])
    ens_xyz_str += cur_xyz + "\n"

with open(f"{exp_name}_final_ensemble.xyz", "w") as ens_writer:
    ens_writer.write(ens_xyz_str)

print(f"Saving all points at `{exp_name}_all_points.json`")
json.dump(
    {
        'query_points' : query_points.tolist(),
        'observations' : observations.tolist()
    },
    open(f'{exp_name}_all_points.json', 'w')
)
