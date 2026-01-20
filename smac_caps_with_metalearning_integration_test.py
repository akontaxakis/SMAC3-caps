from pathlib import Path
import logging

from AutoML_data_manager.data_manager import DataManager
from example_config import update_log_filename, \
    get_metalearning_pipelines, get_tpot_configspace_classifiers_for_SMAC4AC_full
from example_function import train_full
from smac import Scenario, AlgorithmConfigurationFacade
from ConfigSpace import Configuration

logging.basicConfig(level=logging.INFO)

data_id = "jannis"
seed = 7777
selection = 8
sel_algo = "caps_greedy"
lamda = 0.5
N = 20
mode = "CAPS"

import numpy as np

# Restore np.float alias for compatibility with TPOT
if not hasattr(np, "float"):
    np.float = float

# Load dataset with the provided data_id
iris = DataManager(data_id, r'datasets', replace_missing=True)
X = iris.data['X_train']
y = iris.data['Y_train']

def train(config: Configuration, seed: int = 0) -> float:
    scores = train_full(config, X,y, seed)
    return scores

if __name__ == '__main__':
    # SMAC configuration space and logging
    configspace = get_tpot_configspace_classifiers_for_SMAC4AC_full()
    path = r"logging_caps.yaml"
    exp_id = f"SMAC_MT_{data_id}_{seed}_{selection}_{sel_algo}_{lamda}_{N}.log"
    path = update_log_filename(path, exp_id)
    logging_path = Path(path)

    # Define the SMAC scenario
    scenario = Scenario(
        configspace=configspace,
        n_trials=10000,
        deterministic=True,
        objectives=["quality"],
        seed=seed,
        mode=mode,
        data_id=data_id,
        output_directory="smac_output",
        sel_algo=sel_algo,
        selection=selection,
        lamda=lamda,
        N=N,
        trial_walltime_limit=300
    )

    from smac.initial_design.abstract_initial_design import AbstractInitialDesign as InitialDesign
    from ConfigSpace import Configuration


    # Fixed initial design that returns exactly these
    class FixedInitialDesign(InitialDesign):
        def _select_configurations(self):
            return get_metalearning_pipelines(data_id)

    # Use it
    initial_design = FixedInitialDesign(scenario, n_configs=len(get_metalearning_pipelines(data_id)))

    # Configure SMAC
    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=train,
        initial_design=initial_design,
        logging_level=logging_path
    )

    # Run SMAC optimization to find the best configuration
    best_config = smac.optimize()
    print("Best configuration found:", best_config)