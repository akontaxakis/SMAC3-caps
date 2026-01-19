from pathlib import Path
import logging
from ConfigSpace import Configuration

from AutoML_data_manager.data_manager import DataManager
from example_config import get_tpot_configspace_classifiers_for_SMAC4AC, update_log_filename, \
    get_tpot_configspace_classifiers_for_SMAC4AC_full
from example_function import train_without_budget, train_full
from smac import Scenario, AlgorithmConfigurationFacade
from smac.initial_design import RandomInitialDesign
logging.basicConfig(level=logging.INFO)


data_id = "jannis"
seed = 7777
selection = 8
sel_algo = "caps_greedy"
lamda = 0.5
N = 20
mode = "CAPS"
# Load dataset with the provided data_id
data = DataManager(data_id, r'datasets', replace_missing=True)
X = data.data['X_train']
y = data.data['Y_train']

def train(config: Configuration, seed: int = 0) -> float:
    scores = train_full(config, X,y, seed)
    return scores




if __name__ == '__main__':
    # SMAC configuration space and logging
    configspace = get_tpot_configspace_classifiers_for_SMAC4AC_full()
    path = r"logging_caps.yaml"
    exp_id = f"SMAC_{data_id}_{seed}_{selection}_{sel_algo}_{lamda}_{N}.log"
    path = update_log_filename(path, exp_id)
    logging_path = Path(path)

    # Define the SMAC scenario
    scenario = Scenario(
        configspace=configspace,
        n_trials=500,
        deterministic=True,
        objectives=["quality"],
        seed=seed,
        data_id=data_id,
        output_directory="smac_output",
        sel_algo=sel_algo,
        selection=selection,
        lamda=lamda,
        mode=mode,
        N=N,
        trial_walltime_limit=300
    )

    # Define the initial design for SMAC
    initial_design = RandomInitialDesign(scenario, n_configs=20)

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