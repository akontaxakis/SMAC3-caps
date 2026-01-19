from pathlib import Path
import logging

from AutoML_data_manager.data_manager import DataManager
from example_config import get_tpot_configspace_classifiers_for_SMAC4AC, update_log_filename
from example_function import train_quality_cost
from smac import Scenario, AlgorithmConfigurationFacade
from smac.acquisition.function import EIPS
from smac.initial_design import RandomInitialDesign
from ConfigSpace import Configuration


logging.basicConfig(level=logging.INFO)


data_id = "jannis"
seed = 7777
selection = 8
sel_algo = "none"
lamda = 0.5
N = 20
mode = "SMAC"

# Load dataset with the provided data_id
data = DataManager(data_id, r'datasets', replace_missing=True)
X = data.data['X_train']
y = data.data['Y_train']

def train(config: Configuration, seed: int = 0) -> tuple[float, float]:
    return train_quality_cost(config, X,y, seed)

if __name__ == '__main__':

    # SMAC configuration space and logging
    configspace = get_tpot_configspace_classifiers_for_SMAC4AC()
    path = r"logging_caps.yaml"
    exp_id = f"SMACEIcost_{data_id}_{seed}_{selection}_{sel_algo}_{lamda}_{N}.log"
    path = update_log_filename(path, exp_id)
    logging_path = Path(path)

    # Define the SMAC scenario
    scenario = Scenario(
        configspace=configspace,
        n_trials=500,
        deterministic=True,
        objectives=["quality", "cost"],
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

    # Define the initial design for SMAC
    initial_design = RandomInitialDesign(scenario, n_configs=20)

    # Configure SMAC
    from smac.model.multi_objective_model import MultiObjectiveModel

    from smac.model.random_forest import RandomForest
    from smac.model.multi_objective_model import MultiObjectiveModel

    # Define two separate RandomForest models: one for quality, one for cost
    rf_quality = RandomForest(configspace=configspace, seed=seed)
    rf_cost = RandomForest(configspace=configspace, seed=seed)

    # Multi-objective model wrapper
    multi_obj_model = MultiObjectiveModel(models=[rf_quality, rf_cost], objectives=["quality", "cost"], seed=seed)

    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=train,  # Must return (quality, cost)
        initial_design=initial_design,
        acquisition_function=EIPS(),  # EIPS expects two objectives
        model=multi_obj_model,  # Use the available model
        logging_level=logging_path
    )

    # Run SMAC optimization to find the best configuration
    best_config = smac.optimize()
    print("Best configuration found:", best_config)
