from pathlib import Path
import logging

from AutoML_data_manager.data_manager import DataManager
from example_config import get_tpot_configspace_classifiers_for_SMAC4AC, update_log_filename
from smac import Scenario, MultiFidelityFacade
from smac.initial_design import RandomInitialDesign
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from ConfigSpace import Configuration

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

def train(config: Configuration, budget: float, seed: int = 0) -> float:
    """Train a model based on the configuration provided and return the validation error."""

    # Preprocessor configuration
    preprocessor_name = config['preprocessor']
    if preprocessor_name == 'sklearn.preprocessing.Binarizer':
        preprocessor = Binarizer(threshold=config.get('Binarizer__threshold', 0.0))
    elif preprocessor_name == 'sklearn.preprocessing.MaxAbsScaler':
        preprocessor = MaxAbsScaler()
    elif preprocessor_name == 'sklearn.preprocessing.MinMaxScaler':
        preprocessor = MinMaxScaler()
    elif preprocessor_name == 'sklearn.preprocessing.Normalizer':
        preprocessor = Normalizer(norm=config.get('Normalizer__norm', 'l2'))
    elif preprocessor_name == 'sklearn.preprocessing.StandardScaler':
        preprocessor = StandardScaler()
    elif preprocessor_name == 'sklearn.decomposition.PCA':
        preprocessor = PCA(svd_solver=config.get('PCA__svd_solver', 'randomized'),
                           iterated_power=config.get('PCA__iterated_power', 1))
    else:
        raise ValueError(f"Unknown preprocessor: {preprocessor_name}")

    # Classifier configuration
    classifier_name = config['classifier']
    if classifier_name == 'sklearn.ensemble.RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=100,
            max_features=config['RandomForestClassifier__max_features'],
            min_samples_split=config['RandomForestClassifier__min_samples_split'],
            min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
            bootstrap=config['RandomForestClassifier__bootstrap'],
            criterion=config['RandomForestClassifier__criterion'],
            random_state=seed
        )
    elif classifier_name == 'sklearn.ensemble.GradientBoostingClassifier':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=config['GradientBoostingClassifier__learning_rate'],
            max_depth=config['GradientBoostingClassifier__max_depth'],
            min_samples_split=config.get('GradientBoostingClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('GradientBoostingClassifier__min_samples_leaf', 1),
            subsample=config.get('GradientBoostingClassifier__subsample', 1.0),
            max_features=config.get('GradientBoostingClassifier__max_features', None),
            random_state=seed
        )
    elif classifier_name == 'sklearn.ensemble.ExtraTreesClassifier':
        model = ExtraTreesClassifier(
            n_estimators=100,
            max_features=config['ExtraTreesClassifier__max_features'],
            min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
            min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
            criterion=config['ExtraTreesClassifier__criterion'],
            bootstrap=config['ExtraTreesClassifier__bootstrap'],
            random_state=seed
        )
    elif classifier_name == 'sklearn.tree.DecisionTreeClassifier':
        model = DecisionTreeClassifier(
            criterion=config['DecisionTreeClassifier__criterion'],
            max_depth=config['DecisionTreeClassifier__max_depth'],
            min_samples_split=config.get('DecisionTreeClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('DecisionTreeClassifier__min_samples_leaf', 1),
            random_state=seed
        )
    elif classifier_name == 'sklearn.neighbors.KNeighborsClassifier':
        model = KNeighborsClassifier(
            n_neighbors=config['KNeighborsClassifier__n_neighbors'],
            weights=config['KNeighborsClassifier__weights'],
            p=config['KNeighborsClassifier__p']
        )
    elif classifier_name == 'sklearn.naive_bayes.GaussianNB':
        model = GaussianNB()
    elif classifier_name == 'sklearn.naive_bayes.BernoulliNB':
        model = BernoulliNB(
            alpha=config['BernoulliNB__alpha'],
            fit_prior=config['BernoulliNB__fit_prior']
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Construct the pipeline with the preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('model', model)]
    pipeline = Pipeline(steps)
    # subsample by budget
    frac = float(np.clip(budget, 0.05, 1.0))
    n = len(y)
    rng = np.random.RandomState(seed)
    m = max(50, int(frac * n))
    idx = rng.permutation(n)[:m]
    Xb, yb = X[idx], y[idx]
    # Perform Cross-Validation
    cv = StratifiedKFold(n_splits=2, shuffle=False)
    scores = cross_val_score(pipeline, Xb, yb, cv=cv)

    # Return the validation error (1 - mean accuracy)
    return 1 - np.mean(scores)

if __name__ == '__main__':
    # SMAC configuration space and logging
    configspace = get_tpot_configspace_classifiers_for_SMAC4AC()
    path = r"logging_caps.yaml"
    exp_id = f"SMAC_HB_{data_id}_{seed}_{selection}_{sel_algo}_{lamda}_{N}.log"
    path = update_log_filename(path, exp_id)
    logging_path = Path(path)

    # Define the SMAC scenario
    scenario = Scenario(
        configspace=configspace,
        n_trials=100000,
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
        trial_walltime_limit=300,
        min_budget=0.1,  # NEW: 10% of data
        max_budget=1.0  # NEW: 100% of data
    )

    # Define the initial design for SMAC
    initial_design = RandomInitialDesign(scenario, n_configs=20)

    # Configure SMAC
    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=train,
        initial_design=initial_design,
        logging_level=logging_path  # keep your logging setup as you had it
    )

    best_config = smac.optimize()
    print("Best configuration found:", best_config)