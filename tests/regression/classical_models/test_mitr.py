import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_MITR
from pyMAISE.preprocessing import scale_data, train_test_split


def test_mitr():
    # ===========================================================================
    # Regression test parameters
    num_observations = 1000
    num_features = 6
    num_outputs = 22

    # ===========================================================================
    # pyMAISE initialization
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        num_configs_saved=1,
        cuda_visible_devices="-1",
    )

    # Assertions for global settings
    assert global_settings.verbosity == 1
    assert global_settings.random_state == 42
    assert global_settings.num_configs_saved == 1

    # Get MITR data
    data, inputs, outputs = load_MITR()

    # Assert inputs and outputs are the correct size
    assert inputs.shape[0] == num_observations and inputs.shape[1] == num_features
    assert outputs.shape[0] == num_observations and outputs.shape[1] == num_outputs

    # Train test split
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )
    xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, _ = scale_data(ytrain, ytest, MinMaxScaler())
    data = (xtrain, xtest, ytrain, ytest)

    # Train-test split size assertions
    assert (
        data[0].shape[0] == num_observations * (1 - 0.3)
        and data[0].shape[1] == num_features
    )
    assert (
        data[1].shape[0] == num_observations * 0.3 and data[1].shape[1] == num_features
    )
    assert (
        data[2].shape[0] == num_observations * (1 - 0.3)
        and data[2].shape[1] == num_outputs
    )
    assert (
        data[3].shape[0] == num_observations * 0.3 and data[3].shape[1] == num_outputs
    )

    # ===========================================================================
    # Model initialization
    model_settings = {
        "models": ["Linear", "Lasso", "DT", "KN", "RF"],
    }
    tuning = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # ===========================================================================
    # Hyper-parameter tuning
    param_spaces = {
        "Linear": {"fit_intercept": [True, False]},
        "Lasso": {"alpha": np.logspace(-6, -2, 10)},
        "DT": {
            "max_depth": [None, 5, 10, 25, 50],
            "max_features": [None, "sqrt", "log2", 0.2, 0.4, 0.6, 0.8, 1],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
        },
        "RF": {
            "n_estimators": [50, 100, 150],
            "criterion": ["squared_error", "absolute_error"],
            "min_samples_split": [2, 4],
            "max_features": [None, "sqrt", "log2", 1],
        },
        "KN": {
            "n_neighbors": [1, 2, 4, 6, 8, 10, 14, 17, 20],
            "weights": ["uniform", "distance"],
            "leaf_size": [1, 5, 10, 15, 20, 25, 30],
        },
    }

    random_search_configs = tuning.random_search(
        param_spaces=param_spaces,
        models=param_spaces.keys(),
        n_iter=10,
        cv=ShuffleSplit(
            n_splits=1, test_size=0.15, random_state=global_settings.random_state
        ),
    )

    # ===========================================================================
    # Model post-processing
    postprocessor = mai.PostProcessor(
        data=data,
        model_configs=[random_search_configs],
    )
    metrics = postprocessor.metrics(metrics={"MSE": mean_squared_error})[
        [
            "Model Types",
            "Train MAE",
            "Train MSE",
            "Train RMSE",
            "Train R2",
            "Test MAE",
            "Test MSE",
            "Test RMSE",
            "Test R2",
        ]
    ]

    print("pyMAISE Values\n", metrics)

    # Assert correct number of models and that all Test R2 values are positive
    assert metrics.shape[0] == len(model_settings["models"])
    assert (metrics["Test R2"] > 0).all(), (
        f"Some models perform worse than predicting the mean:\n"
        f"{metrics[['Model Types', 'Test R2']]}"
    )
