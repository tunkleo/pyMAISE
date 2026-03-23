import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_chf
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import scale_data

matplotlib.use("Agg")  # non-interactive backend for CI


def test_chf_vfnn():
    # Initialize pyMAISE
    _ = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        num_configs_saved=2,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPUs only
        run_parallel=False,
    )

    # Load CHF data and shrink dataset for test speed
    train_data, xtrain, ytrain, test_data, xtest, ytest = load_chf()

    assert xtrain.shape == (2000, 6)
    assert xtest.shape == (500, 6)
    assert ytrain.shape == (2000, 1)
    assert ytest.shape == (500, 1)

    xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, MinMaxScaler())
    split_data = (xtrain, xtest, ytrain, ytest)

    assert split_data[0].shape == (2000, 6)
    assert split_data[1].shape == (500, 6)
    assert split_data[2].shape == (2000, 1)
    assert split_data[3].shape == (500, 1)
    # print(inputs.shape)

    # Scale KL divergence to match the per-sample MSE loss.
    # Without this the KL term dominates and the network cannot learn.
    kl_weight = 1.0 / xtrain.shape[0]

    # vFNN model settings
    structural = {
        "Dense_input": {
            "units": mai.Choice([50, 100]),
            "activation": "tanh",
        },
        "Dense_hidden1": {
            "units": mai.Choice([50, 100]),
            "activation": "tanh",
        },
        "Variational_output": {
            "units": 1,
            "activation": "linear",
            "kl_weight": kl_weight,
        },
    }
    model_settings = {
        "models": ["vfnn"],
        "vfnn": {
            "structural_params": structural,
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": mai.Choice([0.0001, 0.001]),
            },
            "compile_params": {
                "loss": "mean_squared_error",
                "metrics": ["mean_squared_error"],
            },
            "fitting_params": {"batch_size": 32, "epochs": 30, "validation_split": 0.15},
        },
    }
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="r2_score",
        cv=TimeSeriesSplit(n_splits=2),
    )

    assert isinstance(grid_search_configs["vfnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["vfnn"][1], nnHyperModel)
    assert grid_search_configs["vfnn"][0].shape == (2, 1)
    # Shape is (2, n_trials): row 0 = mean scores, row 1 = std scores
    assert tuner.cv_performance_data["vfnn"].shape[0] == 2

    # ---- PostProcessor -------------------------------------------------------
    # Use a small n_mc_samples so the test stays fast
    postprocessor = mai.PostProcessor(
        data=split_data,
        model_configs=[grid_search_configs],
        yscaler=yscaler,
        n_mc_samples=20,
    )

    metrics = postprocessor.metrics()
    # 2 saved configs × (train + test) × 5 default regression metrics = 10 cols + 2 names
    assert metrics.shape == (2, 12)

    # Both saved configs should show meaningful predictive skill on the test set
    assert (metrics["Test R2"] > 0.4).all(), (
        f"vFNN configs underperforming — check kl_weight scaling:\n"
        f"{metrics[['Parameter Configurations', 'Test R2']]}"
    )

    # MC Samples column should be populated for both saved configs
    for i in range(2):
        samples = postprocessor._models["MC Samples"][i]
        assert samples is not None
        assert samples.shape == (20, 500, 1)  # (n_mc, n_test, n_outputs)
        assert not np.all(samples[0] == samples[1])  # samples must differ across draws

    # ---- Variational parity plot ---------------------------------------------
    fig, ax = plt.subplots()
    ax = postprocessor.variational_parity_plot(ax=ax, idx=0)
    assert ax is not None
    plt.close(fig)

    # Calling on a non-variational model index should raise ValueError.
    # (We only have variational models in this test, so verify the plot
    #  works for both saved configs instead.)
    fig, ax = plt.subplots()
    ax = postprocessor.variational_parity_plot(ax=ax, idx=1)
    assert ax is not None
    plt.close(fig)
