import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_loca, load_chf
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import SplitSequence, scale_data, train_test_split
# from tensorflow_probability.python.layers import DenseVariational


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
            "fitting_params": {"batch_size": 16, "epochs": 5, "validation_split": 0.15},
        },
    }
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="r2_score",
        cv=TimeSeriesSplit(n_splits=2),
    )

    # TODO: figure out why vFNN ain't working
    # assert isinstance(grid_search_configs["rnn"][0], pd.DataFrame)
    # assert isinstance(grid_search_configs["rnn"][1], nnHyperModel)
    # assert grid_search_configs["rnn"][0].shape == (2, 1)
    # assert tuner.cv_performance_data["rnn"].shape == (2, 2)

    # # Model post-processing
    # new_model_settings = {
    #     "rnn": {
    #         "fitting_params": {
    #             "epochs": 10,
    #         },
    #     },
    # }
    # postprocessor = mai.PostProcessor(
    #     data=split_data,
    #     model_configs=[grid_search_configs],
    #     new_model_settings=new_model_settings,
    #     yscaler=yscaler,
    # )
    # assert postprocessor.metrics().shape == (2, 12)
