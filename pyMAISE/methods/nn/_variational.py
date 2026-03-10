from tensorflow_probability.python.layers import DenseReparameterization
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.layers import util as tfp_layers_util
import tensorflow_probability as tfp

from pyMAISE.methods.nn._layer import Layer


class VariationalLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary
        self._data = super().build_data(self._data, parameters)

        # kl_weight scales the KL divergence term relative to the MSE loss.
        # The correct ELBO formulation is: loss = MSE + (1/n_train) * KL.
        # Because Keras adds model.losses (the KL sum) directly to the
        # per-batch mean MSE, kl_weight should be set to 1/n_train so the
        # two terms are on the same scale.  Defaults to 1.0 (unscaled) for
        # backwards compatibility, but users should pass kl_weight in the
        # Variational_output params dict.
        kl_weight = parameters.get("kl_weight", 1.0)
        if kl_weight != 1.0:
            self._data["kernel_divergence_fn"] = (
                lambda q, p, ignore, w=kl_weight: kl_lib.kl_divergence(q, p) * w
            )
            self._data["bias_divergence_fn"] = (
                lambda q, p, ignore, w=kl_weight: kl_lib.kl_divergence(q, p) * w
            )

        # Assert keras non-default variables are defined
        assert self._data["units"] is not None

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Set pyMAISE hyperparameter to keras-tuner hyperparameter
        return DenseReparameterization(**super().sample_parameters(self._data, hp))

    def reset(self):
        self._data = {
            "units": None,
            "activation": None,
            "activity_regularizer": None,
            "trainable": True,
            "kernel_posterior_fn": tfp_layers_util.default_mean_field_normal_fn(),
            "kernel_posterior_tensor_fn": (lambda d: d.sample()),
            "kernel_prior_fn": tfp.layers.default_multivariate_normal_fn,
            "kernel_divergence_fn": (lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
            "bias_posterior_fn": tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
            "bias_posterior_tensor_fn": (lambda d: d.sample()),
            "bias_prior_fn": None,
            "bias_divergence_fn": (lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
        }
        super().reset()

    def increment_layer(self):
        return super().increment_layer()

    # ==========================================================================
    # Getters
    def num_layers(self, hp):
        return super().num_layers(hp)

    def sublayer(self, hp):
        return super().sublayer(hp)

    def wrapper(self):
        return super().wrapper()
