import tensorflow as tf
from tensorflow_probability.python.layers import DenseReparameterization
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.layers import util as tfp_layers_util
import tensorflow_probability as tfp

from pyMAISE.methods.nn._layer import Layer


def _make_prior_fn(prior_variance):
    """Return a kernel_prior_fn for DenseReparameterization with N(0, prior_variance).

    The TFP default prior is N(0, 1).  Reducing prior_variance (e.g. to 0.5 as
    used in the reference viAL implementation) tightens the prior, which allows
    weights to be more precise while still producing a valid KL penalty.

    Parameters
    ----------
    prior_variance : float
        Variance of the isotropic Normal prior.  Standard deviation is
        ``sqrt(prior_variance)``.
    """
    scale = prior_variance ** 0.5

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        del name, trainable, add_variable_fn
        dist = tfp.distributions.Normal(
            loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(scale)
        )
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn


# Keys consumed by VariationalLayer.build() that must be popped before
# passing the sampled dict to DenseReparameterization.
_VARIATIONAL_KEYS = ("kl_weight", "prior_variance", "posterior_scale_init")


class VariationalLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary.
        # kl_weight, prior_variance, and posterior_scale_init are present in
        # self._data (see reset()), so build_data copies them here — including
        # mai.Choice objects for hyperparameter search.
        self._data = super().build_data(self._data, parameters)

        # _kl_weight_container is injected by nnHyperModel when kl_schedule is
        # active.  It is a shared mutable list captured by divergence lambdas so
        # that KLAnnealingCallback can update the weight each epoch.  It is not
        # tunable and not stored in _data; store it as an instance variable so
        # build() can reference it without going through sample_parameters.
        #
        # Python's deepcopy returns the same function object for lambdas, so
        # the shared reference is preserved when _build_tree deepcopies the layer.
        self._kl_weight_container = parameters.get("_kl_weight_container", None)

        # Assert keras non-default variables are defined
        assert self._data["units"] is not None

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Resolve any mai.Choice / HyperParameters values to scalars.
        sampled = super().sample_parameters(self._data, hp)

        # Pop variational-specific keys — DenseReparameterization does not
        # accept them; we convert them to callables below.
        kl_weight          = sampled.pop("kl_weight")
        prior_variance     = sampled.pop("prior_variance")
        posterior_scale_init = sampled.pop("posterior_scale_init")

        # ── KL divergence weight ─────────────────────────────────────────────
        # kl_weight scales the KL divergence term relative to the MSE loss.
        # The correct ELBO formulation is: loss = MSE + (1/n_train) * KL.
        # Because Keras adds model.losses (the KL sum) directly to the
        # per-batch mean MSE, kl_weight should be set to 1/n_train so the
        # two terms are on the same scale.  Defaults to 1.0 for backwards
        # compatibility.
        if self._kl_weight_container is not None:
            c = self._kl_weight_container
            sampled["kernel_divergence_fn"] = (
                lambda q, p, ignore, c=c: kl_lib.kl_divergence(q, p) * c[0]
            )
            sampled["bias_divergence_fn"] = (
                lambda q, p, ignore, c=c: kl_lib.kl_divergence(q, p) * c[0]
            )
        elif kl_weight != 1.0:
            sampled["kernel_divergence_fn"] = (
                lambda q, p, ignore, w=kl_weight: kl_lib.kl_divergence(q, p) * w
            )
            sampled["bias_divergence_fn"] = (
                lambda q, p, ignore, w=kl_weight: kl_lib.kl_divergence(q, p) * w
            )

        # ── Prior variance ───────────────────────────────────────────────────
        # Variance of the isotropic Normal prior N(0, prior_variance).
        # TFP default is 1.0.  Smaller values (e.g. 0.5) tighten the prior.
        if prior_variance != 1.0:
            sampled["kernel_prior_fn"] = _make_prior_fn(prior_variance)

        # ── Posterior scale initializer ──────────────────────────────────────
        # Initial value of the untransformed scale rho; effective sigma =
        # softplus(rho).  TFP default is -3.0 (sigma ≈ 0.049).  Setting to
        # -4.0 (sigma ≈ 0.018) starts the posterior nearly deterministic,
        # improving early convergence.
        if posterior_scale_init is not None:
            sampled["kernel_posterior_fn"] = (
                tfp_layers_util.default_mean_field_normal_fn(
                    untransformed_scale_initializer=tf.initializers.Constant(
                        posterior_scale_init
                    )
                )
            )

        return DenseReparameterization(**sampled)

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
            # Tunable variational hyperparameters — may be scalar or mai.Choice.
            # Popped in build() before passing to DenseReparameterization.
            "kl_weight": 1.0,
            "prior_variance": 1.0,
            "posterior_scale_init": None,
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
