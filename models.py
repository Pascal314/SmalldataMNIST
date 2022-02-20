import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import utils
import optax
# uncertainty func could be a string argument instead with the base class deciding which to use.
class Model():
    def __init__(self, transformed, optimizer, dummy_input, log_likelihood, uncertainty_func):
        self.optimizer = optimizer
        self.transformed = transformed
        self.dummy_input = dummy_input
        self.uncertainty_func = uncertainty_func
        self.log_likelihood = log_likelihood

    def apply(self, *args):
        return self.transformed.apply(*args)

    def init(self, *args):
        return self.transformed.init(*args)

    def loss(self, params, batch):
        raise NotImplementedError

    def train_step(self, params, opt_state, batch):
        raise NotImplementedError


class PlainEnsemble(Model):
    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, batch):
        X, Y, n = batch
        mus = self.apply(params, X)
        nll = - self.log_likelihood(mus, Y, n)
        return jnp.sum(nll)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, batch):
        grads = jax.grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state


class fSVGDEnsemble(Model):
    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, batch):
        X, Y, n = batch
        mus = self.apply(params, X)
        nll = - self.log_likelihood(mus, Y, n)
        # extra_mus = self.apply(params, X + 0.1 * np.random.normal(0, 1, size=X.shape))
        extra_mus = mus
        Kij = utils.gram_matrix_median_trick(extra_mus)
        return jnp.sum(nll + jnp.sum(Kij, axis=1) / jax.lax.stop_gradient(jnp.sum(Kij, axis=1)))

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, batch):
        grads = jax.grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

@partial(jax.jit, static_argnums=0)
def predictions_stddev(self, params, x):
    predictions = self.apply(params, x)
    stddev = jnp.std(predictions, axis=0)
    return np.mean(stddev, axis=-1)

@partial(jax.jit, static_argnums=0)
def probability_stddev(self, params, x):
    predictions = self.apply(params, x)
    probabilities = jax.nn.softmax(predictions, axis=-1)
    stddev = jnp.std(predictions, axis=0)
    return np.mean(stddev**2, axis=-1)

@partial(jax.jit, static_argnums=0)
def disagreement(self, params, x):
    predictions = self.apply(params, x)
    probabilities = jnp.mean(jax.nn.softmax(predictions, axis=-1), axis=0)
    pred_labels = jnp.argmax(probabilities, axis=-1)
    individual_labels = jnp.argmax(predictions, axis=-1) # this is n_ensembles x batch_size
    return jnp.mean(individual_labels != pred_labels, axis=0)

@partial(jax.jit, static_argnums=0)
def predictive_dist_entropy(self, params, x):
    logits = self.apply(params, x)
    probabilities = jax.nn.softmax(logits, axis=-1)
    mean_distribution = jnp.mean(probabilities, axis=0)
    entropy = jnp.sum(mean_distribution * jnp.log(mean_distribution), axis=-1)
    return -entropy

# @partial(jax.jit, static_argnums=0)
def random_uncertainty(self, params, x):
    random = np.random.uniform(size=x.shape[0])
    return random