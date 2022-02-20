import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, List
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
tf.get_logger().setLevel('ERROR')
import tensorflow_datasets as tfds
import numpy as np

class Frame(NamedTuple):
    timestep: int
    predictions: jnp.ndarray
    new_data: jnp.ndarray
    test_loss: float

class Results(NamedTuple):
    model_name: str
    real_x: jnp.ndarray
    real_y: jnp.ndarray
    start_x: jnp.ndarray
    start_y: jnp.ndarray
    timesteps: List[Frame]


def preprocess(batch):
    batch['image'] = batch['image'].astype(np.float32) / 255 - 0.5
    return batch

def load_datasets(n_start_data):
    ds = tfds.load("mnist", split='train', batch_size=-1, )
    ds = preprocess(tfds.as_numpy(ds))
    ds['observed'] = np.random.randint(0, ds['image'].shape[0], size=(n_start_data,))

    test_ds = tfds.load("mnist", split='test', batch_size=-1)
    test_ds = preprocess(tfds.as_numpy(test_ds))
    test_ds['observed'] = np.arange(test_ds['image'].shape[0])
    return ds, test_ds


# Why does jax.jit slow down my code significantly here? It seems like it is recompiled constantly
# @partial(jax.jit, static_argnums=(0,))
def sample_batch(batch_size, rng_key, data_dict):
    indices = jax.random.choice(rng_key, data_dict['observed'], shape=(batch_size,))
    # indices = np.arange(8)
    n = data_dict['observed'].shape[0]
    # batch = dict(image= data_dict['image'][indices], label=data_dict['label'][indices])
    # return batch, data_dict['image'].shape[0]
    return data_dict['image'][indices], data_dict['label'][indices], n

@partial(jax.jit, static_argnums=(0,))
def sample_candidate_batch(batch_size, rng_key, data_dict):
    indices = jax.random.choice(rng_key, data_dict['image'].shape[0], shape=(batch_size,))
    return data_dict['image'][indices], data_dict['label'][indices], indices

dmatrix = jax.vmap(jax.vmap(lambda x, y: jnp.sum( ((jnp.ravel(x) - jnp.ravel(y))**2) ), in_axes=(None, 0)), in_axes=(0, None))

@jax.jit
def compute_median(x):
    n = x.shape[0]
    return jax.lax.stop_gradient(jnp.sort(x)[n // 2])

def sinoidal_prior(mus):
    pass

@jax.jit
def uniform_prior(mus):
    return 0



@jax.jit
def gram_matrix_median_trick(x):
    n = x.shape[0]
    distance_matrix = dmatrix(x, jax.lax.stop_gradient(x))
    median = jax.lax.stop_gradient(compute_median(distance_matrix.flatten()))
    return jnp.exp(-distance_matrix / (median / jnp.log(n)))

if __name__ == "__main__":
    ds, test_ds = load_datasets()
    print(ds['image'].shape, test_ds['image'].shape)
    print(ds['image'].min(), ds['image'].max())