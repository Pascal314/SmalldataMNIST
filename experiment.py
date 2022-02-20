import haiku as hk 
import jax
import jax.numpy as jnp
from jax import nn
import sys
import numpy as np
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import pickle
import models
import utils
from types import SimpleNamespace
from network import ConvNet

def create_models(n_networks):
    """Creates a dictionary containing the models to test in this experiment.

    Returns:
        dict: A dictionary of models 
    """
    dummy_input = np.zeros((1, 28, 28, 1))
    dummy_x = dummy_input
    model_dict = {}

    def forward(x):
        y = ConvNet(16, 32)(x)
        y = hk.Linear(10)(y)
        return y

    net = hk.without_apply_rng(hk.transform(forward))
    parallel_init = jax.vmap(net.init, in_axes=(0, None))
    ensemble_init = lambda key, x: parallel_init(jax.random.split(key, n_networks), x)
    ensemble_apply = jax.vmap(net.apply, in_axes=(0, None))
    ensemble_net = hk.Transformed(ensemble_init, ensemble_apply)
    optimizer = optax.adam(1e-3)

    @jax.jit
    def categorical_cross_entropy(logits, labels, n):
        return n / labels.shape[0] * jnp.sum(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))
    categorical_cross_entropy = jax.vmap(categorical_cross_entropy, in_axes=(0, None, None))

    log_likelihood = lambda logits, labels, n: -categorical_cross_entropy(logits, labels, n) + utils.uniform_prior(logits)

    model_dict['fSVGD_ent'] = models.fSVGDEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.predictive_dist_entropy)
    model_dict['fSVGD_dis'] = models.fSVGDEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.disagreement)
    model_dict['fSVGD_std'] = models.fSVGDEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.predictions_stddev)
    model_dict['fSVGD_pstd'] = models.fSVGDEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.probability_stddev)

    model_dict['plain_ent'] = models.PlainEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.predictive_dist_entropy)
    model_dict['plain_dis'] = models.PlainEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.disagreement)
    model_dict['plain_std'] = models.PlainEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.predictions_stddev)
    model_dict['plain_pstd'] = models.PlainEnsemble(ensemble_net, optimizer, dummy_input, log_likelihood, models.probability_stddev)

    def single_net(x):
        y = ConvNet(16, 32)(x)
        y = hk.Linear(10)(y)
        return y.reshape(1, x.shape[0], 10)

    single_net = hk.without_apply_rng(hk.transform(single_net))
    model_dict['random'] = models.PlainEnsemble(single_net, optimizer, dummy_input, log_likelihood, models.random_uncertainty, )
    
    def forward(x):
        y = ConvNet(16, 32)(x)
        y = hk.Linear(10 * n_networks)(y)
        return y.reshape(x.shape[0], n_networks, 10).swapaxes(0, 1)

    net = hk.without_apply_rng(hk.transform(forward))
    model_dict['sharedbase_fSVGD_ent'] = models.fSVGDEnsemble(net, optimizer, dummy_input, log_likelihood, models.predictive_dist_entropy)
    model_dict['sharedbase_fSVGD_dis'] = models.fSVGDEnsemble(net, optimizer, dummy_input, log_likelihood, models.disagreement)
    model_dict['sharedbase_fSVGD_std'] = models.fSVGDEnsemble(net, optimizer, dummy_input, log_likelihood, models.predictions_stddev)
    model_dict['sharedbase_fSVGD_pstd'] = models.fSVGDEnsemble(net, optimizer, dummy_input, log_likelihood, models.probability_stddev)

    return model_dict

# eerst middelen, of eerst softmax aanroepen en dan middelen? (tweede is logischer)
@partial(jax.jit, static_argnums=0)
def evaluate(model, params, batch):
    image, label, n = batch
    logits = model.apply(params, image)
    probabilities = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)
    pred_labels = jnp.argmax(probabilities, axis=-1)
    return jnp.mean(pred_labels == label)

@partial(jax.jit, static_argnums=2)
def choose_next_points(uncertainty, candidate_indices, n):
    argsorted = jnp.argsort(uncertainty.squeeze())[::-1]
    return candidate_indices[argsorted[0:n]]


def run_experiment(model, args, model_name):
    train_data, test_data = utils.load_datasets(args.start_n_data)
    key = jax.random.PRNGKey(args.rng_seed)

    key, sub_key = jax.random.split(key, 2)
    params = model.init(sub_key, model.dummy_input)
    opt_state = model.optimizer.init(params)

    test_losses = []
    train_losses = []

    outer_pbar = tqdm(range(args.total_steps))
    outer_pbar.set_description(f'{model_name:<15}')
    for i in outer_pbar:
        inner_pbar = tqdm(range(args.train_steps_per_data_sample), leave=False)
        # batch = utils.sample_batch(args.batch_size, sub_key, train_data)

        for j in inner_pbar:
            key, sub_key = jax.random.split(key, num=2)
            batch = utils.sample_batch(args.batch_size, sub_key, train_data)
            params, opt_state = model.train_step(params, opt_state, batch)

        key, sub_key1, sub_key2, sub_key3 = jax.random.split(key, num=4)

        train_accuracy = evaluate(model, params, utils.sample_candidate_batch(1024, sub_key2, train_data))
        train_losses.append(train_accuracy)
        test_batch = utils.sample_batch(1024, sub_key2, test_data)
        test_accuracy = evaluate(model, params, test_batch)
        test_losses.append(test_accuracy)

        candidate_image, _, candidate_indices = utils.sample_candidate_batch(1024, sub_key3, train_data)
        uncertainty = model.uncertainty_func(model, params, candidate_image)
        new_observed = choose_next_points(uncertainty, candidate_indices, args.sample_n_points_at_a_time)

        train_data['observed'] = np.append(train_data['observed'], new_observed)
        outer_pbar.set_postfix(dict(test_acc=test_accuracy, train_acc=train_accuracy))

    return dict(
        train=train_losses,
        test=test_losses,
        observed=train_data['observed']
    )

args = SimpleNamespace()
args.rng_seed = 42
args.batch_size = 64
args.start_n_data = 8
args.train_steps_per_data_sample = 1000
args.sample_n_points_at_a_time = 10
args.total_steps = 100
args.n_networks = 200


if __name__ == "__main__":
    model_dict = create_models(args.n_networks)

    for i in range(1):
        args.rng_seed = i
        fn = f'results/results_big_{i}.pkl'

        try:
            with open(fn, 'rb') as outfile:
                results = pickle.load(outfile)
                assert args == results['args']
                print('Using saved values')
        except Exception:
            results = {
                'args': args,
                'results': {},
            }
        
        
        for name, model in model_dict.items():
            try:
                results['results'][name] = run_experiment(model, args, name)
            except KeyboardInterrupt as e:
                print("Interrupted, skipping", name)
        
        with open(fn, 'wb') as outfile:
            pickle.dump(results, outfile)
