import haiku as hk
import jax
import jax.numpy as jnp

class ConvNet(hk.Module):
    def __init__(self, n_channels, n_linear):
        super().__init__()
        self.n_channels = n_channels
        self.n_linear = n_linear

    def __call__(self, x):
        y = hk.Conv2D(self.n_channels, (4, 4), stride=2)(x)
        y = jax.nn.relu(y)
        y = hk.Conv2D(self.n_channels, (4, 4), stride=2)(y)
        y = jax.nn.relu(y)
        y = y.reshape(x.shape[0], -1)
        y = hk.Linear(self.n_linear)(y)
        y = jax.nn.relu(y)
        return y

def preprocess(batch):
    batch['image'] = tf.cast(batch['image'], jnp.float32) / 255 - 0.5
    return batch

if __name__ == "__main__":

    import tensorflow as tf
    tf.config.set_visible_devices([], device_type='GPU')
    import tensorflow_datasets as tfds
    import optax
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt

    ds = tfds.load("mnist", split='train').shuffle(10000).batch(64).repeat().map(preprocess)
    ds = iter(tfds.as_numpy(ds))

    test_ds = tfds.load("mnist", split='test').batch(256).repeat().map(preprocess)
    test_ds = iter(tfds.as_numpy(test_ds))

    def forward(x):
        y = ConvNet(16, 32)(x)
        y = hk.Linear(10)(y)
        return y

    net = hk.without_apply_rng(hk.transform(forward))
    params = net.init(jax.random.PRNGKey(42), next(ds)['image'])

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    print(hk.experimental.tabulate(net)(next(ds)['image']))
    
    @jax.jit
    def loss_fn(params, batch):
        labels = batch['label']
        logits = net.apply(params, batch['image'])
        return jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def evaluate(params, batch):
        logits = net.apply(params, batch['image'])
        pred_labels = jnp.argmax(logits, axis=-1)
        return jnp.mean(pred_labels == batch['label'])


    losses = []
    accuracy = []
    n_test = 10
    n_steps = 1000
    pbar = tqdm(range(n_steps))
    for i in pbar:
        batch = next(ds)
        params, opt_state, loss = train_step(params, opt_state, batch)
        losses.append(loss)
        if i % n_test == 0:
            accuracy.append(evaluate(params, next(test_ds)))

        pbar.set_postfix(dict(
            loss=np.mean(losses[-20:]),
            acc=accuracy[-1]
        ))
    
        
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(np.arange(n_steps), losses)
    ax2.plot(np.arange(0, n_steps, n_test), accuracy)
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('test_accuracy')
    ax2.set_xlabel('training steps')
    ax1.grid()
    ax2.grid()
    plt.show()