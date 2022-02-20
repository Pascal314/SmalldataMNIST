import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils

def plot(results, filename, match=None):
    if match == None:
        match = lambda s: True
    fig, ax = plt.subplots(1, 1)
    colors = [f'C{i}' for i in range(20)]

    results = {name: result for (name, result) in results.items() if match(name)}

    for i, (name, result) in enumerate(results.items()):
        ax.plot(result['train'], color=colors[i], linestyle=':')
        ax.plot(result['test'], color=colors[i], label=f'{name}: test')
    ax.set_xlabel('timesteps')
    ax.set_ylabel('accuracy')
    ax.grid()
    plot_full_dataset_performance(ax, 'results/full_dataset.pkl', colors)
    legend = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
    plt.savefig(filename, bbox_inches='tight')

def plot_full_dataset_performance(ax, results_file, colors):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    results = results['results']
    for i, (name, result) in enumerate(results.items()):
        ax.axhline(result['train'], color=colors[i], linestyle=':')
        ax.axhline(result['test'], color=colors[i])


# Idea: display the most sampled image for each model. Is it a very weird borderline case?
# Does it always sample e.g. zeros?
def counts(results, match=None, n=10):
    if match == None:
        match = lambda s: True

    dataset, _ = utils.load_datasets(0)

    results = {name: result for (name, result) in results.items() if match(name)}
    fig, ax = plt.subplots(len(results.keys()), n)

    for i, (name, result) in enumerate(results.items()):
        # ax.plot(result['train'], color=colors[i], linestyle=':')
        # ax.plot(result['test'], color=colors[i], label=f'{name}: test')
        unique, counts = np.unique(result['observed'], return_counts=True)
        argsorted = np.argsort(counts)[::-1]
        ax[i, 0].set_ylabel(name)
        for j in range(n):
            ax[i, j].imshow(dataset['image'][unique[argsorted[j]]].squeeze())
            ax[i, j].xaxis.set_ticklabels([])
            ax[i, j].yaxis.set_ticklabels([])
            ax[i, j].xaxis.set_ticks([])
            ax[i, j].yaxis.set_ticks([])


    plt.savefig('commonly_sampled_.png')





if __name__ == "__main__":
    with open('results/third_results.pkl', 'rb') as f:
        results = pickle.load(f)

    match = lambda name: ('random' in name) or ('ent' in name)
    plot(results['results'], 'mnist_4.png', match)

    for (name, result) in (results['results'].items()):
        print(f"{name}: {np.unique(result['observed']).shape[0]}")

    counts(results['results'], match, n=10)
