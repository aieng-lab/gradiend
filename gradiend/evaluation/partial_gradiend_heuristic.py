import os

import torch
from matplotlib import pyplot as plt

from gradiend.model import ModelWithGradiend
from gradiend.data import read_genter, read_namexact

def millify(n, precision=1):
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{precision}f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    return str(n)


# returns how good the heuristic_mask is to approximate the layer_mask
# returns a score in [0, 1]
# only heuristic_mask == layer_mask evaluates to 1
# if the heuristic_mask misses masked entries, the recall and thus the final score is penalized
# if the heuristic_mask has more masked entries than the layer_mask (which may simplify reaching a good recall), the final score is penalized
# a complete mask (all entries are masked) results in the top_k proportion
def get_score(heuristic_mask, layer_mask):
    heuristic_concat = torch.concat([heuristic_mask[layer].flatten() for layer in heuristic_mask], dim=0)
    layer_concat = torch.concat([layer_mask[layer].flatten() for layer in layer_mask], dim=0)

    n_heuristic = torch.sum(heuristic_concat)
    n_layer = torch.sum(layer_concat)

    recall = torch.sum(heuristic_concat * layer_concat) / n_layer
    recall = recall.item()

    precision = torch.sum(heuristic_concat * layer_concat) / n_heuristic
    precision = precision.item()

# todo or just use f1?

    #f1 = 2 * (precision * recall) / (precision + recall)
    prop = n_heuristic.item() / len(heuristic_concat)

    def get_f_score(beta):
        if precision + recall == 0:
            return 0
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    return {
        'recall': recall,
        'prop': prop,
        'recall * (1 - prop)': recall * (1 - prop),
        'precision': precision,
        'f1': get_f_score(1),
        'f10': get_f_score(10),
        'f0.1': get_f_score(0.1),
    }


def get_heuristic_mask(model, *thresholds, n_heuristic_samples=10, n_heuristic_names=5):
    if isinstance(model, str):
        gradiend = ModelWithGradiend.from_pretrained(model)
    else:
        gradiend = model

    genter = read_genter(split='train')

    genter = genter.head(n_heuristic_samples)
    names = read_namexact(split='train')

    # keep only these names per gender and drop the rest
    names = names.groupby('gender', group_keys=False).apply(lambda x: x.head(n_heuristic_names)).reset_index(drop=True)

    heuristic_layer_maps = {}
    heuristic_n_neurons = {}
    all_gradients = []
    for _, entry in genter.iterrows():
        raw_text = entry['masked']
        genter_gradients = []
        for _, name_entry in names.iterrows():
            name = name_entry['name']
            gender = name_entry['gender']

            filled_text = raw_text.replace('[NAME]', name).replace('[PRONOUN]', '[MASK]')
            label = 'he' if gender == 'M' else 'she'
            gradients = gradiend.create_gradients(filled_text, label=label, return_dict=True)
            genter_gradients.append(gradients)
        genter_mean_gradients = {k: torch.mean(torch.stack([v[k] for v in genter_gradients]), dim=0) for k in genter_gradients[0].keys()}
        all_gradients.append(genter_mean_gradients)

    mean_gradients = {k: torch.mean(torch.stack([v[k] for v in all_gradients]), dim=0) for k in all_gradients[0].keys()}

    for threshold in thresholds:
        heuristic_mask = {k: v.abs() > threshold for k, v in mean_gradients.items()}
        heuristic_layer_maps[threshold] = heuristic_mask
        heuristic_n_neurons[threshold] = sum(torch.sum(v).item() for k, v in heuristic_mask.items())

    if len(thresholds) == 1:
        return heuristic_layer_maps[thresholds[0]], heuristic_n_neurons[thresholds[0]]

    return heuristic_layer_maps, heuristic_n_neurons

def evaluate(model='results/models/bert-base-cased', n_heuristic_samples=100, n_heuristic_names=10, top_k=1e-7):
    thresholds = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
    gradiend = ModelWithGradiend.from_pretrained(model)

    heuristic_layer_maps, heuristic_n_neurons = get_heuristic_mask(model, *thresholds, n_heuristic_samples=n_heuristic_samples, n_heuristic_names=n_heuristic_names)

    scores = {}

    base_model = gradiend.base_model
    n = sum(layer.numel() for layer in base_model.parameters())

    for top_k in [top_k]:
        layer_mask = gradiend.get_layer_mask(top_k, part='decoder-bias')
        top_k_scores = {}

        for heuristic, heuristic_layer in heuristic_layer_maps.items():
            score = get_score(heuristic_layer, layer_mask)
            top_k_scores[heuristic] = score

        scores[top_k] = top_k_scores

    print(scores)

    # test different heuristics to approximate layer_masks
    # calculate metric to judge the heuristics
    # metric: recall with penalization for total number of masked elements


    # todo run default_evalution for chosen heuristic!

    labels = list(next(iter(scores[top_k].values())).keys())

    # Initialize the plot
    plt.figure(figsize=(8, 5))

    # Plot each metric separately
    for label in labels:
        score_values = [scores[top_k][t][label] for t in thresholds]
        plt.plot(thresholds, score_values, marker='o', linestyle='-', label=label)

    #labels = [f"{t:.0e} (N={int(t * n)})" for t in thresholds]
    labels = [f"{t:.0e}\nN={millify(heuristic_n_neurons[t], precision=0)}" if i % 2 == 0 else '' for i, t in enumerate(thresholds)]
    plt.xscale('log')
    plt.xticks(thresholds, labels)

    # Labels and title
    plt.xlabel('Threshold (Number of Heuristic Neurons)')
    plt.ylabel('Score')
    model_id = model.split('/')[-1]
    plt.title(model_id)
    plt.ylim(0, 1)
    #plt.xscale('log')

    # Annotate points
    #for label in labels:
    #    score_values = [scores[top_k][t][label] for t in thresholds]
    #    for t, s in zip(thresholds, score_values):
    #        plt.text(t, s, f"{s:.2f}", fontsize=8, ha='right' if t > 0.3 else 'left', va='bottom')

    # Show plot
    plt.legend()
    plt.grid(True)
    output = f'results/img/partial_gradiend_heuristic/{model}.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    plt.show()

if __name__ == '__main__':
    models = [
        'results/models/bert-base-cased',
        #'results/models/bert-large-cased',
        #'results/models/distilbert-base-cased',
        #'results/models/roberta-large',
    ]
    for model in models:
        evaluate(model=model)