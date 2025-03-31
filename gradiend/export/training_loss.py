import matplotlib.pyplot as plt
import json
import os
import numpy as np

from gradiend.util import init_matplotlib
from gradiend.export import models as model_names

def plot_training_metrics(*models, show_means=True):
    init_matplotlib(use_tex=True)

    colors = ["b", "g", "r", "c", "m", "y", "k"]  # Predefined color cycle
    markers = ['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd', '<', '>', 'h', 'H', '+', 'x', '|', '_']
    marker_size = 3

    num_rows, num_cols = (2, 2) if show_means else (1, 2)
    if show_means:
        plt.figure(figsize=(12, 10))
    else:
        plt.figure(figsize=(12, 5))

    x_label = 'Training Steps'

    plt.subplot(num_rows, num_cols, 1)
    plt.title("MSE Loss")
    plt.xlabel(x_label)
    plt.ylabel("Loss")

    plt.subplot(num_rows, num_cols, 2)
    plt.title(r"\cormfval")
    plt.xlabel(x_label)
    plt.ylabel("Score")

    if show_means:
        plt.subplot(2, 2, 3)
        plt.title("Male Means")
        plt.xlabel(x_label)
        plt.ylabel("Mean Value")

        plt.subplot(2, 2, 4)
        plt.title("Female Means")
        plt.xlabel(x_label)
        plt.ylabel("Mean Value")

    for i, model in enumerate(models):
        config_file = os.path.join(model, 'config.json')
        label = os.path.basename(model)
        label = model_names.get(label, label)

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            training = config.get('training', {})
            losses = training.get('losses', [])

            scores = training.get('scores', [])

            male_means = training.get('mean_males', [])
            female_means = training.get('mean_females', [])

            n_evaluation = training.get('n_evaluation', 1)
            x = [n_evaluation * i + 1 for i in range(len(losses))]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            if losses and scores:
                plt.subplot(num_rows, num_cols, 1)
                plt.plot(x, losses, label=label, color=color, marker=marker, markersize=marker_size)
                plt.subplot(num_rows, num_cols, 2)
                plt.plot(x, scores, label=label, color=color, marker=marker, markersize=marker_size)

                print(f'Change in Loss for {label}: ', losses[-1] / losses[0])

            if show_means:
                if male_means:
                    plt.subplot(2, 2, 3)
                    plt.plot(x, male_means, label=label, color=color, marker=marker, markersize=marker_size)
                if female_means:
                    plt.subplot(2, 2, 4)
                    plt.plot(x, female_means, label=label, color=color, marker=marker, markersize=marker_size)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {model}: {e}")

    for i in range(1, num_rows * num_cols + 1):
        plt.subplot(num_rows, num_cols, i)
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_training_metrics('results/models/bert-base-cased', 'results/models/bert-large-cased',
                          'results/models/roberta-large', 'results/models/distilbert-base-cased', 'results/models/gpt2',
                          show_means=False)