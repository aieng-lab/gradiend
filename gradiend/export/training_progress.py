import matplotlib.pyplot as plt
import json
import os
import numpy as np
from matplotlib.lines import Line2D
from gradiend.util import init_matplotlib
from gradiend.export import models as model_names

def plot_training_metrics(*models, show_means=True):
    init_matplotlib(use_tex=True)

    colors = ["b", "g", "r", "c", "y", "k"]  # Predefined color cycle
    markers = ['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd', '<', '>', 'h', 'H', '+', 'x', '|', '_']
    marker_size = 3

    num_rows, num_cols = (1, 4) if show_means else (1, 2)
    if show_means:
        plt.figure(figsize=(12, 3))
    else:
        plt.figure(figsize=(12, 5))

    x_label = 'Training Steps'

    plt.subplot(num_rows, num_cols, 1)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel("MSE Loss")

    plt.subplot(num_rows, num_cols, 2)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(r"\cormfval")

    if show_means:
        plt.subplot(1, 4, 3)
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel("$h_M$")

        plt.subplot(1, 4, 4)
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel("$h_F$")

    # Define the plotting order for models
    order = ['roberta', 'bert-base-cased', 'bert-large-cased', 'distilbert-base-cased', 'gpt2']
    lines = []

    # Plot lines
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
                # Plot Loss and Score
                line_loss, = plt.subplot(num_rows, num_cols, 1).plot(x, losses, label=label, color=color, marker=marker, markersize=marker_size)
                line_score, = plt.subplot(num_rows, num_cols, 2).plot(x, scores, label=label, color=color, marker=marker, markersize=marker_size)

                # Store lines for legend later
                lines.append(Line2D([0], [0], color=color, marker=marker, label=label))

                if show_means:
                    if male_means:
                        plt.subplot(1, 4, 3).plot(x, male_means, label=label, color=color, marker=marker, markersize=marker_size)
                    if female_means:
                        plt.subplot(1, 4, 4).plot(x, female_means, label=label, color=color, marker=marker, markersize=marker_size)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {model}: {e}")

    # Create a single custom legend on top, centered above all subplots
    plt.legend(handles=lines, loc='center', bbox_to_anchor=(0.5, 1.05), ncol=6, title='Models')

    # Adjust layout to ensure everything fits nicely
    plt.tight_layout()
    plt.savefig('img/training_progress.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_training_metrics('results/experiments/model_with_gradiend/meta-llama/Llama-3.2-3B-Instruct/0')

    exit(1)
    plot_training_metrics('results/models/bert-base-cased', 'results/models/bert-large-cased',
                          'results/models/roberta-large', 'results/models/distilbert-base-cased', 'results/models/gpt2',
                          show_means=False)
