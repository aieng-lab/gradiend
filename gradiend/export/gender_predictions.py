import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from gradiend.util import init_matplotlib
from gradiend.export import models as model_mapping


def plot_all(*data, x_key='male_prob_mean', y_key='female_prob_mean', suffix=''):
    dfs = []
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        df['base_model'] = (d.removeprefix('results/gender_prediction/')
                            .removesuffix('.csv')
                            .removesuffix(suffix)
                            .removesuffix('-N')
                            .removesuffix('-M')
                            .removesuffix('-F')
                            .removesuffix('-vFinal'))
        dfs.append(df)
    df = pd.concat(dfs)

    def type_mapper(x):
        # map the type of the model, -N -> neutral, ...
        if '-N' in x:
            return r'\gradiendbpi'
        elif '-M' in x:
            return '\gradiendmpi'
        elif '-F' in x:
            return '\gradiendfpi'
        else:
            return 'Base Model'

    df['type'] = df['model'].apply(type_mapper)


    # Define color and marker mappings
    color_mapper = {
        'Base Model': 'black',
        r'\gradiendbpi': 'blue',
        r'\gradiendmpi': 'red',
        r'\gradiendfpi': 'green',
    }
    marker_mapper = {
        'train': 'o',
        'val': '^',
        'test': 's',
    }

    df['model_pretty_name'] = df['base_model'].apply(lambda x: model_mapping[x])

    df = df[df['split'] != 'total']

    x_y_min = min(df[x_key].min(), df[y_key].min())
    x_y_max = max(df[x_key].max(), df[y_key].max())
    init_matplotlib(use_tex=True)
    fig, axes = plt.subplots(1, len(data), figsize=(16, 4), sharey=True)
    if len(data) == 1:
        axes = [axes]

    # Iterate over each base model and plot on respective axes
    for base_model, base_model_df in df.groupby('base_model'):
        index = list(model_mapping.keys()).index(base_model)
        ax = axes[index]

        # Scatter plot
        for split, sub_df in base_model_df.groupby('split'):
        #for split, sub_df in base_model_df.groupby('type'):
            ax.scatter(
                sub_df[x_key],
                sub_df[y_key],
                label=split,
                c=sub_df['type'].map(color_mapper),
                #c=sub_df['split'].map(color_mapper),
                marker=marker_mapper[split],
                #marker=marker_mapper[split],
                s=30,
            )

        #x_y_min = min(sub_df[x_key].min(), sub_df[y_key].min())
        #x_y_max = max(sub_df[x_key].max(), sub_df[y_key].max())

        # Add identity line
        ax.plot([x_y_min, x_y_max], [x_y_min, x_y_max], color='gray', linestyle='--')

        # Set titles and labels
        ax.set_title(model_mapping[base_model], fontsize=16)
        ax.set_xlabel(r'$\mathbb{P}(man)$', fontsize=14)
        ax.set_ylabel(r'$\mathbb{P}(woman)$', fontsize=14)

        ticks = np.arange(0.0, x_y_max + 0.01, 0.1)  # Generate ticks from 0.0 to 0.9
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:.1f}' for tick in ticks])
        ax.tick_params(axis='y', labelleft=True)

        ticks = np.arange(0.0, x_y_max + 0.01, 0.1)  # Generate ticks from 0.0 to 0.9
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{tick:.1f}' for tick in ticks])


        ax.grid(True, linestyle='--', alpha=0.6)

    marker_legend = ax.legend(
        handles=[
                    Line2D([0], [0], marker='', color='w', label='Split')  # Add title as first entry
            ] + [Line2D([0], [0], marker=style, color='w', markerfacecolor='gray', markersize=10, label=marker)
            for marker, style in marker_mapper.items()
        ],
        loc="upper right",
        bbox_to_anchor=(-2.1, 1.28),
        fontsize=12,
        ncol=4,
    )
    ax.add_artist(marker_legend)

    # Add a separate legend for colors (types)
    color_legend_handles = [
        Line2D([0], [0], marker='', color='w', label='Debiasing Type')  # Add title as first entry
        ] + [
        Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8, label=type_label)
        for type_label, color in color_mapper.items()
    ]
    color_legend = ax.legend(
        handles=color_legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.8, 1.28),
        fontsize=12,
        ncol=5,
    )
    ax.add_artist(color_legend)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the legend
    output_file = f'img/gender_predictions{suffix}.pdf'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()

if __name__ == '__main__':
    # run analyze_decoder.evaluate_all_gender_predictions() before to generate the data!

    suffix = ''

    for suffix in ['_woman_man', '_man_woman']:
        plot_all(
            f'results/gender_prediction/bert-base-cased{suffix}.csv',
                 f'results/gender_prediction/bert-large-cased{suffix}.csv',
                 f'results/gender_prediction/distilbert-base-cased{suffix}.csv',
                 f'results/gender_prediction/roberta-large{suffix}.csv',
                 #f'results/gender_prediction/gpt2{suffix}.csv',
                 #f'results/gender_prediction/Llama-3.2-3B{suffix}.csv',
                 #f'results/gender_prediction/Llama-3.2-3B-Instruct{suffix}.csv',
                 suffix=suffix,
                 )