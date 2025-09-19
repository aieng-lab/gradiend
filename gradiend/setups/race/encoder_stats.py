import pandas as pd

from gradiend.export import models as pretty_models
from gradiend.setups.race.training import WhiteBlackSetup, WhiteAsianSetup, BlackAsianSetup, ChristianJewishSetup, \
    ChristianMuslimSetup, MuslimJewishSetup
from gradiend.util import init_matplotlib

base_models = [
    'bert-base-cased',
    'bert-large-cased',
    'distilbert-base-cased',
    'roberta-large',
    'gpt2',
    'Llama-3.2-3B',
    'Llama-3.2-3B-Instruct',
]

gradiend_suffixes = [
    '-v5'
]


metrics = [
    'corr_black<->white',
    'corr_asian<->white',
    'corr_asian<->black',
    'corr_train',
    'mean_abs_encoded_value_GENEUTRAL',
    'corr',
    #'corr_christian<->jewish'
]

setups = [
    WhiteBlackSetup(),
    WhiteAsianSetup(),
    BlackAsianSetup(),
    #ChristianJewishSetup(),
    #ChristianMuslimSetup(),
    #MuslimJewishSetup()
]


init_matplotlib(use_tex=True)


def plot_split_violin_grid(models, setups, output=None, fig_size=None):
    """
    Plot a grid of split violins: models × variants.
    results_dict = {model: {variant: df}}
    """
    import matplotlib.pyplot as plt
    import os

    n_rows = len(models)
    n_cols = len(setups)

    single_model = n_rows == 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=fig_size or (4 * n_cols, 2 * n_rows - 5.2),
                             sharey=True,
                             squeeze=False,
                             )

    label_size = 16 if single_model else 12
    all_train_handles, all_pair_handles, all_dot_handles = [], [], []

    for i, model in enumerate(models):
        for j, setup in enumerate(setups):
            ax = axes[i, j]

            encoded_values = f'results/models/{setup.id}/{model}-v5/encoded_values.csv'
            try:
                df = pd.read_csv(encoded_values)
                if setup.id in ['religion_christian_muslim', 'religion_muslim_jewish', 'race_white_black']:
                    df['encoded'] *= -1 # standardize
                    df['label'] *= -1

                train_h, pair_h, dot_h = setup.plot_split_violin_ax(df, ax=ax)
                if not all_train_handles:
                    all_train_handles = train_h
                    all_pair_handles = pair_h
                    all_dot_handles = [dot_h]
            except Exception as e:
                print(f"Error plotting {model} with setup {setup.id}: {e}")

            # Row labels (models)
            if j == 0:
                ax.set_ylabel('$h$', fontsize=label_size, rotation=90, labelpad=8, va="center")

            ax.grid(True, zorder=0)

            # Column labels (variants)
            if i == 0:
                gradiend_model = rf"\ensuremath{{\gradiend_\text{{{setup.pretty_id}}}}}"
                ax.set_title(setup.pretty_id, fontsize=16)

    if len(models) > 1:
        for i, model in enumerate(models):
            pretty_model = pretty_models.get(model, model)
            axes[i, 0].annotate(pretty_model,
                                xy=(0, 0.5), xycoords='axes fraction',
                                ha='right', va='center', fontsize=17,
                                rotation=0, annotation_clip=False,
                                xytext=(-50, 0), textcoords='offset points')

    for j in range(n_cols):
        for i in range(n_rows -1):
            axes[i, j].set_xticklabels([])

    # set x tick label size for bottom row
    for j in range(n_cols):
        axes[n_rows - 1, j].tick_params(axis='x', labelsize=label_size)

    # set y tick label size for left column
    y_tick_labels = [-1, -0.5, 0, 0.5, 1]
    for i in range(n_rows):
        axes[i, 0].set_yticks(y_tick_labels)
        axes[i, 0].tick_params(axis='y', labelsize=label_size)
        if i != n_rows - 1:
            axes[i, 0].set_xticklabels([])


    # One global legend at top
    handles = all_pair_handles + all_train_handles + all_dot_handles
    labels = [h.get_label() for h in handles]
    legend_upper = 1.2 if single_model else 1.01
    fig.legend(handles,
               labels,
               loc="upper center",
               bbox_to_anchor=(0.5, legend_upper),
               ncol=6,
               frameon=True,
               fontsize=14 if single_model else 11,
               )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved joint plot to {output}")

    plt.show()


def plot_single_models():
    fig_size = (15, 2.5)
    for model in base_models:
        setup = WhiteBlackSetup()
        plot_split_violin_grid([model],
                               [BlackAsianSetup(), WhiteAsianSetup(), WhiteBlackSetup()],
                               output=f'img/race_encoded_values_{model}.pdf',
                               fig_size=fig_size,
                               )

        plot_split_violin_grid([model],
                               [ChristianJewishSetup(), ChristianMuslimSetup(), MuslimJewishSetup()],
                               output=f'img/religion_encoded_values_{model}.pdf',
                               fig_size=fig_size,
                               )

def plot_all_models():
    plot_split_violin_grid(base_models,
                           [BlackAsianSetup(), WhiteAsianSetup(), WhiteBlackSetup()],
                           output='img/race_encoded_values.pdf'
                           )


    plot_split_violin_grid(base_models,
                           [ChristianJewishSetup(), ChristianMuslimSetup(), MuslimJewishSetup()],
                           output='img/religion_encoded_values.pdf'
                           )


def print_selected_model_stats():
    table_data = []

    for base_model in base_models:
        for setup in setups:
            for gradiend_suffix in gradiend_suffixes:
                try:
                    encoded_values = f'results/models/{setup.id}/{base_model}{gradiend_suffix}/encoded_values.csv'

                    plot_output = f'img/encoder/{setup.id}/{base_model}.pdf'
                    model_metrics = setup.get_model_metrics(encoded_values, plot=False, plot_output=plot_output)
                    metrics_data = []
                    for metric in metrics:
                        metric_value = model_metrics.get(metric)
                        metrics_data.append(f"{metric_value:.4f}" if metric_value is not None else "N/A")
                    table_data.append([base_model, setup.id] + metrics_data)
                except Exception as e:
                    print(f"Error processing {base_model} with setup {setup.id}: {e}")
                    metrics_data = ["Error"] * len(metrics)
                    table_data.append([base_model, setup.id] + metrics_data)

    # Print the table with tabulate
    header = ["Base Model", "Setup"] + list(metrics)
    from tabulate import tabulate
    print(tabulate(table_data, headers=header, tablefmt="grid"))

if __name__ == '__main__':
    #print_selected_model_stats()
    plot_single_models()
    #plot_all_models()
