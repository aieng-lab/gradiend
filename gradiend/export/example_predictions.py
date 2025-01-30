from transformers import pipeline
from tabulate import tabulate

def mlm_predict_top_k(model, top_k=10, text='The man worked as a [MASK].'):
    unmasker = pipeline('fill-mask', model=model)
    text = text.replace('[MASK]', unmasker.tokenizer.mask_token)
    return unmasker(text, top_k=top_k)

def format_predictions(predictions):
    """Format predictions as 'Token (XX%)'."""
    return [f"{pred['token_str']} ({pred['score'] * 100:.1f}\%)" for pred in predictions]


def run(*models, top_k=10, text='The man worked as a [MASK].'):
    results = {}

    # Collect predictions for each model
    for model in models:
        print(f'Running examples for {model}')
        model_results = mlm_predict_top_k(model, top_k=top_k, text=text)
        results[model] = format_predictions(model_results)

    # Get tokens from the first model for comparison
    first_model_tokens = [pred.split(' (')[0] for pred in results[models[0]]]

    # Prepare table data
    headers = ["Prediction"] + list(models)
    table_data = []
    for i in range(top_k):
        row = [f"{i + 1}"]
        for model_index, model in enumerate(models):
            token_entry = results[model][i]
            token = token_entry.split(' (')[0]  # Extract the token part
            if model_index > 0 and token not in first_model_tokens:
                token_entry = f"\\textbf{{{token_entry.strip()}}}"  # Highlight if not in first model
            row.append(token_entry)
        table_data.append(row)

    # Generate LaTeX table using tabulate
    latex_table = tabulate(table_data, headers=headers, tablefmt="latex_raw")

    # Print LaTeX table
    print("\nGenerated LaTeX Table:\n")
    print(latex_table)

    return table_data



if __name__ == '__main__':
    texts = [
        'The woman worked as a [MASK].',
        'The man worked as a [MASK].',
        'Alice worked as a [MASK].',
        'Bob worked as a [MASK].',
    ]

    models = ['bert-base-cased', 'bert-large-cased', 'distilbert-base-cased', 'roberta-large']
    suffix = '-vFinal'
    suffix = ''


    for model in models:
        print('=' * 80)
        print(f'Running examples for {model}')
        print('=' * 80)

        # Example usage
        table_data = []
        for text in texts:
            table_d = run(model,
                f"results/changed_models/{model}{suffix}-N",
                f"results/changed_models/{model}-vFinal-F",
                f"results/changed_models/{model}-vFinal-M",
                text=text,
                top_k=10,
                )
            table_data.append(table_d)

        header = [r'\textbf{Index}', r'\textbf{' + model + '}', r'\, + \textbf{\gradiendbpi}', r'\, + \textbf{\gradiendfpi}', r'\, + \textbf{\gradiendmpi}']
        mid_sections = ['\\multicolumn{5}{c}{' + text + '}\n\\\\\\midrule\n ' +"\\\\\n".join( [" & ".join(row) for row in table]) for table, text in zip(table_data, texts)]
        mid_section = "  \\\\ \n \\midrule\n".join(mid_sections)

        tex = f"""
        \\toprule
        {' & '.join(header)} \\\\
        \\midrule
        {mid_section} \\\\
        \\bottomrule
        """

        print(tex)


