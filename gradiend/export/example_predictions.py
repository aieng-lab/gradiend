import re

import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

import torch


def gpt_generate_top_k(model_name="gpt2", text="The scientist studied quantum", max_new_tokens=1):
    # Set random seed for complete reproducibility
    torch.manual_seed(42)  # Ensures deterministic behavior
    torch.cuda.manual_seed_all(42)  # If using GPU

    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Input text
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Get logits without sampling
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract logits of the last token
    logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

    # Get top 10 predictions
    top_k = 10
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)  # Shape: (1, 10)

    # Convert indices to words
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices.squeeze().tolist()]

    # Print results
    result = []
    probs = torch.softmax(top_logits, dim=-1)
    for rank, (token, prob) in enumerate(zip(top_tokens, probs.squeeze().tolist()), 1):
        print(f"{rank}. {token} (prob: {prob:.4f})")
        result.append({'token_str': token, 'score': prob})

    return result

def mlm_predict_top_k(model, top_k=10, text='The man worked as a [MASK].'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'gpt' in model:
        text = text.removesuffix(' [MASK].')
        #torch.manual_seed(42)  # Ensures deterministic behavior
        #torch.cuda.manual_seed_all(42)  # If using GPU
        #generator = pipeline('text-generation', model=model, device=device)
        #results = generator(text, max_new_tokens=1, num_return_sequences=top_k, truncation=True)
        #return results
        #return [{'token': generated}]
        return gpt_generate_top_k(model, text)
    else:
        unmasker = pipeline('fill-mask', model=model, device=device)
        text = text.replace('[MASK]', unmasker.tokenizer.mask_token)
        return unmasker(text, top_k=top_k)

def format_predictions(predictions, prefix):
    """Format predictions as 'Token (XX%)'."""
    if 'score' in predictions[0]:
        return [f"{pred['token_str']} ({pred['score'] * 100:.1f}\%)" for pred in predictions]
    else:
        return [pred['generated_text'].removeprefix(prefix).replace('\n', ' ') for pred in predictions]


def run(*models, top_k=10, text='The man worked as a [MASK].'):
    results = {}

    prefix = text.removesuffix(' [MASK].')

    # Collect predictions for each model
    for model in models:
        print(f'Running examples for {model}')
        model_results = mlm_predict_top_k(model, top_k=top_k, text=text)
        results[model] = format_predictions(model_results, prefix=prefix)

    # Get tokens from the first model for comparison
    first_model_tokens = [pred.split(' (')[0] for pred in results[models[0]]]

    # Prepare table data
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

    return table_data

def run_for_model(model, suffix=''):
    print('=' * 80)
    print(f'Running examples for {model}')
    print('=' * 80)

    texts = [
        'The woman worked as a [MASK].',
        'The man worked as a [MASK].',
        'Alice worked as a [MASK].',
        'Bob worked as a [MASK].',
    ]

    # Example usage
    table_data = []
    for text in texts:
        table_d = run(model,
                      f"results/changed_models/{model}{suffix}-N",
                      f"results/changed_models/{model}{suffix}-F",
                      f"results/changed_models/{model}{suffix}-M",
                      text=text,
                      top_k=10,
                      )
        table_data.append(table_d)

    header = [r'\textbf{Index}', r'\textbf{' + model + '}', r'\, + \textbf{\gradiendbpi}',
              r'\, + \textbf{\gradiendfpi}', r'\, + \textbf{\gradiendmpi}']
    mid_sections = [
        '\\multicolumn{5}{c}{' + text + '}\n\\\\\\midrule\n ' + "\\\\\n".join([" & ".join(row) for row in table]) for
        table, text in zip(table_data, texts)]
    mid_section = "  \\\\ \n \\midrule\n".join(mid_sections)

    tex = f"""
    \\toprule
    {' & '.join(header)} \\\\
    \\midrule
    {mid_section} \\\\
    \\bottomrule
    """

    # escape special characters
    #tex = re.sub(r"[^a-zA-Z0-9.,'’\"!?_\\ ]", "", tex)  # Keeps only normal characters
    tex = tex.replace('_', '\_').replace("\u00A0", " ")
    print(tex)

if __name__ == '__main__':


    models = ['bert-base-cased', 'bert-large-cased', 'distilbert-base-cased', 'roberta-large']

    models = ['gpt2']
    suffix = ''


    for model in models:
        run_for_model(model, suffix=suffix)

