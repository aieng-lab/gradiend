import os

from transformers import AutoModel

models = [
    'bert-base-cased',
    'bert-large-cased',
    'distilbert-base-cased',
    'roberta-large',
    'gpt2',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-3B-Instruct',
]

diffs = {}

for model in models:

    model_id = model.split('/')[-1]
    base_model = AutoModel.from_pretrained(model)
    gradiend_model_id = f'aieng-lab/{model_id}-gradiend-gender-debiased'
    gradiend_model = AutoModel.from_pretrained(gradiend_model_id)

    # compute the mean absolute weight difference
    weights_base = base_model.state_dict()
    weights_gradiend = gradiend_model.state_dict()
    weight_diff = 0.0
    for key in weights_base.keys():
        if key in weights_gradiend:
            diff = (weights_base[key] - weights_gradiend[key]).abs().mean().item()
            weight_diff += diff
        else:
            print(f"Key {key} not found in gradiend model.")

    weight_diff /= len(weights_base)
    print(f"Model: {model_id}, Mean Absolute Weight Difference: {weight_diff:.6f}")
    diffs[model_id] = weight_diff

# Sort the models by weight difference
sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
print("\nSorted Models by Weight Difference:")
for model_id, diff in sorted_diffs:
    print(f"{model_id}: {diff:.6f}")

# plot the differences
import matplotlib.pyplot as plt
model_ids = [x[0] for x in sorted_diffs]
weight_diffs = [x[1] for x in sorted_diffs]
plt.figure(figsize=(12, 6))
plt.barh(model_ids, weight_diffs, color='skyblue')
plt.xlabel('Mean Absolute Weight Difference')
plt.title('Weight Differences Between Base and Gradiend Models')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('weight_differences.png')
plt.show()


output_dir = 'img/model_diffs'
os.makedirs(output_dir, exist_ok=True)

for model in models:
    model_id = model.split('/')[-1]
    print(f"\nProcessing model: {model_id}")

    try:
        base_model = AutoModel.from_pretrained(model)
        gradiend_model_id = f'aieng-lab/{model_id}-gradiend-gender-debiased'
        gradiend_model = AutoModel.from_pretrained(gradiend_model_id)

        weights_base = base_model.state_dict()
        weights_gradiend = gradiend_model.state_dict()

        diffs = {}
        for key in weights_base:
            if key in weights_gradiend:
                diff = (weights_base[key] - weights_gradiend[key]).abs().mean().item()
                diffs[key] = diff
            else:
                print(f"Key {key} not found in Gradiend model.")

        # Sort by difference for better readability
        sorted_items = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
        param_names = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]

        # Plot
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(values)), values, tick_label=param_names)
        plt.xticks(rotation=90, fontsize=6)
        plt.ylabel("Mean Absolute Weight Difference")
        plt.title(f"Parameter-Wise Weight Differences: {model_id}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_id}_param_diffs.png")
        plt.show()
        plt.close()
        print(f"Saved plot for {model_id}")

    except Exception as e:
        print(f"Error with model {model_id}: {e}")


