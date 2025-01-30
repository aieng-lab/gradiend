import copy

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
import os


def freeze_layers_until_target(model, *target_layer_names):
    assert len(target_layer_names) > 0, 'At least one target layer name must be provided'

    # iterate through the parameters from the model, starting at the input layer
    # we stop iterating as soon as we hit the first target layer
    for name, param in model.named_parameters():
        if name in target_layer_names:
            return

        param.requires_grad = False



def get_activation(activation: str, encoder=False):
    if activation == 'relu':
        activation_fnc = nn.ReLU(inplace=True)
    elif activation == 'leakyrelu':
        activation_fnc = nn.LeakyReLU(inplace=True)
    elif activation == 'tanh':
        activation_fnc = nn.Tanh()
    elif activation == 'smht':
        activation_fnc = nn.Hardtanh()
    elif activation == 'elu':
        activation_fnc = nn.ELU()
    elif activation == 'gelu':
        activation_fnc = nn.GELU()
    elif activation == 'sigmoid':
        activation_fnc = nn.Sigmoid()
    elif activation == 'id':
        if encoder:
            activation_fnc = nn.LayerNorm(1)
        else:
            activation_fnc = nn.Identity()
    else:
        raise ValueError('Unsupported activation function:', activation)
    return activation_fnc



class GradiendModel(nn.Module):
    def __init__(self, input_dim, latent_dim, layers, intermediate=False, activation='relu', bias_decoder=False, decoder_factor=1.0, activation_decoder=None, **kwargs):
        super(GradiendModel, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.intermediate = intermediate
        self.layers = layers
        self.activation = activation.lower()
        self.bias_decoder = bias_decoder
        self.kwargs = kwargs

        activation_fnc = get_activation(self.activation, encoder=True)

        if activation_decoder:
            activation_fnc_decoder = get_activation(activation_decoder)
            self.activation_decoder = activation_decoder
        else:
            activation_fnc_decoder = get_activation(self.activation, encoder=False)
            self.activation_decoder = self.activation

        if intermediate:
            intermediate_size = 16
            # Define the encoder - takes BERT output and compresses to latent dim
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_size),
                activation_fnc,
                nn.Linear(intermediate_size, latent_dim),
            )

            # Define the decoder - expands latent vector back to BERT output size
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, intermediate_size, bias=bias_decoder),
                activation_fnc_decoder,
                nn.Linear(intermediate_size, input_dim, bias=bias_decoder),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                activation_fnc
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim, bias=bias_decoder),
                activation_fnc_decoder
            )

        # Initialize the decoder weights with the same distribution as the encoder weights (up to decoder_factor)
        x = self.encoder[0].weight.max().item() * decoder_factor
        nn.init.uniform_(self.decoder[0].weight, -x, x)
        if bias_decoder:
            nn.init.uniform_(self.decoder[0].bias, -x, x)

        self.avg_gradient_norm = 0.0
        self.ctr = 0

    @property
    def decoder_norm(self):
        return torch.norm(self.decoder[0].weight, p=2).item()

    @property
    def encoder_norm(self):
        return torch.norm(self.encoder[0].weight, p=2).item()


    def extract_gradients(self, bert, return_dict=False):
        layer_map = {k: v for k, v in bert.named_parameters()}
        if return_dict:
            return {layer: layer_map[layer].grad.detach() for layer in self.layers}
        return torch.concat([layer_map[layer].grad.flatten().detach() for layer in self.layers])

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, x):
        orig_shapes = {}

        if hasattr(x, 'named_parameters'): # todo remove?
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            for layer in self.layers:
                param = layer_map[layer]
                grad = param.grad.flatten()
                grads.append(grad)
                orig_shapes[layer] = param.shape
            x = torch.concat(grads)
        elif isinstance(x, dict):
            grads = []
            for layer in self.layers:
                param = x[layer]
                grad = param.flatten()
                grads.append(grad)
                orig_shapes[layer] = param.shape

            x = torch.concat(grads)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        grad_norm = torch.norm(x, p=2).item()
        self.avg_gradient_norm = (self.avg_gradient_norm * self.ctr + grad_norm) / (self.ctr + 1)
        self.ctr += 1

        if orig_shapes:
            decoded_params = {}
            start_idx = 0
            for layer, shape in orig_shapes.items():
                num_elements = shape.numel()
                decoded_params[layer] = decoded[start_idx:start_idx + num_elements].reshape(shape)
                start_idx += num_elements
            decoded = decoded_params

        return decoded

    def forward_encoder(self, x):
        if hasattr(x, 'named_parameters'): # todo remove?
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            for layer in self.layers:
                param = layer_map[layer]
                grad = param.grad.flatten()
                grads.append(grad)

            x = torch.stack(grads)

        encoded = self.encoder(x)
        print('Encoded', encoded)
        return encoded

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        config_path = os.path.join(save_directory, 'config.json')

        # Save model state dictionary
        torch.save(self.state_dict(), model_path)

        self.kwargs.update(kwargs)

        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'intermediate': self.intermediate,
            'layers': self.layers,
            'activation': self.activation,
            'activation_decoder': self.activation_decoder,
            'bias_decoder': self.bias_decoder,
            **self.kwargs,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, load_directory):
        model_path = os.path.join(load_directory, 'pytorch_model.bin')
        config_path = os.path.join(load_directory, 'config.json')

        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)


        # Instantiate the model
        model = cls(**config)

        # Load model state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        model.load_state_dict(state_dict)

        model.name_or_path = load_directory

        return model

    @property
    def grad_iterations(self):
        return self.kwargs.get('grad_iterations', 1)


    def plot(self, fig=None, bins=50, n=None):
        # Initialize lists to store weights and biases
        encoder_weights = []
        decoder_weights = []
        decoder_bias = []

        # Extract weights and biases from encoder and decoder
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                encoder_weights.append(param.flatten().detach().cpu().numpy())
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                decoder_weights.append(param.flatten().detach().cpu().numpy())
            elif 'bias' in name:
                decoder_bias.append(param.flatten().detach().cpu().numpy())

        # Convert lists to numpy arrays
        encoder_weights = np.concatenate(encoder_weights)
        decoder_weights = np.concatenate(decoder_weights)
        decoder_bias = np.concatenate(decoder_bias)

        # Compute bin edges and aggregate values
        encoder_bin_means, encoder_bin_edges, _ = binned_statistic(
            np.arange(len(encoder_weights)), encoder_weights, statistic='mean', bins=bins)
        decoder_bin_means, decoder_bin_edges, _ = binned_statistic(
            np.arange(len(encoder_weights), len(encoder_weights) + len(decoder_weights)),
            decoder_weights, statistic='mean', bins=bins)

        if self.bias_decoder:
            decoder_bias_bin_means, decoder_bias_bin_edges, _ = binned_statistic(
                np.arange(len(encoder_weights) + len(decoder_weights),
                          len(encoder_weights) + len(decoder_weights) + len(decoder_bias)),
                decoder_bias, statistic='mean', bins=bins)

        # Plotting
        if fig is None:
            fig = plt.figure(figsize=(12, 6))
        else:
            fig.clear()

        # Encoder weights (blue)
        plt.scatter(encoder_bin_edges[:-1], encoder_bin_means, color='blue', label='Encoder Weights')

        # Decoder weights (green)
        plt.scatter(decoder_bin_edges[:-1], decoder_bin_means, color='green', label='Decoder Weights')

        plt.title(f'Weights of Encoder and Decoder (Aggregated) {n if n else ""}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return fig


class ModelWithGradiend(nn.Module):

    def __init__(self, bert, ae, tokenizer):
        super().__init__()
        self.bert = bert
        self.ae = ae
        self.tokenizer = tokenizer
        self.grad_iterations = ae.grad_iterations

        self.layer_map = {k: v for k, v in self.bert.named_parameters()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(self.device)
        self.ae.to(self.device)

    @property
    def name(self):
        return os.path.basename(self.ae.name_or_path)

    def encode(self, text, label):
        item = self.tokenizer(text, return_tensors="pt", padding=True)
        item = {k: v.to(self.device) for k, v in item.items()}

        gender_labels = item['input_ids'].clone()
        mask_token_mask = gender_labels == self.tokenizer.mask_token_id
        gender_labels[~mask_token_mask] = -100  # only compute loss on masked tokens
        label_token_id = self.tokenizer(label, add_special_tokens=False)['input_ids']
        if len(label_token_id) == 1:
            label_token_id = label_token_id[0]
        else:
            label_token_id = torch.LongTensor(label_token_id).to(self.device).flatten()
        gender_labels[mask_token_mask] = label_token_id


        item['labels'] = gender_labels

        outputs = self.bert(**item)
        loss_bert = outputs.loss

        self.bert.zero_grad()
        loss_bert.backward()

        gradients = self.ae.extract_gradients(self.bert)

        # print(gradients_flat)
        encoded = self.ae.encoder(gradients).item()

        return encoded


    def modify_bert(self, lr, gender_factor, part='decoder', top_k=None):
        # returns a bert model with enhanced weights based on the auto encoder, use the learning rate parameter to control the influence of the auto encoder
        import copy
        enhanced_bert = copy.deepcopy(self.bert)

        if top_k == 0:
            return enhanced_bert

        layer_map = {k: v for k, v in enhanced_bert.named_parameters()}
        #zero_bias = self.ae.decoder(torch.Tensor([0.0]).to(self.device))
        if part == 'decoder':
            enhancer = self.ae.decoder(torch.Tensor([gender_factor]).to(self.device))
        elif part == 'encoder':
            enhancer = self.ae.encoder[0].weight.flatten().to(self.device)
        else:
            raise ValueError('Unsupported part:', part)

        if top_k is not None and top_k < len(enhancer):
                # we set parts of the enhancer to 0 that are not in the top k highest values (wrt absolute value)
                abs_enhancer = enhancer.abs()
                top_k_values, top_k_indices = abs_enhancer.topk(top_k, sorted=False, largest=True)
                # use first k values as indices
                #top_k_indices = abs_enhancer.argsort(descending=True)[:top_k]
                #top_k_indices = np.arange(top_k)
                mask = torch.zeros_like(enhancer, dtype=torch.bool)
                mask[top_k_indices] = True
                enhancer[~mask] = 0.0
                #assert (enhancer != 0.0).sum() == top_k

        #decoded_inv = self.ae.decoder(torch.Tensor([-gender_factor]).to(self.device))
        idx = 0
        with torch.no_grad():
            for layer in self.ae.layers:
                shape = layer_map[layer].shape
                number_of_elements = shape.numel()
                layer_map[layer] += lr * enhancer[idx:idx + number_of_elements].reshape(shape)
                idx += number_of_elements

        return enhanced_bert

    def enhance_bert(self, *args, **kwargs):
        print('enhance_bert is deprecated, use modify_bert')
        return self.modify_bert(*args, **kwargs)

    def mask_and_encode(self, text, ignore_tokens=False, return_masked_text=False, single_mask=True):
        item = self.tokenizer(text, return_tensors="pt")
        item = {k: v.to(self.device) for k, v in item.items()}

        labels = item['input_ids'].clone()

        if single_mask:
            mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
            # randomly mask a single entry
            mask[0, torch.randint(0, labels.shape[1], (1,))] = True
        else:
            random_mask = torch.rand(labels.shape, dtype=torch.float, device=labels.device) < 0.15
            padding_mask = labels == self.tokenizer.pad_token_id

            if ignore_tokens:
                exclude_mask = (labels.unsqueeze(-1) == torch.Tensor(ignore_tokens).to(labels.device)).any(dim=-1)
            else:
                exclude_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
            mask = random_mask & ~padding_mask & ~exclude_mask
        labels[~mask] = -100  # only compute loss on masked tokens
        item['input_ids'][mask] = self.tokenizer.mask_token_id
        # print('Mask', mask_token_mask.sum())
        item['labels'] = labels

        outputs = self.bert(**item)
        loss_bert = outputs.loss

        self.bert.zero_grad()
        loss_bert.backward()

        gradients = self.ae.extract_gradients(self.bert)

        encoded = self.ae.encoder(gradients).item()

        if return_masked_text:
            masked_str = self.tokenizer.decode(item['input_ids'].squeeze())
            masked_str = masked_str.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            labels = [self.tokenizer.decode(id) for id in item['labels'][mask]]
            return encoded, masked_str, labels

        return encoded

    def create_inputs(self, masked_text, label):
        item = self.tokenizer(masked_text, return_tensors="pt")
        item = {k: v.to(self.device) for k, v in item.items()}
        labels = item['input_ids'].clone()
        labels[labels != self.tokenizer.mask_token_id] = -100  # only compute loss on masked tokens
        label_token_id = self.tokenizer(label, add_special_tokens=False)['input_ids']
        if not len(label_token_id) == 1:
            raise ValueError('Only a single label token is supported')
        label_token_id = label_token_id[0]
        labels[labels == self.tokenizer.mask_token_id] = label_token_id
        item['labels'] = labels
        return item

    def forward_pass(self, inputs, return_dict=False, lr=1e-4): # todo implement batched=True
        # Forward pass through BERT
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        grads = []

        if self.grad_iterations > 1:
            bert = copy.deepcopy(self.bert)
            for i in range(self.grad_iterations):
                outputs = bert(**inputs)
                loss_bert = outputs.loss
                bert.zero_grad()
                loss_bert.backward()
                gradients = self.ae.extract_gradients(bert, return_dict=True)
                grads.append(gradients)

                if i < self.grad_iterations - 1:
                    # perform the training step
                    # Step 6: Update the model's weights

                    with torch.no_grad():
                        for name, param in bert.named_parameters():
                            if param.grad is not None:
                                param.add_(-lr * param.grad)

                # todo save memory for now
                grads = [grads[-1]]

            if return_dict:
                return grads[-1]
            flatten_gradient = torch.concat(tuple(grad.flatten() for grad in grads[-1].values()))
            return flatten_gradient

        else:
            outputs = self.bert(**inputs)
            loss_bert = outputs.loss
            self.bert.zero_grad()
            loss_bert.backward()
            return self.ae.extract_gradients(self.bert, return_dict=return_dict)

    # invert the encoded value, i.e. encoded value * (-1), while keeping the decoders value
    def invert_encoding(self):
        with torch.no_grad():
            self.ae.encoder[0].weight = torch.nn.Parameter(self.ae.encoder[0].weight * -1)
            self.ae.encoder[0].bias = torch.nn.Parameter(self.ae.encoder[0].bias * -1)
            self.ae.decoder[0].weight = torch.nn.Parameter(self.ae.decoder[0].weight * -1)

    def save_pretrained(self, save_directory, **kwargs):
        self.ae.save_pretrained(save_directory, bert=self.bert.name_or_path, tokenizer=self.tokenizer.name_or_path, **kwargs)


    @classmethod
    def from_pretrained(cls, load_directory, *layers, latent_dim=1, **kwargs):
        if len(layers) == 1 and isinstance(layers[0], list):
            layers = layers[0]

        try:
            ae = GradiendModel.from_pretrained(load_directory)

            if layers and ae.layers != layers:
                raise ValueError(f'The provided layers {layers} do not match the layers in the model configuration {ae.layers}')
            else:
                layers = ae.layers

            base_model_id = ae.kwargs['base_model']
            base_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
            tokenizer = ae.kwargs.get('tokenizer', base_model_id)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except FileNotFoundError:
            print('No model with auto encoder found in the specified directory:', load_directory, ' -> creating a new auto encoder')

            if isinstance(load_directory, str):
                base_model = AutoModelForMaskedLM.from_pretrained(load_directory)
                tokenizer = AutoTokenizer.from_pretrained(load_directory)
            else:
                base_model = load_directory
                tokenizer = base_model.tokenizer

            layer_map = {k: v for k, v in base_model.named_parameters()}

            if not layers:
                # no layer provided, i.e. all layers are used that are part of the core model, i.e. all layers that are not part of prediction layers
                layers = [layer for layer in layer_map if 'cls.prediction' not in layer.lower()]

            input_dim = sum([layer_map[layer].numel() for layer in layers])
            ae = GradiendModel(input_dim, layers=layers, latent_dim=latent_dim, base_model=load_directory, **kwargs)

        # freeze all layers that do not require gradient calculations
        freeze_layers_until_target(base_model, *layers)

        model = ModelWithGradiend(base_model, ae, tokenizer)
        model.name_or_path = load_directory
        return model

    def ae_named_parameters(self, part='all'):
        idx = 0
        if part == 'all':
            yield from self.ae.named_parameters()
            return
        elif part == 'encoder':
            layer_map = {k: v for k, v in self.ae.encoder.named_parameters()}
            weights = layer_map['0.weight'].squeeze()

        elif 'decoder' in part:
            layer_map = {k: v for k, v in self.ae.decoder.named_parameters()}
            if part == 'decoder':
                weights = layer_map['0.weight'].squeeze()
            elif part == 'decoder-sum':
                weights = (layer_map['0.weight'] + layer_map['0.bias']).squeeze()
            elif part == 'decoder-bias':
                weights = layer_map['0.bias'].squeeze()
            else:
                raise ValueError('Unsupported part:', part)
        else:
            raise ValueError('Unsupported part:', part)

        for layer in self.ae.layers:
            orig_shape = self.layer_map[layer].shape
            num_elements = orig_shape.numel()
            yield layer, weights[idx:idx + num_elements].reshape(orig_shape)
            idx += num_elements
        if idx != weights.numel():
            raise ValueError(f'Inconsistent number of elements in the weights and expected number of elements in the layers ({idx} vs. {weights.numel()})')


def run_mlm_example(model, tokenizer):
    # Sample text
    text2 = "Anna was working. [MASK] coded in Python."
    text1 = "Ben was working. [MASK] coded in Python."

    def run_for_text(text):
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt')

        # Get the index of the [MASK] token
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        # Forward pass, get predictions
        outputs, ae_outputs = model(**inputs)
        logits = outputs.logits


        # Get the predicted token id
        mask_token_logits = logits[0, mask_token_index, :]
        predicted_token_id = torch.argmax(mask_token_logits, dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)

        print(f"Predicted token: {predicted_token}")

        # Decode the whole sentence with the predicted token
        input_ids = inputs["input_ids"].tolist()[0]
        input_ids[mask_token_index] = predicted_token_id[0].item()
        predicted_sentence = tokenizer.decode(input_ids)

        print(f"Predicted sentence: {predicted_sentence}")
        return ae_outputs

    output1 = run_for_text(text1)
    output2 = run_for_text(text2)

    print((output1 - output2).abs().sum().item())

if __name__ == '__main__':
    gradiend = ModelWithGradiend.from_pretrained('results/models/bert-base-cased')