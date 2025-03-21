import copy
import re

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
import json
import os

from gradiend.util import hash_it


class AutoModelForLM(nn.Module):

    @classmethod
    def from_pretrained(self, name_or_path):
        try:
            return AutoModelForMaskedLM.from_pretrained(name_or_path)
        except ValueError:
            return AutoModelForCausalLM.from_pretrained(name_or_path)

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
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
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

    @property
    def layers_hash(self):
        if isinstance(self.layers, dict):
            layers_keys_hash = hash_it(list(self.layers.keys()))
            sparse_layers = torch.concat([self.layers[k].flatten() for k in self.layers], dim=0).cpu().to_sparse()
            layers_indices = sparse_layers.indices().cpu().numpy()
            layers_values = sparse_layers.values().cpu().numpy()
            layers_hash = hash_it((layers_keys_hash, layers_indices, layers_values))
        else:
            layers_hash = hash_it(self.layers)
        return layers_hash

    def extract_gradients(self, bert, return_dict=False, ):
        layer_map = {k: v for k, v in bert.named_parameters()}
        if isinstance(self.layers, dict):
            if return_dict:
                return {k: v.grad.detach() * self.layers[k] if v.grad is not None else torch.zeros_like(v) for k, v in layer_map.items() if k in self.layers}
            else:
                layer_map = {k: v.grad[self.layers[k]] for k, v in layer_map.items() if k in self.layers}
        elif return_dict:
            return {layer: layer_map[layer].grad.detach() for layer in self.layers}

        return torch.concat([layer_map[layer].flatten().detach() for layer in self.layers])

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, x):
        orig_shapes = {}

        if hasattr(x, 'named_parameters'):
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            if isinstance(self.layers, dict):
                # New behavior: process only masked elements
                for layer, mask in self.layers.items():
                    param = layer_map[layer]
                    grad = param.grad.flatten()
                    selected_grad = grad[mask.flatten()]  # Extract relevant elements
                    grads.append(selected_grad)
                    orig_shapes[layer] = (param.shape, mask)  # Store shape & mask
            else:
                for layer in self.layers:
                    param = layer_map[layer]
                    grad = param.grad.flatten()
                    grads.append(grad)
                    orig_shapes[layer] = param.shape
            x = torch.concat(grads)
        elif isinstance(x, dict):
            grads = []
            if isinstance(self.layers, dict):
                for layer, mask in self.layers.items():
                    param = x[layer]
                    grad = param.flatten()
                    selected_grad = grad[mask.flatten()]  # Extract relevant elements
                    grads.append(selected_grad)
                    orig_shapes[layer] = (param.shape, mask)  # Store shape & mask
            else:
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
            for layer, shape_info in orig_shapes.items():
                if isinstance(shape_info, tuple):
                    shape, mask = shape_info  # Extract shape and mask
                    num_elements = mask.sum().item()

                    # Reconstruct full tensor, placing decoded values only where mask is True
                    reconstructed = torch.zeros_like(mask, dtype=decoded.dtype)
                    reconstructed[mask] = decoded[start_idx:start_idx + num_elements]
                    decoded_params[layer] = reconstructed.reshape(shape)
                else:
                    shape = shape_info
                    num_elements = shape.numel()
                    decoded_params[layer] = decoded[start_idx:start_idx + num_elements].reshape(shape)

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
        layers_path = os.path.join(save_directory, 'layers.pth')

        # Save model state dictionary
        torch.save(self.state_dict(), model_path)

        self.kwargs.update(kwargs)

        # Save sparse tensor layers separately
        if isinstance(self.layers, dict):
            torch.save(self.layers, layers_path)

        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'intermediate': self.intermediate,
            'layers_path': 'layers.pth',  # Reference the separate file
            'layers': list(self.layers),
            'activation': self.activation,
            'activation_decoder': self.activation_decoder,
            'bias_decoder': self.bias_decoder,
            **self._serialize_kwargs(),
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

    def _serialize_kwargs(self):
        kwargs = self.kwargs.copy()
        training_kwargs = kwargs['training'].copy()

        if isinstance(training_kwargs['layers'], dict):
            training_kwargs['layers'] = list(training_kwargs['layers'].keys())
            training_kwargs['layers_path'] = 'layers.pth'
        kwargs['training'] = training_kwargs

        return kwargs



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

        if 'layers_path' in config:
            layers_path = os.path.join(load_directory, config['layers_path'])
            # Load sparse layers
            model.layers = torch.load(layers_path)

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

def is_generative(model):
    return hasattr(model, 'lm_head')

class ModelWithGradiend(nn.Module):

    def __init__(self, base_model, gradiend, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.gradiend = gradiend
        self.tokenizer = tokenizer
        self.grad_iterations = gradiend.grad_iterations

        self.layer_map = {k: v for k, v in self.base_model.named_parameters()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        self.gradiend.to(self.device)

        self.is_generative = is_generative(self.base_model)

        if self.is_generative:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self):
        return os.path.basename(self.gradiend.name_or_path)

    def create_gradients(self, text, label, return_dict=False):
        item = self.create_inputs(text, label)

        outputs = self.base_model(**item)
        loss_base_model = outputs.loss

        self.base_model.zero_grad()
        loss_base_model.backward()

        gradients = self.gradiend.extract_gradients(self.base_model, return_dict=return_dict)
        return gradients


    def encode(self, text, label):
        gradients = self.create_gradients(text, label)
        encoded = self.gradiend.encoder(gradients).item()
        return encoded


    def modify_model(self, lr, gender_factor, part='decoder', top_k=None, top_k_part=None):
        # returns a base_model model with enhanced weights based on the auto encoder, use the learning rate parameter to control the influence of the auto encoder
        top_k_part = top_k_part or part

        import copy
        enhanced_model = copy.deepcopy(self.base_model)

        if top_k == 0:
            return enhanced_model

        layer_map = {k: v for k, v in enhanced_model.named_parameters()}
        if part == 'decoder':
            enhancer = self.gradiend.decoder(torch.Tensor([gender_factor]).to(self.device))
        elif part == 'encoder':
            enhancer = self.gradiend.encoder[0].weight.flatten().to(self.device)
        else:
            raise ValueError('Unsupported part:', part)

        if top_k is not None and top_k < len(enhancer):
            mask = self.get_enhancer_mask(top_k, part=top_k_part)

            if mask.sum() == 0.0:
                return enhanced_model

            enhancer[~mask] = 0.0

        idx = 0
        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                for layer, layer_mask in self.gradiend.layers.items():
                    number_of_elements = layer_mask.sum().item()  # Convert to Python integer if it's a tensor

                    # Extract the relevant elements from enhancer
                    update_values = enhancer[idx:idx + number_of_elements]

                    # Create an update tensor of the same shape as layer_mask, filling only True positions
                    update_tensor = torch.zeros_like(layer_mask, dtype=update_values.dtype)
                    update_tensor[layer_mask] = update_values  # Assign values only to True positions

                    # Apply update
                    layer_map[layer] += lr * update_tensor

                    # Increment index
                    idx += number_of_elements
            else:
                for layer in self.gradiend.layers:
                    shape = layer_map[layer].shape
                    number_of_elements = shape.numel()
                    layer_map[layer] += lr * enhancer[idx:idx + number_of_elements].reshape(shape)
                    idx += number_of_elements

        return enhanced_model

    def get_enhancer(self, part='decoder'):
        # we set parts of the enhancer to 0 that are not in the top k highest values (wrt absolute value)
        if part == 'decoder':
            abs_enhancer = self.gradiend.decoder[0].weight.flatten().abs()
        elif part == 'decoder-bias':
            abs_enhancer = self.gradiend.decoder[0].bias.abs()
        elif part == 'decoder-sum':
            abs_enhancer = (self.gradiend.decoder[0].weight.flatten() + self.gradiend.decoder[0].bias).abs()
        else:
            raise ValueError('Unsupported part:', part)

        return abs_enhancer


    def get_enhancer_mask(self, top_k, part='decoder'):
        gradiend_vector = self.gradiend.decoder[0].weight.flatten()

        if 0.0 < top_k <= 1.0:
            top_k = int(top_k * len(gradiend_vector))

        abs_enhancer = self.get_enhancer(part=part).cpu() # move to cpu to ensure deterministic behavior
        sorted_indices = torch.argsort(abs_enhancer, stable=True)  # Ensure stable order
        sorted_enhancer = abs_enhancer[sorted_indices]
        #with torch.use_deterministic_algorithms(True):
        top_k_values, top_k_sorted_indices = sorted_enhancer.topk(top_k, sorted=False, largest=True)

        # Convert back to original indices
        top_k_indices = sorted_indices[top_k_sorted_indices]
        mask = torch.zeros_like(gradiend_vector, dtype=torch.bool)
        mask[top_k_indices] = True
        return mask

    def get_layer_mask(self, top_k, part='decoder'):
        enhancer_mask = self.get_enhancer_mask(top_k=top_k, part=part)
        all_layer_map = {k: v for k, v in self.base_model.named_parameters()}
        layer_map = {}

        idx = 0
        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                raise NotImplementedError()
                for layer, layer_mask in self.gradiend.layers.items():
                    shape = layer_map[layer].shape
                    number_of_elements = layer_mask.sum().item()  # Convert to Python integer if it's a tensor

                    # Extract the relevant elements from enhancer
                    #update_values = enhancer[idx:idx + number_of_elements]

                    # Create an update tensor of the same shape as layer_mask, filling only True positions
                    update_tensor = torch.zeros_like(layer_mask, dtype=update_values.dtype)
                    update_tensor[layer_mask] = update_values  # Assign values only to True positions

                    # Apply update
                    layer_map[layer] = lr * update_tensor

                    # Increment index
                    idx += number_of_elements
            else:
                for layer in self.gradiend.layers:
                    shape = all_layer_map[layer].shape
                    number_of_elements = shape.numel()
                    layer_map[layer] = enhancer_mask[idx:idx + number_of_elements].reshape(shape) #.to_sparse()
                    idx += number_of_elements

        return layer_map


    def mask_and_encode(self, text, ignore_tokens=False, return_masked_text=False, single_mask=True):
        item = self.tokenizer(text, return_tensors="pt")
        item = {k: v.to(self.device) for k, v in item.items()}
        labels = item['input_ids'].clone()

        if self.is_generative:

            n = labels.shape[1]

            # left shift the labels by one
            labels = torch.cat([labels[:, 1:], torch.full_like(labels[:, :1], self.tokenizer.pad_token_id)], dim=1)

            # use random idx to predict in the 2nd half of the sequence to ensure enough context
            if ignore_tokens:
                # Create a mask for valid indices in the second half, excluding the ignore tokens
                valid_indices_mask = ~torch.isin(labels[0, n // 2:], torch.tensor(ignore_tokens, device=labels.device))

                # Get the indices of the valid tokens
                valid_indices = torch.nonzero(valid_indices_mask, as_tuple=False).squeeze()

                if valid_indices.numel() > 0:
                    # Randomly select an index from the valid tokens
                    random_idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,))] + n // 2
                else:
                    # Handle case where no valid indices exist (fallback behavior, could raise an error or select any index)
                    raise ValueError('No valid indices found in the second half of the sequence', text)
            else:
                random_idx = torch.randint(n // 2, n, (1,))

            labels[:, :random_idx-1] = self.tokenizer.pad_token_id
            labels[:, random_idx:] = self.tokenizer.pad_token_id

            mask = labels != self.tokenizer.pad_token_id
        else:
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

        item['labels'] = labels

        outputs = self.base_model(**item)
        loss_bert = outputs.loss

        self.base_model.zero_grad()
        loss_bert.backward()

        gradients = self.gradiend.extract_gradients(self.base_model)

        encoded = self.gradiend.encoder(gradients).item()

        if return_masked_text:
            masked_str = self.tokenizer.decode(item['input_ids'].squeeze())
            masked_str = masked_str.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            labels = [self.tokenizer.decode(id) for id in item['labels'][mask]]
            return encoded, masked_str, labels

        return encoded

    def create_inputs(self, masked_text, label):
        item = self.tokenizer(masked_text, return_tensors="pt")
        item = {k: v.to(self.device) for k, v in item.items()}
        label_token_id = self.tokenizer(label, add_special_tokens=False)['input_ids']
        if not len(label_token_id) == 1:
            raise ValueError('Only a single label token is supported')
        label_token_id = label_token_id[0]

        if self.is_generative:
            item['labels'] = torch.full_like(item['input_ids'], -100)
            last_idx = (item['input_ids'].squeeze() != self.tokenizer.pad_token_id).nonzero()[-1].item()
            item['labels'][:, -last_idx] = label_token_id
        else:
            labels = item['input_ids'].clone()
            labels[labels != self.tokenizer.mask_token_id] = -100  # only compute loss on masked tokens
            labels[labels == self.tokenizer.mask_token_id] = label_token_id
            item['labels'] = labels
        return item

    def forward_pass(self, inputs, return_dict=False, lr=1e-4): # todo implement batched=True
        # Forward pass through BERT
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        grads = []

        if self.grad_iterations > 1:
            base_model = copy.deepcopy(self.base_model)
            for i in range(self.grad_iterations):
                outputs = base_model(**inputs) # todo CLM
                loss_bert = outputs.loss
                base_model.zero_grad()
                loss_bert.backward()
                gradients = self.gradiend.extract_gradients(base_model, return_dict=True)
                grads.append(gradients)

                if i < self.grad_iterations - 1:
                    # perform the training step
                    # Step 6: Update the model's weights

                    with torch.no_grad():
                        for name, param in base_model.named_parameters():
                            if param.grad is not None:
                                param.add_(-lr * param.grad)

                # save only the last gradient (for now, maybe add some advanced logic later)
                grads = [grads[-1]]

            if return_dict:
                return grads[-1]
            flatten_gradient = torch.concat(tuple(grad.flatten() for grad in grads[-1].values()))
            return flatten_gradient

        else:
            outputs = self.base_model(**inputs)
            loss_bert = outputs.loss

            self.base_model.zero_grad()
            loss_bert.backward()
            return self.gradiend.extract_gradients(self.base_model, return_dict=return_dict)

    @property
    def layers_hash(self):
        return self.gradiend.layers_hash

    # invert the encoded value, i.e. encoded value * (-1), while keeping the decoders value
    def invert_encoding(self):
        with torch.no_grad():
            self.gradiend.encoder[0].weight = torch.nn.Parameter(self.gradiend.encoder[0].weight * -1)
            self.gradiend.encoder[0].bias = torch.nn.Parameter(self.gradiend.encoder[0].bias * -1)
            self.gradiend.decoder[0].weight = torch.nn.Parameter(self.gradiend.decoder[0].weight * -1)

    def save_pretrained(self, save_directory, **kwargs):
        self.gradiend.save_pretrained(save_directory, bert=self.base_model.name_or_path, tokenizer=self.tokenizer.name_or_path, **kwargs)

    @classmethod
    def from_pretrained(cls, load_directory, layers=None, latent_dim=1, **kwargs):
        layers = layers or []
        if len(layers) == 1 and isinstance(layers[0], list):
            layers = layers[0]

        try:
            ae = GradiendModel.from_pretrained(load_directory)

            if layers and ae.layers != layers:
                raise ValueError(f'The provided layers {layers} do not match the layers in the model configuration {ae.layers}')
            else:
                layers = ae.layers

            base_model_id = ae.kwargs['base_model']
            base_model = AutoModelForLM.from_pretrained(base_model_id)
            tokenizer = ae.kwargs.get('tokenizer', base_model_id)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except FileNotFoundError:
            print('No model with auto encoder found in the specified directory:', load_directory, ' -> creating a new auto encoder')

            if isinstance(load_directory, str):
                base_model = AutoModelForLM.from_pretrained(load_directory)
                tokenizer = AutoTokenizer.from_pretrained(load_directory)
            else:
                base_model = load_directory
                tokenizer = base_model.tokenizer

            layer_map = {k: v for k, v in base_model.named_parameters() if 'cls.prediction' not in k.lower()}

            if layers:
                if not isinstance(layers, dict):
                    # layers information are provided
                    # check if layer description can be matched with layer_map keys
                    # layer description could also be "*.layer.10*"
                    matched_layers = []
                    for layer in layers:
                        if layer in layer_map:
                            matched_layers.append(layer)
                        else:
                            # Handle wildcard matching (e.g., "*.layer.10*")
                            layer_pattern = layer.replace('.', r'\.').replace('*', r'.*')  # Convert to regex pattern
                            layer_regex = re.compile(layer_pattern)

                            for layer_name in layer_map:
                                if layer_regex.fullmatch(layer_name):
                                    matched_layers.append(layer_name)

                    layers = list(sorted(matched_layers, key=lambda x: list(layer_map.keys()).index(x)))
            else:
                # no layer provided, i.e. all layers are used that are part of the core model, i.e. all layers that are not part of prediction layers
                layers = [layer for layer in layer_map]

            if isinstance(layers, dict):
                input_dim = sum([v.sum() for v in layers.values()])
            else:
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
            yield from self.gradiend.named_parameters()
            return
        elif part == 'encoder':
            layer_map = {k: v for k, v in self.gradiend.encoder.named_parameters()}
            weights = layer_map['0.weight'].squeeze()

        elif 'decoder' in part:
            layer_map = {k: v for k, v in self.gradiend.decoder.named_parameters()}
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

        for layer in self.gradiend.layers:
            orig_shape = self.layer_map[layer].shape
            num_elements = orig_shape.numel()
            yield layer, weights[idx:idx + num_elements].reshape(orig_shape)
            idx += num_elements
        if idx != weights.numel():
            raise ValueError(f'Inconsistent number of elements in the weights and expected number of elements in the layers ({idx} vs. {weights.numel()})')


if __name__ == '__main__':
    gradiend = ModelWithGradiend.from_pretrained('results/models/bert-base-cased')