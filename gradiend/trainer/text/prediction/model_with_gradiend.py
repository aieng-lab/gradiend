"""
TextPredictionModelWithGradiend: MLM/CLM implementation of TextModelWithGradiend.
"""

import torch

from gradiend.util.logging import get_logger
from gradiend.trainer.text.common.model_base import TextModelWithGradiend

logger = get_logger(__name__)


class TextPredictionModelWithGradiend(TextModelWithGradiend):
    """Text model for prediction (MLM/CLM): create_gradients, create_inputs, forward_pass_create_gradients, mask_and_encode."""

    def create_gradients(self, text, label, return_dict=False, verbose=False):
        item = self.create_inputs(text, label)
        outputs = self.base_model(**item)
        if verbose:
            labels = item['labels']
            input_ids = item['input_ids']
            if self.is_decoder_only_model:
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                for i in range(input_ids.size(0)):
                    label_mask = labels[i] != -100
                    label_positions = label_mask.nonzero(as_tuple=True)[0]
                    logger.debug("Sample %d:", i)
                    for pos in label_positions:
                        pred_id = predictions[i, pos - 1].item()
                        true_id = labels[i, pos].item()
                        pred_token = self.tokenizer.decode([pred_id], skip_special_tokens=True)
                        true_token = self.tokenizer.decode([true_id], skip_special_tokens=True)
                        logger.debug("  Position %s: Predicted = '%s', True = '%s'", pos.item(), pred_token, true_token)
        loss_base_model = outputs.loss
        self.base_model.zero_grad()
        loss_base_model.backward()
        gradients = self.gradiend.extract_gradients(
            self.base_model,
            return_dict=return_dict,
            target_device=self.gradiend.device_encoder,
        )
        self.base_model.zero_grad(set_to_none=True)
        return gradients

    def create_inputs(self, masked_text, label):
        item = self.tokenizer(masked_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True)
        item = {k: v.to(self.base_model_device) for k, v in item.items()}
        if self.is_instruction_model:
            label_token_id = self.tokenizer.tokenizer(f'{label}', add_special_tokens=False)['input_ids']
            if self.tokenizer.tokenizer.decode(label_token_id[0]).strip() == '':
                label_token_id = label_token_id[1:]
        elif self.is_decoder_only_model:
            label_token_id = self.tokenizer(f' {label}', add_special_tokens=False)['input_ids']
        else:
            label_token_id = self.tokenizer(f'{label}', add_special_tokens=False)['input_ids']
        if not len(label_token_id) == 1:
            raise ValueError('Only a single label token is supported', label_token_id, label)
        label_token_id = label_token_id[0]
        if self.is_decoder_only_model:
            item['labels'] = torch.full_like(item['input_ids'], -100)
            last_idx = (item['input_ids'].squeeze() != self.tokenizer.pad_token_id).nonzero()[-1].item()
            item['labels'][:, last_idx] = label_token_id
        else:
            labels = item['input_ids'].clone()
            labels[labels != self.tokenizer.mask_token_id] = -100
            labels[labels == self.tokenizer.mask_token_id] = label_token_id
            item['labels'] = labels
        return item

    def forward(self, inputs, return_dict=False, verbose=False, **kwargs):
        inputs = {k: v.to(self.base_model_device) for k, v in inputs.items()}
        inputs = {k: v.unsqueeze(0) if v.ndim == 1 else (v.squeeze(dim=1) if v.ndim == 3 and v.shape[1] == 1 else v) for k, v in inputs.items()}
        outputs = self.base_model(**inputs)
        if verbose:
            logits = outputs.logits
            input_ids = inputs["input_ids"]
            if not self.is_decoder_only_model:
                mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
                batch_logits = logits[mask_positions[:, 0], mask_positions[:, 1], :]
                next_token_ids = batch_logits.argmax(dim=-1)
                next_tokens = self.tokenizer.batch_decode(next_token_ids)
            else:
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                mask = input_ids != pad_id
                mask_2d = mask.view(mask.size(0), -1)
                seq_len = mask_2d.size(1)
                last_non_pad_indices = seq_len - 1 - mask_2d.int().flip(dims=[1]).argmax(dim=1)
                batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
                batch_logits = logits[batch_indices, last_non_pad_indices, :]
                next_token_ids = batch_logits.argmax(dim=-1)
                next_tokens = self.tokenizer.batch_decode(next_token_ids)
            if verbose:
                logger.debug('Predicted next tokens: %s', next_tokens)
        loss_bert = outputs.loss
        self.base_model.zero_grad()
        loss_bert.backward()
        target_device = kwargs.pop("target_device", self.gradiend.device_encoder)
        return self.gradiend.extract_gradients(
            self.base_model,
            return_dict=return_dict,
            target_device=target_device,
        )

    def mask_and_encode(self, text, ignore_tokens=False, return_masked_text=False, single_mask=True, topk=None, topk_part=None):
        item = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        item = {k: v.to(self.base_model_device) for k, v in item.items()}
        labels = item['input_ids'].clone()
        if self.is_decoder_only_model:
            n = labels.shape[1]
            labels = torch.cat([labels[:, 1:], torch.full_like(labels[:, :1], self.tokenizer.pad_token_id)], dim=1)
            if ignore_tokens:
                valid_indices_mask = ~torch.isin(labels[0, n // 2:], torch.tensor(ignore_tokens, device=labels.device))
                valid_indices = torch.nonzero(valid_indices_mask, as_tuple=False).squeeze(-1)
                if valid_indices.numel() > 0:
                    random_idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,))] + n // 2
                else:
                    raise ValueError('No valid indices found in the second half of the sequence', text)
            else:
                random_idx = torch.randint(n // 2, n, (1,))
            labels[:, :random_idx - 1] = self.tokenizer.pad_token_id
            labels[:, random_idx:] = self.tokenizer.pad_token_id
            mask = labels != self.tokenizer.pad_token_id
        else:
            if single_mask:
                mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
                mask[0, torch.randint(0, labels.shape[1], (1,))] = True
            else:
                random_mask = torch.rand(labels.shape, dtype=torch.float, device=labels.device) < 0.15
                padding_mask = labels == self.tokenizer.pad_token_id
                exclude_mask = (labels.unsqueeze(-1) == torch.Tensor(ignore_tokens).to(labels.device)).any(dim=-1) if ignore_tokens else torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
                mask = random_mask & ~padding_mask & ~exclude_mask
            labels[~mask] = -100
            item['input_ids'][mask] = self.tokenizer.mask_token_id
        item['labels'] = labels
        outputs = self.base_model(**item)
        loss_base_model = outputs.loss
        if loss_base_model is None:
            if return_masked_text:
                return None, None, None
            return None
        self.base_model.zero_grad()
        loss_base_model.backward()
        gradients = self.gradiend.extract_gradients(
            self.base_model, target_device=self.gradiend.device_encoder
        )
        self.base_model.zero_grad(set_to_none=True)
        if topk is not None:
            topk_part = topk_part or "encoder-weight"
            enhancer_mask = self.get_enhancer_mask(topk=topk, part=topk_part)
            masked_encoder = self.gradiend.encoder[0].weight.flatten()[enhancer_mask]
            masked_input = gradients[enhancer_mask]
            encoded = torch.matmul(masked_input.unsqueeze(0), masked_encoder.unsqueeze(1)).squeeze(1) + self.gradiend.encoder[0].bias
            encoded = self.gradiend.encoder[1](encoded)
        else:
            encoded = self.gradiend.encoder(gradients)
        if hasattr(encoded, 'tolist'):
            encoded = encoded.tolist()
        if len(encoded) == 1:
            encoded = encoded[0]
        if return_masked_text:
            masked_str = self.tokenizer.decode(item['input_ids'].squeeze())
            masked_str = masked_str.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            labels_list = [self.tokenizer.decode(id) for id in item['labels'][mask]]
            return encoded, masked_str, labels_list
        return encoded
