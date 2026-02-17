"""
Shared pytest fixtures for GRADIEND tests.

Provides mock models, tokenizers, and common test utilities.
"""

import os
import tempfile
import shutil
from typing import Optional

import pytest
import torch
import torch.nn as nn

# Force CPU for all tests to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if torch.cuda.is_available():
    torch.cuda.set_device(torch.device("cpu"))


class SimpleMockModel(nn.Module):
    """Simple mock base model with minimal parameters for testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2, name_or_path='mock-model', dtype=torch.float32):
        super().__init__()
        self.name_or_path = name_or_path
        self._dtype = dtype
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
        })()
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.cls = type('Cls', (), {'predictions': self.classifier})()
        
        # Initialize parameters with the specified dtype
        self.to(dtype=dtype)
    
    @property
    def dtype(self):
        """Return the dtype of the first parameter (PyTorch convention)."""
        if len(list(self.parameters())) > 0:
            return next(self.parameters()).dtype
        return self._dtype
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.get('input_ids')
        if input_ids is None:
            logits = torch.zeros(1, 10, self.config.vocab_size, dtype=self.dtype)
            loss = torch.tensor(0.0, dtype=self.dtype, requires_grad=True)
            return type('Output', (), {'logits': logits, 'loss': loss})()
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        x = self.embeddings(input_ids)
        for layer in self.encoder:
            x = layer(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            if len(logits.shape) == 3:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            else:
                loss = torch.tensor(0.0, dtype=self.dtype, requires_grad=True)
        
        return type('Output', (), {'logits': logits, 'loss': loss})()


class MockTokenizer:
    """Simple mock tokenizer."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.name_or_path = 'mock-tokenizer'
        self.mask_token = '[MASK]'
        self.mask_token_id = 103
        self.pad_token = '[PAD]'
        self.pad_token_id = 0
        self.eos_token = '[EOS]'
        self.eos_token_id = 102
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.vocab = {f'token_{i}': i for i in range(vocab_size)}
        self.vocab.update({
            '[MASK]': 103, '[PAD]': 0, '[CLS]': 101, '[SEP]': 102
        })
        
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs."""
        if isinstance(tokens, str):
            return self.vocab.get(tokens, 0)
        elif isinstance(tokens, list):
            return [self.vocab.get(t, 0) for t in tokens]
        else:
            return tokens
    
    def __call__(self, text, return_tensors=None, padding=True, truncation=True, 
                 max_length=48, add_special_tokens=True, **kwargs):
        tokens = text.split()[:max_length-2] if truncation else text.split()
        token_ids = [self.vocab.get(token, 1) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        
        if padding and len(token_ids) < max_length:
            token_ids = token_ids + [self.vocab['[PAD]']] * (max_length - len(token_ids))
        
        result = {'input_ids': token_ids[:max_length]}
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor([result['input_ids']])
            result['attention_mask'] = torch.tensor([[1 if tid != self.vocab['[PAD]'] else 0 
                                                      for tid in result['input_ids'][0]]])
        return result
    
    def encode(self, text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        return [self.vocab.get(token, 1) for token in tokens]
    
    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab.get(tid, f'<unk_{tid}>') for tid in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if not (t.startswith('[') and t.endswith(']'))]
        return ' '.join(tokens)


@pytest.fixture
def mock_model():
    """Fixture providing a simple mock base model."""
    return SimpleMockModel(name_or_path='mock-model', dtype=torch.float32)


@pytest.fixture(autouse=True)
def force_cpu():
    """Automatically force CPU device for all tests."""
    # Set environment variable to hide CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Force CPU device
    original_device = torch.cuda.current_device if torch.cuda.is_available() else None
    yield
    # Restore if needed (though tests shouldn't use CUDA)
    if original_device is not None:
        try:
            torch.cuda.set_device(original_device)
        except:
            pass


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a simple mock tokenizer."""
    return MockTokenizer(vocab_size=1000)


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def set_seed():
    """Fixture to set random seeds for reproducibility."""
    def _set_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    return _set_seed


@pytest.fixture
def patch_model_loading():
    """Fixture to patch model loading to avoid HuggingFace calls."""
    from unittest.mock import patch
    from gradiend.trainer.text.common.model_base import TextModelWithGradiend
    
    def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
        """Mock _load_model to return mock objects instead of loading from HuggingFace."""
        # This will be used in tests that need to avoid HuggingFace loading
        # The actual mock_model and mock_tokenizer will be passed via closure
        # For now, return None - tests should provide their own mocks
        return None, None
    
    return patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model))
