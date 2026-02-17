import io
import json
import pickle
import hashlib
import torch

def set_requires_grad_true(model):
    """Set requires_grad=True for all parameters (e.g. after loading for GRADIEND use)."""
    for p in model.parameters():
        p.requires_grad = True

def convert_tuple_keys_recursively(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert tuple keys to JSON strings (of lists)
            if isinstance(k, tuple):
                k = json.dumps(k)
            new_dict[k] = convert_tuple_keys_recursively(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys_recursively(item) for item in obj]
    else:
        return obj

def restore_tuple_keys_recursively(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try:
                k_loaded = json.loads(k)
                if isinstance(k_loaded, list):
                    k = tuple(k_loaded)
            except (ValueError, json.JSONDecodeError):
                pass
            new_dict[k] = restore_tuple_keys_recursively(v)
        return new_dict
    elif isinstance(obj, list):
        return [restore_tuple_keys_recursively(item) for item in obj]
    else:
        return obj


def normalize_split_name(split: str) -> str:
    """
    Normalize split names to standard form.

    Maps common variations to standard names:
    - 'val' -> 'validation'
    - 'validation' -> 'validation'
    - 'train' -> 'train'
    - 'test' -> 'test'

    Args:
        split: Split name to normalize (case-insensitive)

    Returns:
        Normalized split name (lowercase, standard form)
    """
    split_normalization = {
        "val": "validation",
        "validation": "validation",
        "train": "train",
        "test": "test",
    }
    return split_normalization.get(split.lower(), split.lower())


def json_loads(x):
    """Parse JSON string or list-like string; return numeric/other as-is."""
    if isinstance(x, (float, int)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return [
            xx.removeprefix("'").removesuffix("'")
            for xx in x.removeprefix("[").removesuffix("]").split(",")
        ]


def hash_model_weights(model):

    if hasattr(model, 'hash'):
        model_hash = model.hash()
    else:
        # Create a BytesIO buffer to store the model's state_dict
        buffer = io.BytesIO()
        # Save the state_dict to the buffer
        torch.save(model.state_dict(), buffer)
        # Get the byte data from the buffer
        model_bytes = buffer.getvalue()
        # Create a SHA-256 hash
        model_hash = hashlib.sha256(model_bytes).hexdigest()
    return model_hash


# hashes a string deterministically across multiple program runs (in contrast to Python's built-in hash function)
def hash_it(x, return_num=False):
    """
    Returns a deterministic hash for any hashable object.

    Parameters:
    x (hashable): The object to hash.

    Returns:
    str: The SHA-256 hash of the object.
    """
    # Serialize the object to a byte stream using pickle
    byte_stream = pickle.dumps(x)

    # Compute the SHA-256 hash of the byte stream
    hash_object = hashlib.sha256(byte_stream)
    hash_hex = hash_object.hexdigest()

    if return_num:
        # Convert the hex hash to an integer
        hash_num = int(hash_hex, 16)
        return hash_num

    return hash_hex