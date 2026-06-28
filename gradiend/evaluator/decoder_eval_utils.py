"""
Decoder evaluation utilities for decoder grid analysis.
"""

import json
from typing import Any, Dict, Optional, Tuple


def parse_grid_candidate_id(id_key: Any, entry: Optional[Dict[str, Any]] = None) -> Optional[Tuple[float, float]]:
    """Extract ``(feature_factor, learning_rate)`` from a decoder grid key or entry id."""
    if id_key == "base":
        return None
    if isinstance(id_key, tuple) and len(id_key) == 2:
        return float(id_key[0]), float(id_key[1])
    if isinstance(id_key, dict):
        ff = id_key.get("feature_factor")
        lr = id_key.get("learning_rate")
        if ff is not None and lr is not None:
            return float(ff), float(lr)
    if entry:
        id_payload = entry.get("id")
        if isinstance(id_payload, dict):
            ff = id_payload.get("feature_factor")
            lr = id_payload.get("learning_rate")
            if ff is not None and lr is not None:
                return float(ff), float(lr)
    return None


def convert_results_to_dict(list_results):
    """Convert serialized decoder-grid entries to a mapping keyed by candidate id.

    Args:
        list_results: List of decoder result entries. Each entry must contain an
            ``id`` field: either ``"base"`` or a mapping with ``feature_factor``
            and ``learning_rate``.

    Returns:
        Dict keyed by ``"base"`` or ``(feature_factor, learning_rate)``.
    """
    dict_result = {}
    for entry in list_results:
        id = entry['id']
        if isinstance(id, str):
            key = id
        else:
            if 'feature_factor' in id:
                feature_factor = id['feature_factor']
            else:
                raise ValueError(f"Invalid id format: {id}")

            if isinstance(feature_factor, list):
                feature_factor = tuple(feature_factor)

            lr = id['learning_rate']
            key = (feature_factor, lr)

        dict_result[key] = entry
    return dict_result


def convert_results_to_list(dict_results):
    """Convert decoder-grid mapping to the JSON-serializable list format.

    Args:
        dict_results: Mapping keyed by ``"base"`` or
            ``(feature_factor, learning_rate)``.

    Returns:
        List of result entries with normalized ``id`` fields.
    """
    return [{**dict_result, 'id': (key if isinstance(key, str) else {'feature_factor': key[0], 'learning_rate': key[1]})} for key, dict_result in dict_results.items()]


def read_decoder_stats_file(stats_file: str) -> Dict[str, Any]:
    """
    Read a decoder stats JSON file and normalize the grid to a dict.

    Args:
        stats_file: Path to decoder stats JSON.

    Returns:
        Flat dict: summary entries at top level (e.g. "3SG", "3PL") plus "grid".
        Supports both legacy files (with "summary" key) and flat format (summary
        entries at top level; no "summary" key).
    """
    with open(stats_file, "r") as f:
        data = json.load(f)
    grid_list = data.get("grid")
    if grid_list is not None:
        grid = convert_results_to_dict(grid_list)
    else:
        grid = {}
    if "summary" in data:
        summary = data["summary"]
    else:
        summary = {k: v for k, v in data.items() if k != "grid"}
    return {**summary, "grid": grid}
