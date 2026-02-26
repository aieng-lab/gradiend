"""
Decoder evaluation utilities for decoder grid analysis.
"""

import json
from typing import Any, Dict


def convert_results_to_dict(list_results):
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
    return [{**dict_result, 'id': (key if isinstance(key, str) else {'feature_factor': key[0], 'learning_rate': key[1]})} for key, dict_result in dict_results.items()]


def read_decoder_stats_file(stats_file: str) -> Dict[str, Any]:
    """
    Read a decoder stats JSON file and normalize the grid to a dict.

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
