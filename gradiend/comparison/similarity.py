"""Weight-space and top-k similarity matrices for GRADIEND models."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from gradiend.comparison.common import (
    GroupingSpec,
    _aggregate_seed_scores,
    _pearson,
    _rankdata_average,
    _validate_aggregate_dispersion_combo,
    _validate_models,
    _validate_topk_optional,
    _normalize_model_groups,
)


def _resolve_overlap_fraction(size_a: int, size_b: int, inter: int) -> float:
    denom = min(size_a, size_b)
    return (inter / denom) if denom else 0.0


def _resolve_local_indices(gradiend: object, part: str, topk: Optional[Union[int, float]]) -> List[int]:
    if topk is None:
        return list(range(int(gradiend.input_dim)))
    return list(gradiend.get_topk_weights(part=part, topk=topk))


def _extract_sparse_blocks(
    model: object,
    *,
    part: str,
    topk: Optional[Union[int, float]] = None,
    include_param_names: bool = False,
) -> Dict[str, Any]:
    if not hasattr(model, "gradiend"):
        raise TypeError("Similarity measures require models with a .gradiend attribute")

    gradiend = model.gradiend
    gradiend._require_built()
    local_indices = _resolve_local_indices(gradiend, part, topk)
    base_map = gradiend._get_base_global_index_map().detach().cpu()
    blocks: Dict[int, torch.Tensor] = {}
    param_names: Dict[int, str] = {}

    def _remember(local_idx: int, tensor: torch.Tensor) -> None:
        base_idx = int(base_map[local_idx].item())
        blocks[base_idx] = tensor.detach().cpu().flatten().clone()
        if include_param_names and hasattr(gradiend, "decode_base_global_index"):
            try:
                meta = gradiend.decode_base_global_index(base_idx)
                param_name = meta.get("param_name")
                if isinstance(param_name, str):
                    param_names[base_idx] = param_name
            except Exception:
                pass

    if part == "encoder-weight":
        weight = gradiend.encoder[0].linear.weight.detach().cpu()
        block_dim = int(weight.shape[0])
        for local_idx in local_indices:
            _remember(local_idx, weight[:, local_idx])
    elif part == "decoder-weight":
        weight = gradiend.decoder[0].linear.weight.detach().cpu()
        block_dim = int(weight.shape[1])
        for local_idx in local_indices:
            _remember(local_idx, weight[local_idx, :])
    elif part in {"decoder-bias", "decoder-sum"}:
        vec = gradiend.get_update_vector(part=part).detach().cpu()
        block_dim = 1
        for local_idx in local_indices:
            _remember(local_idx, vec[local_idx:local_idx + 1])
    else:
        raise ValueError(
            "Supported parts are 'encoder-weight', 'decoder-weight', 'decoder-bias', and 'decoder-sum'"
        )

    total_base_size = None
    param_map = getattr(gradiend, "param_map", None)
    if isinstance(param_map, dict):
        total_base_size = 0
        for spec in param_map.values():
            shape = tuple(spec.get("shape", ()))
            total_base_size += int(torch.tensor(shape).prod().item())

    return {
        "blocks": blocks,
        "block_dim": int(block_dim),
        "resolved_topk": len(local_indices),
        "param_names": param_names,
        "total_base_size": total_base_size,
    }


def _extract_importance_by_index(
    model: object,
    *,
    part: str,
    topk: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    if not hasattr(model, "gradiend"):
        raise TypeError("Similarity measures require models with a .gradiend attribute")
    gradiend = model.gradiend
    gradiend._require_built()
    local_indices = _resolve_local_indices(gradiend, part, topk)
    base_map = gradiend._get_base_global_index_map().detach().cpu()
    importance = gradiend.get_weight_importance(part=part).detach().cpu()
    by_index: Dict[int, float] = {}
    param_names: Dict[int, str] = {}
    for local_idx in local_indices:
        base_idx = int(base_map[local_idx].item())
        by_index[base_idx] = float(abs(importance[local_idx].item()))
        if hasattr(gradiend, "decode_base_global_index"):
            try:
                meta = gradiend.decode_base_global_index(base_idx)
                param_name = meta.get("param_name")
                if isinstance(param_name, str):
                    param_names[base_idx] = param_name
            except Exception:
                pass
    return {
        "importance": by_index,
        "resolved_topk": len(local_indices),
        "param_names": param_names,
    }


def _cosine_from_blocks(blocks_i: Dict[int, torch.Tensor], blocks_j: Dict[int, torch.Tensor]) -> float:
    norm_i = math.sqrt(sum(float(torch.dot(b, b).item()) for b in blocks_i.values()))
    norm_j = math.sqrt(sum(float(torch.dot(b, b).item()) for b in blocks_j.values()))
    if norm_i == 0.0 or norm_j == 0.0:
        return 0.0
    if len(blocks_i) <= len(blocks_j):
        left, right = blocks_i, blocks_j
    else:
        left, right = blocks_j, blocks_i
    dot = 0.0
    for key, vec in left.items():
        other = right.get(key)
        if other is not None:
            dot += float(torch.dot(vec, other).item())
    return float(dot / (norm_i * norm_j))


def _vector_from_union(blocks_i: Dict[int, torch.Tensor], blocks_j: Dict[int, torch.Tensor]) -> Tuple[List[float], List[float]]:
    keys = sorted(set(blocks_i.keys()) | set(blocks_j.keys()))
    out_i: List[float] = []
    out_j: List[float] = []
    if not keys:
        return out_i, out_j
    block_dim = None
    for source in (blocks_i, blocks_j):
        if source:
            block_dim = len(next(iter(source.values())))
            break
    assert block_dim is not None
    zeros = [0.0] * int(block_dim)
    for key in keys:
        vec_i = blocks_i.get(key)
        vec_j = blocks_j.get(key)
        out_i.extend(vec_i.tolist() if vec_i is not None else zeros)
        out_j.extend(vec_j.tolist() if vec_j is not None else zeros)
    return out_i, out_j


def _dense_vector_from_extracted(info: Dict[str, Any]) -> List[float]:
    total_base_size = info.get("total_base_size")
    block_dim = int(info.get("block_dim", 0))
    if not isinstance(total_base_size, int) or total_base_size <= 0 or block_dim <= 0:
        keys = sorted(info["blocks"].keys())
        out: List[float] = []
        for key in keys:
            out.extend(info["blocks"][key].tolist())
        return out
    dense = torch.zeros((total_base_size, block_dim), dtype=torch.float32)
    for base_idx, block in info["blocks"].items():
        dense[int(base_idx)] = block.to(dtype=torch.float32)
    return dense.flatten().tolist()


def _group_name_from_param(param_name: str, grouping: GroupingSpec) -> str:
    if grouping in (None, "param"):
        return param_name
    if isinstance(grouping, dict):
        return str(grouping.get(param_name, "other"))
    if callable(grouping):
        return str(grouping(param_name))
    grouping_name = str(grouping).lower()
    if grouping_name == "layer":
        match = re.search(r"(?:layers?|layer|encoder\.layer|decoder\.layer|h)\.(\d+)", param_name)
        return f"layer_{match.group(1)}" if match else "other"
    if grouping_name == "component":
        name = param_name.lower()
        if "embeddings" in name or "embedding" in name:
            return "embeddings"
        if re.search(r"(query|q_proj|wq)", name):
            return "attention_query"
        if re.search(r"(key|k_proj|wk)", name):
            return "attention_key"
        if re.search(r"(value|v_proj|wv)", name):
            return "attention_value"
        if re.search(r"(o_proj|out_proj|dense)", name) and "attention" in name:
            return "attention_output"
        if re.search(r"(intermediate|up_proj|gate_proj|fc1)", name):
            return "mlp_up"
        if re.search(r"(output|down_proj|fc2)", name) and "attention" not in name:
            return "mlp_down"
        if "norm" in name or "ln" in name:
            return "normalization"
        if "lm_head" in name or "classifier" in name or "score" in name:
            return "head"
        return "other"
    raise ValueError("grouping must be None, 'param', 'layer', 'component', a dict, or a callable")


def _compute_topk_overlap_similarity_matrix(models: Dict[str, object], *, part: str, topk: Union[int, float], value: str) -> Dict[str, Any]:
    per_model: Dict[str, List[int]] = {}
    sets: Dict[str, set] = {}
    set_sizes: Dict[str, int] = {}
    for model_id, model in models.items():
        indices = model.get_topk_weights(part=part, topk=topk)
        per_model[model_id] = indices
        sets[model_id] = set(indices)
        set_sizes[model_id] = len(indices)
    if any(size <= 0 for size in set_sizes.values()):
        raise ValueError("Each model must contribute at least one selected weight")
    model_ids = list(models.keys())
    matrix = [[0.0] * len(model_ids) for _ in range(len(model_ids))]
    for i, mi in enumerate(model_ids):
        for j, mj in enumerate(model_ids):
            inter = len(sets[mi] & sets[mj])
            if value == "intersection":
                matrix[i][j] = float(inter)
            elif value == "intersection_frac":
                matrix[i][j] = _resolve_overlap_fraction(set_sizes[mi], set_sizes[mj], inter)
            else:
                raise ValueError("value must be 'intersection' or 'intersection_frac'")
    return {
        "measure": "topk_overlap",
        "model_ids": model_ids,
        "matrix": matrix,
        "part": part,
        "topk": topk,
        "value": value,
        "per_model": per_model,
        "resolved_topk": {mid: set_sizes[mid] for mid in model_ids},
    }


def _compute_cosine_like_matrix(models: Dict[str, object], *, part: str, topk: Optional[Union[int, float]], take_abs: bool, measure_name: str) -> Dict[str, Any]:
    extracted = {mid: _extract_sparse_blocks(model, part=part, topk=topk) for mid, model in models.items()}
    model_ids = list(models.keys())
    matrix = [[0.0] * len(model_ids) for _ in range(len(model_ids))]
    for i, mi in enumerate(model_ids):
        for j, mj in enumerate(model_ids):
            score = _cosine_from_blocks(extracted[mi]["blocks"], extracted[mj]["blocks"])
            matrix[i][j] = abs(score) if take_abs else score
    return {
        "measure": measure_name,
        "model_ids": model_ids,
        "matrix": matrix,
        "part": part,
        "topk": topk,
        "resolved_topk": {mid: int(extracted[mid]["resolved_topk"]) for mid in model_ids},
        "block_dim": int(next(iter(extracted.values()))["block_dim"]),
    }


def _sparse_values_from_extracted(info: Dict[str, Any]) -> Dict[int, float]:
    block_dim = int(info.get("block_dim", 0))
    values: Dict[int, float] = {}
    for base_idx, block in info["blocks"].items():
        flat = block.flatten().tolist()
        base_offset = int(base_idx) * block_dim
        for offset, value in enumerate(flat):
            val = float(value)
            if val != 0.0:
                values[base_offset + offset] = val
    return values


def _assign_average_ranks(items: List[Tuple[int, float]], start_rank: int) -> Tuple[Dict[int, float], float]:
    ranks: Dict[int, float] = {}
    sum_sq = 0.0
    pos = 0
    while pos < len(items):
        end = pos + 1
        current = items[pos][1]
        while end < len(items) and items[end][1] == current:
            end += 1
        avg_rank = (start_rank + pos + start_rank + end - 1) / 2.0
        for idx, _ in items[pos:end]:
            ranks[idx] = avg_rank
        sum_sq += float(end - pos) * (avg_rank ** 2)
        pos = end
    return ranks, sum_sq


def _build_sparse_rank_info(info: Dict[str, Any]) -> Dict[str, Any]:
    total_base_size = info.get("total_base_size")
    block_dim = int(info.get("block_dim", 0))
    total_length = int(total_base_size) * block_dim if isinstance(total_base_size, int) and total_base_size > 0 and block_dim > 0 else 0
    if total_length <= 0:
        dense = _dense_vector_from_extracted(info)
        ranks = _rankdata_average(dense)
        mu = (len(dense) + 1.0) / 2.0 if dense else 0.0
        sum_sq = sum(r * r for r in ranks)
        return {
            "total_length": len(dense),
            "zero_rank": 0.0,
            "nonzero_ranks": {idx: float(rank) for idx, rank in enumerate(ranks) if dense[idx] != 0.0},
            "sum_sq": float(sum_sq),
            "mean_rank": float(mu),
            "fallback_dense": True,
        }
    sparse_values = _sparse_values_from_extracted(info)
    negatives = sorted(((idx, val) for idx, val in sparse_values.items() if val < 0.0), key=lambda item: item[1])
    positives = sorted(((idx, val) for idx, val in sparse_values.items() if val > 0.0), key=lambda item: item[1])
    n_neg = len(negatives)
    n_pos = len(positives)
    n_zero = total_length - n_neg - n_pos
    if n_zero < 0:
        raise ValueError("Sparse Spearman bookkeeping produced a negative zero-count")
    negative_ranks, neg_sum_sq = _assign_average_ranks(negatives, start_rank=1)
    positive_ranks, pos_sum_sq = _assign_average_ranks(positives, start_rank=n_neg + n_zero + 1)
    zero_rank = n_neg + (n_zero + 1.0) / 2.0 if n_zero > 0 else 0.0
    nonzero_ranks = dict(negative_ranks)
    nonzero_ranks.update(positive_ranks)
    sum_sq = neg_sum_sq + pos_sum_sq + float(n_zero) * (zero_rank ** 2)
    return {
        "total_length": total_length,
        "zero_rank": float(zero_rank),
        "nonzero_ranks": nonzero_ranks,
        "sum_sq": float(sum_sq),
        "mean_rank": float((total_length + 1.0) / 2.0),
        "fallback_dense": False,
    }


def _sparse_spearman_from_rank_info(info_x: Dict[str, Any], info_y: Dict[str, Any]) -> float:
    n = int(info_x["total_length"])
    if n == 0 or n != int(info_y["total_length"]):
        return 0.0
    mean_rank = float(info_x["mean_rank"])
    ranks_x = info_x["nonzero_ranks"]
    ranks_y = info_y["nonzero_ranks"]
    zero_x = float(info_x["zero_rank"])
    zero_y = float(info_y["zero_rank"])
    union_keys = set(ranks_x.keys()) | set(ranks_y.keys())
    both_zero_count = n - len(union_keys)
    sum_xy = float(both_zero_count) * zero_x * zero_y
    for idx in union_keys:
        sum_xy += ranks_x.get(idx, zero_x) * ranks_y.get(idx, zero_y)
    centered_xy = sum_xy - float(n) * mean_rank * mean_rank
    centered_xx = float(info_x["sum_sq"]) - float(n) * mean_rank * mean_rank
    centered_yy = float(info_y["sum_sq"]) - float(n) * mean_rank * mean_rank
    if centered_xx <= 0.0 or centered_yy <= 0.0:
        return 0.0
    return float(centered_xy / math.sqrt(centered_xx * centered_yy))


def _compute_spearman_matrix(models: Dict[str, object], *, part: str, topk: Optional[Union[int, float]], take_abs: bool, measure_name: str) -> Dict[str, Any]:
    extracted = {mid: _extract_sparse_blocks(model, part=part, topk=topk) for mid, model in models.items()}
    model_ids = list(models.keys())
    rank_infos = {mid: _build_sparse_rank_info(extracted[mid]) for mid in model_ids}
    fallback_dense = any(bool(rank_infos[mid].get("fallback_dense")) for mid in model_ids)
    matrix = [[0.0] * len(model_ids) for _ in range(len(model_ids))]
    for i, mi in enumerate(model_ids):
        for j, mj in enumerate(model_ids):
            score = _sparse_spearman_from_rank_info(rank_infos[mi], rank_infos[mj])
            matrix[i][j] = abs(score) if take_abs else score
    return {
        "measure": measure_name,
        "model_ids": model_ids,
        "matrix": matrix,
        "part": part,
        "topk": topk,
        "resolved_topk": {mid: int(extracted[mid]["resolved_topk"]) for mid in model_ids},
        "full_zero_filled": not fallback_dense,
        "sparse_exact": not fallback_dense,
        "fallback_dense": fallback_dense,
    }


def _compute_mass_overlap_matrix(models: Dict[str, object], *, part: str, topk: Union[int, float]) -> Dict[str, Any]:
    extracted = {mid: _extract_importance_by_index(model, part=part, topk=topk) for mid, model in models.items()}
    model_ids = list(models.keys())
    matrix = [[0.0] * len(model_ids) for _ in range(len(model_ids))]
    for i, mi in enumerate(model_ids):
        imp_i = extracted[mi]["importance"]
        set_i = set(imp_i.keys())
        mass_i = math.sqrt(sum(v * v for v in imp_i.values()))
        for j, mj in enumerate(model_ids):
            imp_j = extracted[mj]["importance"]
            set_j = set(imp_j.keys())
            mass_j = math.sqrt(sum(v * v for v in imp_j.values()))
            if mass_i == 0.0 or mass_j == 0.0:
                matrix[i][j] = 0.0
                continue
            shared = set_i & set_j
            shared_mass_i = math.sqrt(sum(imp_i[k] * imp_i[k] for k in shared))
            shared_mass_j = math.sqrt(sum(imp_j[k] * imp_j[k] for k in shared))
            matrix[i][j] = float(0.5 * ((shared_mass_i / mass_i) + (shared_mass_j / mass_j)))
    return {
        "measure": "mass_overlap",
        "model_ids": model_ids,
        "matrix": matrix,
        "part": part,
        "topk": topk,
        "resolved_topk": {mid: int(extracted[mid]["resolved_topk"]) for mid in model_ids},
    }


def _grouped_blocks(extracted: Dict[str, Dict[str, Any]], grouping: GroupingSpec) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
    all_groups: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    for model_id, info in extracted.items():
        grouped: Dict[str, Dict[int, torch.Tensor]] = {}
        for base_idx, block in info["blocks"].items():
            param_name = info.get("param_names", {}).get(base_idx, "other")
            group_name = _group_name_from_param(param_name, grouping)
            grouped.setdefault(group_name, {})[base_idx] = block
        for group_name, group_blocks in grouped.items():
            all_groups.setdefault(group_name, {})[model_id] = group_blocks
    return all_groups


def compute_grouped_similarity_matrices(models: Dict[str, object], *, measure: str = "cosine", part: Optional[str] = None, topk: Optional[Union[int, float]] = None, group_by: GroupingSpec = "param") -> Dict[str, Dict[str, Any]]:
    """Compute one similarity matrix per parameter group.

    This is the grouped variant of :func:`compute_similarity_matrix`. It first
    selects GRADIEND coordinates via ``part`` and ``topk``, assigns each decoded
    base-model parameter name to a group, and then computes one square
    model-by-model matrix for every group.

    Args:
        models: Mapping from display/model id to a trained model with a
            ``.gradiend`` attribute. At least two models are required.
        measure: Similarity measure. Supported values are ``"cosine"``,
            ``"cosine_signed"``, ``"spearman"``, and ``"spearman_signed"``.
            Unsigned variants take the absolute value.
        part: GRADIEND part to compare. Supported values are
            ``"encoder-weight"``, ``"decoder-weight"``, ``"decoder-bias"``,
            and ``"decoder-sum"``. Defaults to ``"encoder-weight"``.
        topk: Optional coordinate selection. ``None`` uses all coordinates, an
            integer selects that many top coordinates, and a float in ``(0, 1]``
            selects that fraction using the model's own ``get_topk_weights``.
        group_by: Grouping strategy. ``"param"`` keeps decoded parameter names
            separate, ``"layer"`` extracts layer numbers from common transformer
            names, and ``"component"`` uses common transformer component naming
            conventions. Component grouping is heuristic and may not classify
            every architecture perfectly. A dict maps exact parameter names to
            labels, and a callable receives each parameter name and returns a
            label. Coordinates whose parameter name cannot be decoded are put
            into ``"other"``.

    Returns:
        A dict keyed by group name. Each value contains ``measure``, ``group``,
        ``group_by``, ``model_ids``, ``matrix``, ``part``, and ``topk``.

    Raises:
        TypeError: If ``models`` or ``topk`` has an invalid type.
        ValueError: If fewer than two models are passed, ``measure`` is
            unsupported, ``part`` is unsupported, or ``group_by`` is invalid.
    """
    _validate_models(models)
    _validate_topk_optional(topk)
    measure = (measure or "cosine").lower()
    if measure not in {"cosine", "cosine_signed", "spearman", "spearman_signed"}:
        raise ValueError("Grouped similarities currently support cosine/cosine_signed/spearman/spearman_signed")
    resolved_part = (part or "encoder-weight").lower()
    extracted = {mid: _extract_sparse_blocks(model, part=resolved_part, topk=topk, include_param_names=True) for mid, model in models.items()}
    grouped = _grouped_blocks(extracted, group_by)
    out: Dict[str, Dict[str, Any]] = {}
    model_ids = list(models.keys())
    for group_name, group_blocks in grouped.items():
        matrix = [[0.0] * len(model_ids) for _ in range(len(model_ids))]
        for i, mi in enumerate(model_ids):
            for j, mj in enumerate(model_ids):
                blocks_i = group_blocks.get(mi, {})
                blocks_j = group_blocks.get(mj, {})
                if measure.startswith("cosine"):
                    score = _cosine_from_blocks(blocks_i, blocks_j)
                else:
                    vec_i, vec_j = _vector_from_union(blocks_i, blocks_j)
                    score = 0.0 if not vec_i else _pearson(_rankdata_average(vec_i), _rankdata_average(vec_j))
                matrix[i][j] = abs(score) if measure in {"cosine", "spearman"} else score
        out[group_name] = {
            "measure": measure,
            "group": group_name,
            "group_by": "param" if group_by is None else str(group_by),
            "model_ids": model_ids,
            "matrix": matrix,
            "part": resolved_part,
            "topk": topk,
        }
    return out


def _score_similarity_pair(
    model_i: object,
    model_j: object,
    *,
    measure: str,
    part: str,
    topk: Optional[Union[int, float]],
    value: str,
) -> float:
    if measure == "cosine":
        return abs(_cosine_from_blocks(
            _extract_sparse_blocks(model_i, part=part, topk=topk)["blocks"],
            _extract_sparse_blocks(model_j, part=part, topk=topk)["blocks"],
        ))
    if measure == "cosine_signed":
        return _cosine_from_blocks(
            _extract_sparse_blocks(model_i, part=part, topk=topk)["blocks"],
            _extract_sparse_blocks(model_j, part=part, topk=topk)["blocks"],
        )
    if measure == "spearman":
        info_i = _build_sparse_rank_info(_extract_sparse_blocks(model_i, part=part, topk=topk))
        info_j = _build_sparse_rank_info(_extract_sparse_blocks(model_j, part=part, topk=topk))
        return abs(_sparse_spearman_from_rank_info(info_i, info_j))
    if measure == "spearman_signed":
        info_i = _build_sparse_rank_info(_extract_sparse_blocks(model_i, part=part, topk=topk))
        info_j = _build_sparse_rank_info(_extract_sparse_blocks(model_j, part=part, topk=topk))
        return _sparse_spearman_from_rank_info(info_i, info_j)
    if measure == "topk_overlap":
        if topk is None:
            raise ValueError("topk must be provided for measure='topk_overlap'")
        indices_i = model_i.get_topk_weights(part=part, topk=topk)
        indices_j = model_j.get_topk_weights(part=part, topk=topk)
        inter = len(set(indices_i) & set(indices_j))
        if value == "intersection":
            return float(inter)
        if value == "intersection_frac":
            return _resolve_overlap_fraction(len(indices_i), len(indices_j), inter)
        raise ValueError("value must be 'intersection' or 'intersection_frac'")
    if measure == "mass_overlap":
        if topk is None:
            raise ValueError("topk must be provided for measure='mass_overlap'")
        imp_i = _extract_importance_by_index(model_i, part=part, topk=topk)["importance"]
        imp_j = _extract_importance_by_index(model_j, part=part, topk=topk)["importance"]
        set_i = set(imp_i.keys())
        set_j = set(imp_j.keys())
        mass_i = math.sqrt(sum(v * v for v in imp_i.values()))
        mass_j = math.sqrt(sum(v * v for v in imp_j.values()))
        if mass_i == 0.0 or mass_j == 0.0:
            return 0.0
        shared = set_i & set_j
        shared_mass_i = math.sqrt(sum(imp_i[k] * imp_i[k] for k in shared))
        shared_mass_j = math.sqrt(sum(imp_j[k] * imp_j[k] for k in shared))
        return float(0.5 * ((shared_mass_i / mass_i) + (shared_mass_j / mass_j)))
    raise ValueError(
        "measure must be one of 'cosine', 'cosine_signed', 'spearman', 'spearman_signed', 'topk_overlap', or 'mass_overlap'"
    )


def compute_similarity_matrix(
    models: Dict[str, object],
    *,
    measure: str = "cosine",
    part: Optional[str] = None,
    topk: Optional[Union[int, float]] = None,
    value: str = "intersection_frac",
    seed_aggregate: str = "mean",
    dispersion: str = "none",
) -> Dict[str, Any]:
    """Compute a pairwise similarity matrix for trained GRADIEND models.

    Args:
        models: Mapping from display/model id to a trained model. For
            ``"topk_overlap"``, models must provide ``get_topk_weights``. For
            vector and mass measures, models must provide a ``.gradiend`` object
            with built encoder/decoder weights. Values may also be non-empty
            lists or tuples of seed models. At least two model ids are required.
        measure: Similarity measure. Supported values are ``"cosine"``,
            ``"cosine_signed"``, ``"spearman"``, ``"spearman_signed"``,
            ``"topk_overlap"``, and ``"mass_overlap"``. Unsigned ``cosine`` and
            ``spearman`` take absolute values.
        part: GRADIEND part to compare. Supported values for vector/mass
            measures are ``"encoder-weight"``, ``"decoder-weight"``,
            ``"decoder-bias"``, and ``"decoder-sum"``. If omitted,
            ``"topk_overlap"`` uses ``"decoder-weight"`` and all other measures
            use ``"encoder-weight"``.
        topk: Optional coordinate selection. Required for ``"topk_overlap"``
            and ``"mass_overlap"``. ``None`` uses all coordinates where allowed,
            an integer selects that many top coordinates, and a float in
            ``(0, 1]`` selects that fraction using each model's own
            ``get_topk_weights`` implementation.
        value: Output value for ``"topk_overlap"``. ``"intersection_frac"``
            divides the intersection by the smaller selected set size.
            ``"intersection"`` returns the raw intersection count. Ignored by
            other measures.
        seed_aggregate: Aggregation used when a model id maps to a list/tuple of
            seed models. Supported values are ``"mean"``, ``"median"``,
            ``"min"``, and ``"max"``.
        dispersion: Optional dispersion metadata for list/tuple seed cells.
            Supported values are ``"none"``, ``"std"``, ``"range"``, and
            ``"minmax"``. ``"range"`` is incompatible with ``seed_aggregate`` of
            ``"min"`` or ``"max"``.

    Returns:
        For one model per id, a payload with ``measure``, ``model_ids``,
        ``matrix``, ``part``, ``topk``, and measure-specific metadata such as
        ``value``, ``resolved_topk``, ``per_model``, ``block_dim``,
        ``full_zero_filled``, ``sparse_exact``, or ``fallback_dense``. For
        list/tuple seed groups, every cell aggregates all seed-pair scores,
        diagonal cells include self-pairs, and the payload also contains
        ``seed_aggregate``, ``dispersion``, ``n_matrix``, ``cell_stats``,
        ``multi_seed=True``, plus ``global_n`` or ``global_n_range`` when
        available.

    Raises:
        TypeError: If ``models`` or ``topk`` has an invalid type, or vector/mass
            measures receive models without a ``.gradiend`` attribute.
        ValueError: If fewer than two model ids are passed, a model group is
            empty, ``measure``/``part``/``value`` is unsupported, required
            ``topk`` is missing, ``topk`` is out of range, or seed aggregation
            and dispersion settings are incompatible.
    """
    model_groups = _normalize_model_groups(models)
    _validate_topk_optional(topk)
    measure = (measure or "cosine").lower()
    resolved_part = (part or ("decoder-weight" if measure == "topk_overlap" else "encoder-weight")).lower()
    _validate_aggregate_dispersion_combo(seed_aggregate, dispersion)
    if all(len(group) == 1 for group in model_groups.values()):
        flat_models = {mid: group[0] for mid, group in model_groups.items()}
        _validate_models(flat_models)
        if measure == "cosine":
            return _compute_cosine_like_matrix(flat_models, part=resolved_part, topk=topk, take_abs=True, measure_name="cosine")
        if measure == "cosine_signed":
            return _compute_cosine_like_matrix(flat_models, part=resolved_part, topk=topk, take_abs=False, measure_name="cosine_signed")
        if measure == "spearman":
            return _compute_spearman_matrix(flat_models, part=resolved_part, topk=topk, take_abs=True, measure_name="spearman")
        if measure == "spearman_signed":
            return _compute_spearman_matrix(flat_models, part=resolved_part, topk=topk, take_abs=False, measure_name="spearman_signed")
        if measure == "topk_overlap":
            if topk is None:
                raise ValueError("topk must be provided for measure='topk_overlap'")
            return _compute_topk_overlap_similarity_matrix(flat_models, part=resolved_part, topk=topk, value=value)
        if measure == "mass_overlap":
            if topk is None:
                raise ValueError("topk must be provided for measure='mass_overlap'")
            return _compute_mass_overlap_matrix(flat_models, part=resolved_part, topk=topk)
        raise ValueError(
            "measure must be one of 'cosine', 'cosine_signed', 'spearman', 'spearman_signed', 'topk_overlap', or 'mass_overlap'"
        )

    model_ids = list(model_groups.keys())
    matrix = [[0.0] * len(model_ids) for _ in range(len(model_ids))]
    n_matrix = [[0] * len(model_ids) for _ in range(len(model_ids))]
    cell_stats: List[List[Dict[str, Any]]] = []
    all_n: List[int] = []
    for i, mi in enumerate(model_ids):
        stats_row: List[Dict[str, Any]] = []
        for j, mj in enumerate(model_ids):
            scores = [
                _score_similarity_pair(left_model, right_model, measure=measure, part=resolved_part, topk=topk, value=value)
                for left_model in model_groups[mi]
                for right_model in model_groups[mj]
            ]
            stats = _aggregate_seed_scores(scores, seed_aggregate=seed_aggregate, dispersion=dispersion)
            matrix[i][j] = float(stats["aggregate"])
            n_matrix[i][j] = int(stats["n"])
            stats_row.append(stats)
            all_n.append(int(stats["n"]))
        cell_stats.append(stats_row)
    payload: Dict[str, Any] = {
        "measure": measure,
        "model_ids": model_ids,
        "matrix": matrix,
        "part": resolved_part,
        "topk": topk,
        "value": value,
        "seed_aggregate": seed_aggregate,
        "dispersion": dispersion,
        "n_matrix": n_matrix,
        "cell_stats": cell_stats,
        "multi_seed": True,
    }
    if all_n:
        if min(all_n) == max(all_n):
            payload["global_n"] = int(all_n[0])
        else:
            payload["global_n_range"] = [int(min(all_n)), int(max(all_n))]
    return payload
