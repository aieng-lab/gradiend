"""Derive display labels for top-k overlap plots from checkpoint locations.

``plot_topk_overlap`` and friends take a mapping ``{label: model}``. For the common case of
comparing trained trainers it is convenient to pass just the models,
``[trainer.get_model() for trainer in trainers]``, and let the plot name them after their
checkpoint folders (e.g. ``runs/3SG_3PL/model`` -> ``"3SG_3PL"``).
"""

import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

# Folder names that carry no information about the feature (trainer layout: <run>/model,
# multi-seed layout: <run>/seeds/seed_42/model); the first other parent folder is used instead.
_GENERIC_FOLDER = re.compile(r"^(model|models|seeds?|seed[_-]?\d+|checkpoints?|checkpoint-\d+|best)$", re.IGNORECASE)


def _first_member(model: Any) -> Any:
    """Return a representative model for seed groups / seed lists, else the model itself."""
    if isinstance(model, (list, tuple)):
        return model[0] if model else None
    members = getattr(model, "models", None)
    if isinstance(members, (list, tuple)) and members:
        return members[0]
    return model


def checkpoint_label(model: Any) -> Optional[str]:
    """Return a label derived from the model's checkpoint folder, or ``None`` if unavailable."""
    model = _first_member(model)
    if model is None:
        return None
    gradiend = getattr(model, "gradiend", None)
    path = getattr(gradiend, "name_or_path", None) or getattr(model, "name_or_path", None)
    if not path or not isinstance(path, (str, Path)):
        return None
    # Accept both separators: a checkpoint path written on Windows is read on Linux and vice versa.
    for part in reversed(PurePosixPath(str(path).replace("\\", "/")).parts):
        if part in ("", ".", "..") or part.endswith((":", ":\\", "/")):
            continue
        if not _GENERIC_FOLDER.match(part):
            return part
    return None


def as_named_models(models: Any) -> Any:
    """Return ``models`` as a ``{label: model}`` dict.

    A dict is returned unchanged. A list/tuple of models is labelled after each model's checkpoint
    folder. Raises ``ValueError`` if a label cannot be derived or two models would get the same
    label, because anonymous labels such as "model 3" would make the plot meaningless; pass an
    explicit ``{label: model}`` dict in that case.
    """
    if not isinstance(models, (list, tuple)):
        return models
    labels: List[Optional[str]] = [checkpoint_label(m) for m in models]
    missing = [i for i, lab in enumerate(labels) if lab is None]
    if missing:
        raise ValueError(
            f"Cannot derive plot labels for the models at positions {missing}: they have no checkpoint "
            "path. Pass a dict {label: model} instead of a list."
        )
    duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
    if duplicates:
        raise ValueError(
            f"Checkpoint folder names are not unique ({duplicates}), so they cannot label the models. "
            "Pass a dict {label: model} instead of a list."
        )
    return dict(zip(labels, models))
