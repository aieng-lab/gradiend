"""
Filter configuration for text: targets, spacy tags, lemma matching.

Not prediction-specific; used by text filtering in general.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Type alias for spacy morph/tag constraints (e.g. {"pos": "DET", "Case": "Nom"})
SpacyTagSpec = Dict[str, str]


def _merge_spacy_tags(parent: Optional[SpacyTagSpec], child: Optional[SpacyTagSpec]) -> Optional[SpacyTagSpec]:
    """Merge child tags into parent; child overrides parent for same keys."""
    if parent is None and child is None:
        return None
    if parent is None:
        return dict(child) if child else None
    if child is None:
        return dict(parent)
    out = dict(parent)
    out.update(child)
    return out


def _first_target_string(targets: List[Union["TextFilterConfig", str]]) -> Optional[str]:
    """Extract first target string from targets for default id."""
    for t in targets:
        if isinstance(t, str):
            return t
        if isinstance(t, TextFilterConfig):
            s = _first_target_string(t.targets)
            if s is not None:
                return s
    return None


@dataclass
class TextFilterConfig:
    """Configuration for filtering and masking target tokens in text.

    Simple use: TextFilterConfig(target="das") or TextFilterConfig(targets=["der", "die"], spacy_tags={...})
    Advanced: TextFilterConfig(targets=[TextFilterConfig(["das"], spacy_tags={...}), "der"], spacy_tags={...})
    Nested entries inherit parent spacy_tags and may add/override their own.

    When used in TextPredictionDataCreator.feature_targets (a list), id names the output class.
    If id is None, it is inferred from the first target string.
    """

    targets: Optional[List[Union["TextFilterConfig", str]]] = None
    target: Optional[str] = None
    spacy_tags: Optional[SpacyTagSpec] = None
    use_lemma: bool = False
    id: Optional[str] = None
    mask: str = "[MASK]"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.target is not None and self.targets is not None:
            raise ValueError("TextFilterConfig: provide either target or targets, not both")
        if self.target is not None:
            self.targets = [self.target]
        elif isinstance(self.targets, str):
            self.targets = [self.targets]
        if self.targets is None:
            self.targets = []
        if not self.targets:
            raise ValueError("TextFilterConfig.targets must be non-empty")
        if self.id is None:
            self.id = _first_target_string(self.targets)

    def get_effective_tags(self, parent_tags: Optional[SpacyTagSpec] = None) -> Optional[SpacyTagSpec]:
        """Get merged spacy_tags (self + parent)."""
        return _merge_spacy_tags(parent_tags, self.spacy_tags)

    def flatten_targets_with_tags(
        self,
        parent_tags: Optional[SpacyTagSpec] = None,
    ) -> List[tuple[str, Optional[SpacyTagSpec]]]:
        """Flatten nested config to list of (target_str, effective_spacy_tags)."""
        effective = self.get_effective_tags(parent_tags)
        out: List[tuple[str, Optional[SpacyTagSpec]]] = []
        for t in self.targets:
            if isinstance(t, str):
                out.append((t, effective))
            elif isinstance(t, TextFilterConfig):
                out.extend(t.flatten_targets_with_tags(effective))
            else:
                raise TypeError(f"target must be str or TextFilterConfig; got {type(t)}")
        return out
