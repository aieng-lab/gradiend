"""Model containers for multi-seed comparison and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Sequence


@dataclass(frozen=True)
class SeedModelGroup:
    """A set of GRADIEND checkpoints selected for one trainer analysis run.

    ``compute_similarity_matrix`` and top-k overlap helpers accept a
    ``SeedModelGroup`` wherever a single model is allowed: pairwise scores are
    aggregated across all seed pairs automatically.
    """

    models: Tuple[Any, ...]
    selection: str
    aggregate: str
    dispersion: str
    seed_values: Tuple[int, ...]

    def __init__(
        self,
        models: Sequence[Any],
        *,
        selection: str = "all_convergent",
        aggregate: str = "mean",
        dispersion: str = "none",
        seed_values: Optional[Sequence[int]] = None,
    ) -> None:
        model_list = tuple(models)
        object.__setattr__(self, "models", model_list)
        object.__setattr__(self, "selection", str(selection))
        object.__setattr__(self, "aggregate", str(aggregate))
        object.__setattr__(self, "dispersion", str(dispersion))
        if seed_values is None:
            seed_values = tuple(range(len(model_list)))
        object.__setattr__(self, "seed_values", tuple(int(v) for v in seed_values))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    @property
    def primary(self) -> Any:
        """First seed model (best/primary checkpoint ordering)."""
        if not self.models:
            raise ValueError("SeedModelGroup is empty")
        return self.models[0]

    def __bool__(self) -> bool:
        return bool(self.models)
