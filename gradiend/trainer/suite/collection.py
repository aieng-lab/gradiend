"""Combine pre-built trainers without pair-definition generation."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple, Union

from gradiend.trainer.trainer import Trainer

from .base import TrainerSuite


def _require_trainer_run_id(trainer: Trainer) -> str:
    run_id = getattr(trainer, "run_id", None)
    if run_id is None or not str(run_id).strip():
        raise ValueError(
            "Every trainer in a TrainerCollection must have a non-empty run_id. "
            f"Got {trainer!r}."
        )
    return str(run_id).strip()


class TrainerCollection:
    """
    Group trainers that are already built.

    Pass trainers directly; ids come from ``trainer.run_id``. When flattening a
    :class:`TrainerSuite`, ids are the suite child ids from ``suite.items()``
    (which match ``trainer.run_id`` when the suite has no parent ``run_id``).
    """

    def __init__(
        self,
        *trainers: Trainer,
        retain_models_in_memory: bool = True,
    ) -> None:
        self.retain_models_in_memory = bool(retain_models_in_memory)
        self.trainers: Dict[str, Trainer] = {}
        for trainer in trainers:
            self._add_trainer(trainer)

    @classmethod
    def merge(
        cls,
        *parts: Union[Trainer, TrainerSuite, TrainerCollection],
        retain_models_in_memory: bool = True,
    ) -> TrainerCollection:
        """Combine trainers, suites, and collections into one group."""
        collection = cls(retain_models_in_memory=retain_models_in_memory)
        for part in parts:
            collection._add_part(part)
        return collection

    def _add_part(self, part: Union[Trainer, TrainerSuite, TrainerCollection]) -> None:
        if isinstance(part, TrainerCollection):
            for trainer_id, trainer in part.trainers.items():
                self._add_trainer_with_id(trainer_id, trainer)
        elif isinstance(part, TrainerSuite):
            for child_id, trainer in part.items():
                self._add_trainer_with_id(str(child_id), trainer)
        elif isinstance(part, Trainer):
            self._add_trainer(part)
        else:
            raise TypeError(
                "TrainerCollection.merge expected Trainer, TrainerSuite, or TrainerCollection; "
                f"got {type(part).__name__}"
            )

    def _add_trainer(self, trainer: Trainer) -> None:
        self._add_trainer_with_id(_require_trainer_run_id(trainer), trainer)

    def _add_trainer_with_id(self, trainer_id: str, trainer: Trainer) -> None:
        if trainer_id in self.trainers:
            raise ValueError(f"Duplicate trainer id {trainer_id!r}")
        self.trainers[trainer_id] = trainer

    def train(self, *, use_cache: bool = True) -> None:
        for trainer in self.trainers.values():
            trainer.train(use_cache=use_cache)
            used_cache = bool(getattr(trainer, "_last_train_used_cache", False))
            if not self.retain_models_in_memory and not used_cache and hasattr(trainer, "unload_model"):
                trainer.unload_model()

    def __len__(self) -> int:
        return len(self.trainers)

    def __iter__(self) -> Iterator[str]:
        return iter(self.trainers)

    def keys(self) -> Iterable[str]:
        return self.trainers.keys()

    def values(self) -> Iterable[Trainer]:
        return self.trainers.values()

    def items(self) -> Iterable[Tuple[str, Trainer]]:
        return self.trainers.items()

    def get_trainer(self, trainer_id: str) -> Trainer:
        return self.trainers[trainer_id]
