"""
Utilities for selecting extra transition groups during encoder evaluation.

These helpers are intentionally lightweight and text-agnostic. They describe
which additional transitions, beyond the active target pair, should be included
when evaluation data is constructed.

Two common transition types are supported:

- ``pair(a, b, symmetric=True)`` for contrastive updates such as ``happy -> sad``.
- ``identity(a)`` for neutral/self updates such as ``calm -> calm``.

The public evaluation API can accept a sequence of these transition specs via
``transition_selection=...``. Non-target transitions selected this way keep
their natural factual/alternative ids but receive label ``0`` during encoder
evaluation; the current target pair remains the only supervised ``+1/-1`` axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Literal, Set, Tuple


TransitionEdge = Tuple[str, str]


@dataclass(frozen=True)
class TransitionSpec:
    """
    Describe an extra transition group to include during evaluation.

    Attributes:
        kind:
            Either ``"pair"`` for contrastive transitions or ``"identity"``
            for self/self transitions.
        source:
            Source class id.
        target:
            Target class id. For ``kind="identity"``, this must equal
            ``source``.
        symmetric:
            Whether to include both ``source -> target`` and ``target -> source``.
            This is meaningful for ``kind="pair"`` and ignored for
            ``kind="identity"``.
    """

    kind: Literal["pair", "identity"]
    source: str
    target: str
    symmetric: bool = True

    def edges(self) -> FrozenSet[TransitionEdge]:
        """
        Expand the spec into concrete directed transitions.

        Returns:
            A frozenset of ``(source, target)`` tuples.
        """
        if self.kind == "identity":
            return frozenset({(self.source, self.source)})
        if self.symmetric:
            return frozenset({(self.source, self.target), (self.target, self.source)})
        return frozenset({(self.source, self.target)})


def pair(source: str, target: str, *, symmetric: bool = True) -> TransitionSpec:
    """
    Create a contrastive transition spec.

    Args:
        source:
            Source class id.
        target:
            Target class id.
        symmetric:
            If True (default), include both ``source -> target`` and
            ``target -> source`` during evaluation.

    Returns:
        TransitionSpec describing the pair.
    """
    return TransitionSpec(kind="pair", source=str(source), target=str(target), symmetric=bool(symmetric))


def identity(class_id: str) -> TransitionSpec:
    """
    Create an identity-transition spec.

    Args:
        class_id:
            Class id for the self/self transition.

    Returns:
        TransitionSpec describing ``class_id -> class_id``.
    """
    cls = str(class_id)
    return TransitionSpec(kind="identity", source=cls, target=cls, symmetric=False)


def expand_transition_selection(
    selection: Iterable[TransitionSpec | Tuple[str, str]],
) -> FrozenSet[TransitionEdge]:
    """
    Expand transition specs or raw edge tuples into directed transitions.

    Args:
        selection:
            Iterable of ``TransitionSpec`` objects or raw ``(source, target)``
            tuples. Raw tuples are treated as directed transitions.

    Returns:
        Frozenset of directed ``(source, target)`` tuples.
    """
    edges: Set[TransitionEdge] = set()
    for item in selection:
        if isinstance(item, TransitionSpec):
            edges.update(item.edges())
        else:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "transition_selection entries must be TransitionSpec instances "
                    f"or (source, target) tuples, got {type(item).__name__}."
                )
            src, tgt = item
            edges.add((str(src), str(tgt)))
    return frozenset(edges)
