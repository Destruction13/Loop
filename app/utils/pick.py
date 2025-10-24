"""Utilities for picking catalog models."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence

from app.models import ModelItem


def pick_four(models: Sequence[ModelItem], seen: Iterable[str], limit: int = 4) -> List[ModelItem]:
    """Pick up to four models preferring unseen ones and rotating shapes."""

    seen_set = set(seen)
    unseen = [model for model in models if model.model_id not in seen_set]
    groups = defaultdict(list)
    for model in unseen:
        groups[model.meta.shape or "unknown"].append(model)

    picked: List[ModelItem] = []
    while len(picked) < limit and groups:
        for shape in list(groups.keys()):
            if groups[shape]:
                picked.append(groups[shape].pop(0))
                if len(picked) >= limit:
                    break
            if not groups[shape]:
                groups.pop(shape, None)

    if len(picked) < limit:
        remaining = [model for model in models if model not in picked]
        for model in remaining:
            if len(picked) >= limit:
                break
            picked.append(model)

    return picked[:limit]
