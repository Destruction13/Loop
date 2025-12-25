"""Style recommendation helpers using Thompson Sampling on Beta priors."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from app.models import GlassModel, STYLE_UNKNOWN
from app.services.repository import Repository

VOTE_OK = "ok"
VOTE_DUPLICATE = "duplicate"
VOTE_INVALID = "invalid"


class StyleRecommender:
    """Encapsulates style preference logic and voting updates."""

    def __init__(
        self,
        repository: Repository,
        *,
        rng: random.Random | None = None,
        exploration_rate: float = 0.2,
    ) -> None:
        self._repository = repository
        self._rng = rng or random.Random()
        self._exploration_rate = min(max(exploration_rate, 0.0), 1.0)

    async def has_feedback(self, user_id: int) -> bool:
        """Return True when the user has at least one style vote."""

        return await self._repository.has_style_feedback(user_id)

    async def ensure_user_styles(
        self, user_id: int, styles_from_catalog: Iterable[str]
    ) -> None:
        """Ensure user preferences exist for every catalog style."""

        styles = _unique_styles(styles_from_catalog)
        if not styles:
            return
        await self._repository.ensure_style_preferences(user_id, styles)

    async def select_styles_for_collage(
        self,
        user_id: int,
        available_styles: Iterable[str],
        *,
        n: int = 2,
        exploration_rate: float | None = None,
    ) -> list[str]:
        """Select styles for the next collage respecting feedback state."""

        if n == 2:
            pair = await self.select_style_pair_for_collage(
                user_id,
                available_styles,
                exploration_rate=exploration_rate,
                allow_duplicates=True,
            )
            return list(pair)

        styles = _unique_styles(available_styles)
        if not styles or n <= 0:
            return []

        if not await self.has_feedback(user_id):
            return _sample_styles(self._rng, styles, n)

        prefs = await self._repository.list_style_preferences(user_id, styles=styles)
        scores = {
            style: self._rng.betavariate(*prefs.get(style, (1, 1)))
            for style in styles
        }
        rate = (
            self._exploration_rate
            if exploration_rate is None
            else min(max(exploration_rate, 0.0), 1.0)
        )
        return _select_with_exploration(self._rng, styles, scores, n, rate)

    async def select_style_pair_for_collage(
        self,
        user_id: int,
        available_styles: Iterable[str],
        *,
        exploration_rate: float | None = None,
        allow_duplicates: bool = True,
    ) -> tuple[str, str]:
        """Select a style pair using Thompson Sampling with exploration."""

        styles = _unique_styles(available_styles)
        if not styles:
            return tuple()
        if len(styles) == 1:
            only = styles[0]
            return (only, only)

        if not await self.has_feedback(user_id):
            weights = [1.0] * len(styles)
            return _select_style_pair(
                self._rng,
                styles,
                weights,
                allow_duplicates=allow_duplicates,
            )

        rate = (
            self._exploration_rate
            if exploration_rate is None
            else min(max(exploration_rate, 0.0), 1.0)
        )
        use_random = self._rng.random() < rate
        if use_random:
            weights = [1.0] * len(styles)
            return _select_style_pair(
                self._rng,
                styles,
                weights,
                allow_duplicates=allow_duplicates,
            )

        prefs = await self._repository.list_style_preferences(user_id, styles=styles)
        weights = [
            self._rng.betavariate(*prefs.get(style, (1, 1))) for style in styles
        ]
        return _select_style_pair(
            self._rng,
            styles,
            weights,
            allow_duplicates=allow_duplicates,
        )

    async def rank_styles_for_collage(
        self, user_id: int, available_styles: Iterable[str]
    ) -> list[str]:
        """Rank styles by current Thompson scores for model fallbacks."""

        styles = _unique_styles(available_styles)
        if not styles:
            return []
        if not await self.has_feedback(user_id):
            return styles
        prefs = await self._repository.list_style_preferences(user_id, styles=styles)
        scored = [
            (self._rng.betavariate(*prefs.get(style, (1, 1))), style)
            for style in styles
        ]
        scored.sort(reverse=True)
        return [style for _, style in scored]

    def select_models_for_collage(
        self,
        candidates: Sequence[GlassModel],
        styles: Sequence[str],
        *,
        n: int = 2,
        fallback_styles: Sequence[str] | None = None,
    ) -> list[GlassModel]:
        """Pick models for the collage with style-based fallbacks."""

        unique_models: list[GlassModel] = []
        seen_ids: set[str] = set()
        for model in candidates:
            if model.unique_id in seen_ids:
                continue
            seen_ids.add(model.unique_id)
            unique_models.append(model)
        if not unique_models or n <= 0:
            return []

        buckets: dict[str, list[GlassModel]] = {}
        for model in unique_models:
            style = (model.style or STYLE_UNKNOWN).strip() or STYLE_UNKNOWN
            buckets.setdefault(style, []).append(model)

        selected: list[GlassModel] = []
        used_ids: set[str] = set()
        fallback_order = _unique_styles(fallback_styles or [])
        if not fallback_order:
            fallback_order = list(buckets.keys())

        def _pick_from_pool(pool: Sequence[GlassModel]) -> GlassModel | None:
            remaining = [model for model in pool if model.unique_id not in used_ids]
            if not remaining:
                return None
            return self._rng.choice(remaining)

        def _pick_from_fallback() -> GlassModel | None:
            for style in fallback_order:
                choice = _pick_from_pool(buckets.get(style, []))
                if choice is not None:
                    return choice
            return None

        for style in styles:
            if len(selected) >= n:
                break
            choice = _pick_from_pool(buckets.get(style, []))
            if choice is None:
                choice = _pick_from_fallback()
            if choice is None:
                choice = _pick_from_pool(unique_models)
            if choice is None:
                break
            selected.append(choice)
            used_ids.add(choice.unique_id)

        if len(selected) < n:
            while len(selected) < n:
                choice = _pick_from_fallback()
                if choice is None:
                    choice = _pick_from_pool(unique_models)
                if choice is None:
                    break
                selected.append(choice)
                used_ids.add(choice.unique_id)
            remaining = [
                model for model in unique_models if model.unique_id not in used_ids
            ]
            while len(selected) < n and remaining:
                choice = self._rng.choice(remaining)
                selected.append(choice)
                used_ids.add(choice.unique_id)
                remaining = [
                    model
                    for model in unique_models
                    if model.unique_id not in used_ids
                ]
            while len(selected) < n and unique_models:
                selected.append(self._rng.choice(unique_models))

        return selected

    async def update_from_vote(
        self,
        user_id: int,
        generation_id: str,
        style: str,
        vote: str,
    ) -> str:
        """Apply a vote and update the corresponding Beta parameters."""

        clean_style = (style or "").strip()
        if not generation_id or not clean_style:
            return VOTE_INVALID
        if vote not in {"like", "dislike"}:
            return VOTE_INVALID

        inserted = await self._repository.insert_style_vote(
            user_id=user_id,
            generation_id=generation_id,
            style=clean_style,
            vote=vote,
        )
        if not inserted:
            return VOTE_DUPLICATE

        await self._repository.ensure_style_preferences(user_id, [clean_style])
        if vote == "like":
            await self._repository.increment_style_preference(
                user_id, clean_style, alpha_inc=1
            )
        else:
            await self._repository.increment_style_preference(
                user_id, clean_style, beta_inc=1
            )
        return VOTE_OK

    async def get_debug_snapshot(self, user_id: int) -> dict[str, object]:
        """Return preference stats for logs/debugging."""

        prefs = await self._repository.list_style_preferences(user_id)
        scores = {
            style: self._rng.betavariate(alpha, beta)
            for style, (alpha, beta) in prefs.items()
        }
        return {"prefs": prefs, "scores": scores}


def _unique_styles(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        style = (value or "").strip()
        if not style or style in seen:
            continue
        seen.add(style)
        unique.append(style)
    return unique


def _sample_styles(rng: random.Random, styles: Sequence[str], n: int) -> list[str]:
    if n <= 0 or not styles:
        return []
    if len(styles) >= n:
        return rng.sample(list(styles), n)
    picks = rng.sample(list(styles), len(styles))
    while len(picks) < n and styles:
        picks.append(rng.choice(list(styles)))
    return picks


def _select_with_exploration(
    rng: random.Random,
    styles: Sequence[str],
    scores: dict[str, float],
    n: int,
    rate: float,
) -> list[str]:
    if not styles or n <= 0:
        return []
    base = list(styles)
    allow_repeats = len(base) < n
    available = list(base)
    picks: list[str] = []
    for _ in range(n):
        if not available:
            if not allow_repeats:
                break
            available = list(base)
        use_random = rng.random() < rate
        if use_random:
            choice = rng.choice(available)
        else:
            choice = max(available, key=lambda item: scores.get(item, 0.0))
        picks.append(choice)
        if choice in available:
            available.remove(choice)
    return picks


def _weighted_choice(
    rng: random.Random, styles: Sequence[str], weights: Sequence[float]
) -> str:
    if not styles:
        raise ValueError("styles must not be empty")
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(styles))
    target = rng.random() * total
    cumulative = 0.0
    for style, weight in zip(styles, weights):
        if weight <= 0:
            continue
        cumulative += weight
        if target <= cumulative:
            return style
    return styles[-1]


def _select_style_pair(
    rng: random.Random,
    styles: Sequence[str],
    weights: Sequence[float],
    *,
    allow_duplicates: bool,
) -> tuple[str, str]:
    if not styles:
        return tuple()
    if len(styles) == 1:
        only = styles[0]
        return (only, only)

    first = _weighted_choice(rng, styles, weights)
    if allow_duplicates:
        second = _weighted_choice(rng, styles, weights)
    else:
        remaining_styles = [style for style in styles if style != first]
        remaining_weights = [
            weight
            for style, weight in zip(styles, weights)
            if style != first
        ]
        if not remaining_styles:
            second = first
        else:
            second = _weighted_choice(rng, remaining_styles, remaining_weights)

    pair = [first, second]
    rng.shuffle(pair)
    return (pair[0], pair[1])


__all__ = [
    "StyleRecommender",
    "VOTE_OK",
    "VOTE_DUPLICATE",
    "VOTE_INVALID",
]
