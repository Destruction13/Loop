"""Tests for daily try-on limits."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.services.repository import Repository


def test_daily_limit_decrements_and_resets(tmp_path: Path) -> None:
    async def _run() -> None:
        db_path = tmp_path / "test.db"
        repo = Repository(db_path, daily_limit=2)
        await repo.init()
        await repo.ensure_user(1)

        remaining_initial = await repo.remaining_tries(1)
        assert remaining_initial == 2

        await repo.inc_used_on_success(1)
        remaining_after = await repo.remaining_tries(1)
        assert remaining_after == 1

        profile = await repo.get_user(1)
        assert profile is not None
        future = (profile.last_reset_at or datetime.now(UTC)) + timedelta(hours=25)
        updated = await repo.ensure_daily_reset(1, now=future)
        assert updated.daily_used == 0
        assert updated.seen_models == []
        assert await repo.remaining_tries(1) == 2

    asyncio.run(_run())
