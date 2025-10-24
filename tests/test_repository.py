"""Repository tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.services.repository import Repository


def test_daily_reset(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    repo = Repository(db_path, daily_limit=2)

    async def scenario() -> None:
        await repo.init()
        await repo.ensure_user(1)
        await repo.inc_used_on_success(1)
        profile = await repo.get_user(1)
        assert profile is not None
        assert profile.daily_used == 1

        future = (profile.last_reset_at or datetime.now(UTC)) + timedelta(hours=25)
        updated = await repo.ensure_daily_reset(1, now=future)
        assert updated.daily_used == 0
        assert updated.seen_models == []
        remaining = await repo.remaining_tries(1)
        assert remaining == 2

    asyncio.run(scenario())
