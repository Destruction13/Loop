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


def test_catalog_version_sync_clears_seen(tmp_path: Path) -> None:
    db_path = tmp_path / "sync.db"
    repo = Repository(db_path, daily_limit=5)

    async def scenario() -> None:
        await repo.init()
        await repo.sync_catalog_version("v1", clear_on_change=True)
        await repo.record_seen_models(1, ["a", "b"])
        seen = await repo.list_seen_models(1, since=None)
        assert seen == {"a", "b"}

        changed, cleared = await repo.sync_catalog_version("v1", clear_on_change=True)
        assert changed is False
        assert cleared is False

        changed, cleared = await repo.sync_catalog_version("v2", clear_on_change=True)
        assert changed is True
        assert cleared is True
        assert await repo.list_seen_models(1, since=None) == set()

    asyncio.run(scenario())
