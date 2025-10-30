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
        assert profile.tries_used == 1
        assert profile.locked_until is None

        await repo.inc_used_on_success(1)
        profile = await repo.get_user(1)
        assert profile is not None
        assert profile.tries_used == 2
        assert profile.locked_until is not None
        assert await repo.remaining_tries(1) == 0

        # attempt during lock should not change the counters
        await repo.inc_used_on_success(1)
        locked_until = profile.locked_until
        assert locked_until is not None
        future = locked_until + timedelta(minutes=1)
        updated = await repo.ensure_daily_reset(1, now=future)
        assert updated.tries_used == 0
        assert updated.cycle_index == 1
        assert updated.locked_until is None
        assert await repo.remaining_tries(1) == 2

    asyncio.run(scenario())


def test_catalog_version_sync_clears_seen(tmp_path: Path) -> None:
    db_path = tmp_path / "sync.db"
    repo = Repository(db_path, daily_limit=5)

    async def scenario() -> None:
        await repo.init()
        await repo.sync_catalog_version("v1", clear_on_change=True)
        await repo.record_seen_models(1, ["a", "b"])
        seen = await repo.list_seen_models(1, context="global")
        assert seen == {"a", "b"}

        changed, cleared = await repo.sync_catalog_version("v1", clear_on_change=True)
        assert changed is False
        assert cleared is False

        changed, cleared = await repo.sync_catalog_version("v2", clear_on_change=True)
        assert changed is True
        assert cleared is True
        assert await repo.list_seen_models(1, context="global") == set()

    asyncio.run(scenario())
