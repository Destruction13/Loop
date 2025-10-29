"""Concurrency primitives used across the application."""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")

GEN_SEMAPHORE = asyncio.Semaphore(5)


async def with_generation_slot(coro: Awaitable[T]) -> T:
    """Run the coroutine under the global generation semaphore."""

    async with GEN_SEMAPHORE:
        return await coro


__all__ = ["GEN_SEMAPHORE", "with_generation_slot"]

