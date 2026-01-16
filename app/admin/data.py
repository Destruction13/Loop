"""Data access for the admin Mini App."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdminUserRow:
    user_id: int
    username: str | None
    first_name: str | None
    last_name: str | None
    full_name: str | None
    generations: int
    tries_used: int
    tries_limit: int
    tries_remaining: int
    site_clicks: int
    social_clicks: int
    phone: str | None
    # Event attempts: paid only (free not counted per user request)
    event_paid_used: int
    event_paid_limit: int = 10
    
    @property
    def event_paid_remaining(self) -> int:
        return max(0, self.event_paid_limit - self.event_paid_used)

    @property
    def display_name(self) -> str:
        """Name for display in the list - prefer real name over username."""
        if self.full_name:
            return self.full_name
        if self.first_name or self.last_name:
            parts = [p for p in (self.first_name, self.last_name) if p]
            return " ".join(parts)
        if self.username:
            return f"@{self.username}"
        return f"User {self.user_id}"

    @property
    def telegram_link(self) -> str:
        if self.username:
            return f"https://t.me/{self.username}"
        return f"tg://user?id={self.user_id}"


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    db_uri = f"{db_path.resolve().as_posix()}"
    conn = sqlite3.connect(f"file:{db_uri}?mode=ro", uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 3000")
    return conn


def _open_readwrite(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path.resolve()), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    )
    return cur.fetchone() is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({name})")
    return {row[1] for row in cur.fetchall()}


def list_admin_users(
    db_path: Path,
    *,
    offset: int,
    limit: int,
    sort: str,
    order: str,
) -> tuple[list[AdminUserRow], int]:
    offset = max(int(offset or 0), 0)
    limit = max(min(int(limit or 0) or 50, 200), 1)
    sort_key = (sort or "generations").lower()
    order_key = (order or "desc").lower()
    order_sql = "ASC" if order_key == "asc" else "DESC"

    sort_map = {
        "generations": "generations",
        "gen_count": "generations",
        "site_clicks": "site_clicks",
        "social_clicks": "social_clicks",
        "username": "username",
        "user_id": "user_id",
        "tries_remaining": "tries_remaining",
    }
    with _open_readonly(db_path) as conn:
        has_contacts = _table_exists(conn, "user_contacts")
        has_analytics = _table_exists(conn, "analytics_events")
        has_event_attempts = _table_exists(conn, "event_user_attempts")
        user_columns = _table_columns(conn, "users")
        has_username = "username" in user_columns
        has_full_name = "full_name" in user_columns
        has_first_name = "first_name" in user_columns
        has_last_name = "last_name" in user_columns
        has_gen_count = "gen_count" in user_columns
        has_tries_used = "tries_used" in user_columns
        has_daily_try_limit = "daily_try_limit" in user_columns

        sort_column = sort_map.get(sort_key, "generations")
        if sort_column == "username" and not has_username:
            sort_column = "user_id"

        total = conn.execute("SELECT COUNT(1) FROM users").fetchone()[0]

        joins = []
        select_fields = [
            "u.user_id AS user_id",
            ("u.username AS username" if has_username else "NULL AS username"),
            ("u.full_name AS full_name" if has_full_name else "NULL AS full_name"),
            ("u.first_name AS first_name" if has_first_name else "NULL AS first_name"),
            ("u.last_name AS last_name" if has_last_name else "NULL AS last_name"),
            ("u.gen_count AS generations" if has_gen_count else "0 AS generations"),
            ("u.tries_used AS tries_used" if has_tries_used else "0 AS tries_used"),
            ("u.daily_try_limit AS tries_limit" if has_daily_try_limit else "3 AS tries_limit"),
        ]
        
        # Calculate remaining tries
        if has_tries_used and has_daily_try_limit:
            select_fields.append(
                "MAX(0, COALESCE(u.daily_try_limit, 3) - COALESCE(u.tries_used, 0)) AS tries_remaining"
            )
        else:
            select_fields.append("0 AS tries_remaining")

        if has_contacts:
            joins.append("LEFT JOIN user_contacts uc ON uc.tg_user_id = u.user_id")
            select_fields.append("uc.phone_e164 AS phone")
        else:
            select_fields.append("NULL AS phone")

        if has_analytics:
            joins.append(
                """
                LEFT JOIN (
                    SELECT user_id, COUNT(1) AS cnt
                    FROM analytics_events
                    WHERE event = 'cta_book_opened'
                    GROUP BY user_id
                ) site_counts ON site_counts.user_id = CAST(u.user_id AS TEXT)
                """
            )
            joins.append(
                """
                LEFT JOIN (
                    SELECT user_id, COUNT(1) AS cnt
                    FROM analytics_events
                    WHERE event IN ('social_ad_shown', 'social_link_opened')
                    GROUP BY user_id
                ) social_counts ON social_counts.user_id = CAST(u.user_id AS TEXT)
                """
            )
            select_fields.append("COALESCE(site_counts.cnt, 0) AS site_clicks")
            select_fields.append("COALESCE(social_counts.cnt, 0) AS social_clicks")
        else:
            select_fields.append("0 AS site_clicks")
            select_fields.append("0 AS social_clicks")

        # Event attempts aggregation (only paid, free not counted per user request)
        if has_event_attempts:
            joins.append(
                """
                LEFT JOIN (
                    SELECT user_id, 
                           SUM(paid_used) AS total_paid_used
                    FROM event_user_attempts
                    GROUP BY user_id
                ) event_stats ON event_stats.user_id = u.user_id
                """
            )
            select_fields.append("COALESCE(event_stats.total_paid_used, 0) AS event_paid_used")
        else:
            select_fields.append("0 AS event_paid_used")

        query = f"""
            SELECT {", ".join(select_fields)}
            FROM users u
            {" ".join(joins)}
            ORDER BY {sort_column} {order_sql}, u.user_id DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, (limit, offset)).fetchall()

    items: list[AdminUserRow] = []
    for row in rows:
        first_name = (row["first_name"] or "").strip() or None
        last_name = (row["last_name"] or "").strip() or None
        full_name = (row["full_name"] or "").strip() or None
        # Build full_name from parts if not set
        if not full_name and (first_name or last_name):
            parts = [p for p in (first_name, last_name) if p]
            full_name = " ".join(parts) if parts else None
        items.append(
            AdminUserRow(
                user_id=int(row["user_id"]),
                username=(row["username"] or "").strip() or None,
                first_name=first_name,
                last_name=last_name,
                full_name=full_name,
                generations=int(row["generations"] or 0),
                tries_used=int(row["tries_used"] or 0),
                tries_limit=int(row["tries_limit"] or 3),
                tries_remaining=int(row["tries_remaining"] or 0),
                site_clicks=int(row["site_clicks"] or 0),
                social_clicks=int(row["social_clicks"] or 0),
                phone=(row["phone"] or "").strip() or None,
                event_paid_used=int(row["event_paid_used"] or 0),
            )
        )
    return items, int(total or 0)


def get_user_details(db_path: Path, user_id: int) -> dict[str, Any] | None:
    """Get detailed info about a single user."""
    with _open_readonly(db_path) as conn:
        user_columns = _table_columns(conn, "users")
        cur = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        
        result: dict[str, Any] = dict(row)
        
        # Get event attempts if table exists
        if _table_exists(conn, "event_user_attempts"):
            events_cur = conn.execute(
                "SELECT event_id, free_unlocked, free_used, paid_used FROM event_user_attempts WHERE user_id = ?",
                (user_id,),
            )
            result["events"] = [dict(r) for r in events_cur.fetchall()]
        else:
            result["events"] = []
        
        return result


def delete_user(db_path: Path, user_id: int) -> bool:
    """Delete a user completely from all tables."""
    with _open_readwrite(db_path) as conn:
        # Check if user exists
        cur = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        if not cur.fetchone():
            return False
        
        # Delete from all related tables
        tables_to_clean = [
            ("users", "user_id"),
            ("user_contacts", "tg_user_id"),
            ("user_seen_models", "user_id"),
            ("user_style_pref", "user_id"),
            ("user_style_votes", "user_id"),
            ("event_user_attempts", "user_id"),
            ("event_trigger_shown", "user_id"),
            ("contact_shares", "user_id"),
        ]
        
        for table_name, column_name in tables_to_clean:
            if _table_exists(conn, table_name):
                conn.execute(f"DELETE FROM {table_name} WHERE {column_name} = ?", (user_id,))
        
        # Also clean analytics_events if exists (user_id is TEXT there)
        if _table_exists(conn, "analytics_events"):
            conn.execute(
                "DELETE FROM analytics_events WHERE user_id = ?",
                (str(user_id),),
            )
        
        conn.commit()
        return True


def update_user_tries(
    db_path: Path,
    user_id: int,
    *,
    tries_remaining: int | None = None,
    tries_limit: int | None = None,
) -> bool:
    """Update user's tries. 
    
    Setting tries_remaining will calculate tries_used = tries_limit - tries_remaining.
    This allows the user to continue generating even if they had 0 remaining.
    """
    with _open_readwrite(db_path) as conn:
        user_columns = _table_columns(conn, "users")
        if "tries_used" not in user_columns or "daily_try_limit" not in user_columns:
            return False
        
        # Get current values
        cur = conn.execute(
            "SELECT tries_used, daily_try_limit, locked_until FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return False
        
        current_limit = row["daily_try_limit"] or 3
        new_limit = tries_limit if tries_limit is not None else current_limit
        
        if tries_remaining is not None:
            # Calculate new tries_used based on desired remaining
            new_tries_used = max(0, new_limit - tries_remaining)
        else:
            # Keep current tries_used ratio
            current_used = row["tries_used"] or 0
            new_tries_used = current_used
        
        # If giving more tries, clear locked_until to allow generation
        updates = ["tries_used = ?", "daily_try_limit = ?", "daily_used = ?"]
        params: list[Any] = [new_tries_used, new_limit, new_tries_used]
        
        if tries_remaining is not None and tries_remaining > 0:
            updates.append("locked_until = NULL")
        
        conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?",
            params + [user_id],
        )
        conn.commit()
        return True


def update_event_tries(
    db_path: Path,
    user_id: int,
    event_id: str,
    *,
    free_used: int | None = None,
    paid_used: int | None = None,
) -> bool:
    """Update user's event attempts."""
    with _open_readwrite(db_path) as conn:
        if not _table_exists(conn, "event_user_attempts"):
            return False
        
        cur = conn.execute(
            "SELECT free_used, paid_used FROM event_user_attempts WHERE user_id = ? AND event_id = ?",
            (user_id, event_id),
        )
        row = cur.fetchone()
        if not row:
            return False
        
        new_free = free_used if free_used is not None else row["free_used"]
        new_paid = paid_used if paid_used is not None else row["paid_used"]
        
        conn.execute(
            "UPDATE event_user_attempts SET free_used = ?, paid_used = ? WHERE user_id = ? AND event_id = ?",
            (new_free, new_paid, user_id, event_id),
        )
        conn.commit()
        return True


def get_stats(db_path: Path) -> dict[str, Any]:
    """Get overall statistics."""
    with _open_readonly(db_path) as conn:
        user_columns = _table_columns(conn, "users")
        
        stats: dict[str, Any] = {}
        
        # Total users
        stats["total_users"] = conn.execute("SELECT COUNT(1) FROM users").fetchone()[0]
        
        # Total generations
        if "gen_count" in user_columns:
            result = conn.execute("SELECT SUM(gen_count) FROM users").fetchone()[0]
            stats["total_generations"] = result or 0
        else:
            stats["total_generations"] = 0
        
        # Users with phone
        if _table_exists(conn, "user_contacts"):
            stats["users_with_phone"] = conn.execute(
                "SELECT COUNT(DISTINCT tg_user_id) FROM user_contacts"
            ).fetchone()[0]
        else:
            stats["users_with_phone"] = 0
        
        # Event stats (only paid, free not counted per user request)
        if _table_exists(conn, "event_user_attempts"):
            event_stats = conn.execute(
                "SELECT SUM(paid_used) as paid, COUNT(DISTINCT user_id) as users FROM event_user_attempts"
            ).fetchone()
            stats["event_paid_used"] = event_stats["paid"] or 0
            stats["event_users"] = event_stats["users"] or 0
        else:
            stats["event_paid_used"] = 0
            stats["event_users"] = 0
        
        return stats
