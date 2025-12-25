"""Read-only data access for the admin Mini App."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdminUserRow:
    user_id: int
    username: str | None
    full_name: str | None
    generations: int
    site_clicks: int
    social_clicks: int
    phone: str | None

    @property
    def telegram_link(self) -> str:
        if self.username:
            return f"https://t.me/{self.username}"
        return f"tg://user?id={self.user_id}"


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 3000")
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
    }
    with _open_readonly(db_path) as conn:
        has_contacts = _table_exists(conn, "user_contacts")
        has_analytics = _table_exists(conn, "analytics_events")
        user_columns = _table_columns(conn, "users")
        has_username = "username" in user_columns
        has_full_name = "full_name" in user_columns
        has_first_name = "first_name" in user_columns
        has_last_name = "last_name" in user_columns
        has_gen_count = "gen_count" in user_columns

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
        ]
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
        full_name = (row["full_name"] or "").strip() or None
        if not full_name:
            first_name = (row["first_name"] or "").strip()
            last_name = (row["last_name"] or "").strip()
            combined = " ".join(part for part in (first_name, last_name) if part).strip()
            full_name = combined or None
        items.append(
            AdminUserRow(
                user_id=int(row["user_id"]),
                username=(row["username"] or "").strip() or None,
                full_name=full_name,
                generations=int(row["generations"] or 0),
                site_clicks=int(row["site_clicks"] or 0),
                social_clicks=int(row["social_clicks"] or 0),
                phone=(row["phone"] or "").strip() or None,
            )
        )
    return items, int(total or 0)
