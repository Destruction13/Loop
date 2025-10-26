PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    tg_id INTEGER PRIMARY KEY,
    username TEXT,
    name TEXT,
    gender TEXT,
    phone TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    tg_id INTEGER PRIMARY KEY,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_activity_ts TEXT NOT NULL,
    ecom_prompt_sent INTEGER NOT NULL DEFAULT 0,
    social_ad_sent INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id INTEGER,
    type TEXT NOT NULL,
    payload_json TEXT,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (tg_id) REFERENCES users (tg_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_tg_ts ON events (tg_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events (type, ts);
