CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_path TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending',
    owner TEXT,
    lease_expires REAL NOT NULL DEFAULT 0,
    verdict TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, id);
