import sqlite3
from contextlib import closing
from datetime import datetime

from settings import DB_PATH, CAMERA_NAME


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                camera TEXT,
                people_count INTEGER,
                seconds_in_zone INTEGER,
                image_path TEXT,
                alert_type TEXT DEFAULT 'zone',
                details TEXT DEFAULT '',
                clip_path TEXT DEFAULT ''
            )
        """)

        columns = [
            row[1]
            for row in db.execute("PRAGMA table_info(incidents)")
        ]

        migrations = [
            ("seconds_in_zone", "INTEGER DEFAULT 0"),
            ("alert_type", "TEXT DEFAULT 'zone'"),
            ("details", "TEXT DEFAULT ''"),
            ("clip_path", "TEXT DEFAULT ''")
        ]

        for name, definition in migrations:
            if name not in columns:
                db.execute(
                    f"ALTER TABLE incidents ADD COLUMN {name} {definition}"
                )

        db.commit()


def add_incident(
    timestamp,
    people,
    seconds,
    path,
    alert_type="zone",
    details="",
    clip_path=""
):
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            """
            INSERT INTO incidents (
                timestamp,
                camera,
                people_count,
                seconds_in_zone,
                image_path,
                alert_type,
                details,
                clip_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                CAMERA_NAME,
                people,
                seconds,
                path,
                alert_type,
                details,
                clip_path
            )
        )

        db.commit()


def alerts_today(alert_type=None):
    today = datetime.now().strftime("%Y-%m-%d")

    query = "SELECT COUNT(*) FROM incidents WHERE timestamp LIKE ?"
    params = [today + "%"]

    if alert_type:
        query += " AND alert_type = ?"
        params.append(alert_type)

    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(query, params).fetchone()[0]


def recent_incidents(limit=6):
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(
            """
            SELECT
                timestamp,
                seconds_in_zone,
                image_path,
                alert_type,
                details,
                clip_path
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()