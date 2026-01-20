import os
import sqlite3
import time
from typing import Optional

_conn: Optional[sqlite3.Connection] = None


def init_db(db_path: str = "data/inference_logs/inference_logs.db") -> None:
    """Initialize a sqlite3 database and create the inference_logs table."""
    global _conn
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    _conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = _conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latency_ms REAL,
            timestamp REAL,
            embedding REAL,
            prediction REAL
        )
        """
    )
    _conn.commit()


def insert_row(latency: float, embedding: float, prediction: float) -> int:
    """Insert a single inference log into the database.

    Returns the inserted row id.
    """
    global _conn
    if _conn is None:
        raise RuntimeError("Database not initialized. Call init_db first.")

    cur = _conn.cursor()
    ts = time.time()
    cur.execute(
        "INSERT INTO inference_logs (latency_ms, timestamp, embedding, prediction) VALUES (?, ?, ?, ?)",
        (latency, ts, embedding, prediction),
    )
    _conn.commit()
    return cur.lastrowid
