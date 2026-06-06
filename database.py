"""
database.py — Prediction Logging
Async SQLite store for every prediction, explanation, and drift check.
Enables audit trails, monitoring, and dashboard history.
"""

import json
import time
import logging
import aiosqlite
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "predictions.db"


# ── Schema ────────────────────────────────────────────────────────────────────
CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    endpoint      TEXT    NOT NULL,
    features      TEXT    NOT NULL,
    prediction    INTEGER NOT NULL,
    label         TEXT    NOT NULL,
    anomaly_score REAL    NOT NULL,
    latency_ms    REAL    NOT NULL,
    top_driver    TEXT,
    shap_values   TEXT,
    source        TEXT    DEFAULT 'api'
)
"""

CREATE_DRIFT_CHECKS = """
CREATE TABLE IF NOT EXISTS drift_checks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    features      TEXT    NOT NULL,
    ks_scores     TEXT    NOT NULL,
    drift_detected INTEGER NOT NULL,
    drifted_sensors TEXT  NOT NULL,
    latency_ms    REAL    NOT NULL
)
"""

CREATE_SIMULATOR_SESSIONS = """
CREATE TABLE IF NOT EXISTS simulator_sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL    NOT NULL,
    ended_at      REAL,
    dataset       TEXT    NOT NULL,
    engine_id     INTEGER NOT NULL,
    fault_mode    TEXT    DEFAULT 'none',
    total_cycles  INTEGER DEFAULT 0,
    anomalies_detected INTEGER DEFAULT 0
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source)",
]


# ── Init ──────────────────────────────────────────────────────────────────────
async def init_db():
    """Create tables and indexes on startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_PREDICTIONS)
        await db.execute(CREATE_DRIFT_CHECKS)
        await db.execute(CREATE_SIMULATOR_SESSIONS)
        for idx in INDEXES:
            await db.execute(idx)
        await db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


# ── Prediction logging ────────────────────────────────────────────────────────
async def log_prediction(
    endpoint: str,
    features: list,
    prediction: int,
    label: str,
    anomaly_score: float,
    latency_ms: float,
    top_driver: Optional[str] = None,
    shap_values: Optional[dict] = None,
    source: str = "api",
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO predictions
                (timestamp, endpoint, features, prediction, label,
                 anomaly_score, latency_ms, top_driver, shap_values, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                endpoint,
                json.dumps(features),
                prediction,
                label,
                anomaly_score,
                latency_ms,
                top_driver,
                json.dumps(shap_values) if shap_values else None,
                source,
            ),
        )
        await db.commit()


# ── Drift logging ─────────────────────────────────────────────────────────────
async def log_drift_check(
    features: list,
    ks_scores: dict,
    drift_detected: bool,
    drifted_sensors: list,
    latency_ms: float,
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO drift_checks
                (timestamp, features, ks_scores, drift_detected, drifted_sensors, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                json.dumps(features),
                json.dumps(ks_scores),
                int(drift_detected),
                json.dumps(drifted_sensors),
                latency_ms,
            ),
        )
        await db.commit()


# ── Simulator session logging ─────────────────────────────────────────────────
async def start_simulator_session(dataset: str, engine_id: int, fault_mode: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO simulator_sessions (started_at, dataset, engine_id, fault_mode)
            VALUES (?, ?, ?, ?)
            """,
            (time.time(), dataset, engine_id, fault_mode),
        )
        await db.commit()
        return cursor.lastrowid


async def end_simulator_session(session_id: int, total_cycles: int, anomalies: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE simulator_sessions
            SET ended_at=?, total_cycles=?, anomalies_detected=?
            WHERE id=?
            """,
            (time.time(), total_cycles, anomalies, session_id),
        )
        await db.commit()


# ── Query helpers (used by dashboard endpoints) ───────────────────────────────
async def get_recent_predictions(limit: int = 50, label_filter: Optional[str] = None):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if label_filter:
            cursor = await db.execute(
                "SELECT * FROM predictions WHERE label=? ORDER BY timestamp DESC LIMIT ?",
                (label_filter, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_prediction_stats():
    """Aggregate stats for the dashboard health panel."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        total = (await (await db.execute("SELECT COUNT(*) as n FROM predictions")).fetchone())["n"]
        anomalies = (await (await db.execute(
            "SELECT COUNT(*) as n FROM predictions WHERE label='anomaly'"
        )).fetchone())["n"]
        avg_latency = (await (await db.execute(
            "SELECT AVG(latency_ms) as avg FROM predictions"
        )).fetchone())["avg"]
        avg_score = (await (await db.execute(
            "SELECT AVG(anomaly_score) as avg FROM predictions"
        )).fetchone())["avg"]

        # Last 100 predictions anomaly rate
        recent_anomalies = (await (await db.execute(
            """SELECT COUNT(*) as n FROM predictions
               WHERE label='anomaly'
               AND id > (SELECT MAX(id) - 100 FROM predictions)"""
        )).fetchone())["n"]

        return {
            "total_predictions": total,
            "total_anomalies": anomalies,
            "anomaly_rate": round(anomalies / total * 100, 2) if total > 0 else 0,
            "recent_anomaly_rate": round(recent_anomalies, 2),
            "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0,
            "avg_anomaly_score": round(avg_score, 6) if avg_score else 0,
        }


async def get_anomaly_timeline(hours: int = 24):
    """Hourly anomaly counts for the timeline chart."""
    cutoff = time.time() - (hours * 3600)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT
                CAST((timestamp - ?) / 3600 AS INTEGER) as hour_bucket,
                COUNT(*) as total,
                SUM(CASE WHEN label='anomaly' THEN 1 ELSE 0 END) as anomalies
            FROM predictions
            WHERE timestamp > ?
            GROUP BY hour_bucket
            ORDER BY hour_bucket
            """,
            (cutoff, cutoff),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_top_drivers(limit: int = 10):
    """Most frequently occurring top_driver sensors across all anomalies."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT top_driver, COUNT(*) as count
            FROM predictions
            WHERE label='anomaly' AND top_driver IS NOT NULL
            GROUP BY top_driver
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]