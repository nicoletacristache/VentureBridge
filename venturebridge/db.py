"""SQLite persistence and append-only audit helpers for VentureBridge."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app, g


SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    alias TEXT NOT NULL,
    access_code_hash TEXT NOT NULL,
    consent_accepted INTEGER NOT NULL CHECK (consent_accepted IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    source_json TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    llm_call_id TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE(user_id, version)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    recommendations_json TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    llm_call_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recommendation_id TEXT REFERENCES recommendations(id) ON DELETE SET NULL,
    direction_json TEXT NOT NULL,
    main_state_json TEXT NOT NULL,
    turn_index INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'complete')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    purpose TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    api_version TEXT,
    prompt_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_json TEXT,
    status TEXT NOT NULL CHECK (status IN ('success', 'failed')),
    error_code TEXT,
    latency_ms INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_deliveries (
    id TEXT PRIMARY KEY,
    simulation_id TEXT NOT NULL REFERENCES simulations(id) ON DELETE CASCADE,
    turn_no INTEGER NOT NULL,
    event_kind TEXT NOT NULL CHECK (event_kind IN ('main', 'exact_anchor', 'surface_anchor')),
    blueprint_id TEXT NOT NULL,
    form_id TEXT NOT NULL,
    selection_probability REAL NOT NULL CHECK (selection_probability > 0 AND selection_probability <= 1),
    scheduler_json TEXT NOT NULL,
    visible_state_json TEXT NOT NULL,
    event_json TEXT NOT NULL,
    library_version TEXT NOT NULL,
    domain_snapshot_hash TEXT NOT NULL,
    generation_call_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'cancelled')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(simulation_id, turn_no)
);

CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    delivery_id TEXT NOT NULL UNIQUE REFERENCES event_deliveries(id) ON DELETE CASCADE,
    decision_text TEXT NOT NULL,
    rationale_text TEXT NOT NULL,
    information_requests TEXT NOT NULL,
    revision_text TEXT NOT NULL,
    action_code TEXT NOT NULL,
    classification_confidence REAL NOT NULL CHECK (classification_confidence >= 0 AND classification_confidence <= 1),
    evidence_json TEXT NOT NULL,
    feedback_json TEXT NOT NULL,
    main_state_before_json TEXT NOT NULL,
    main_state_after_json TEXT NOT NULL,
    sandbox_state_after_json TEXT,
    scoring_call_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rating_observations (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    rater_key TEXT NOT NULL,
    rater_type TEXT NOT NULL CHECK (rater_type IN ('llm', 'human', 'rule', 'other')),
    version_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    api_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_template_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    build_id TEXT NOT NULL,
    prompt_hash TEXT,
    rating_time_key TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    source_call_id TEXT REFERENCES llm_calls(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    UNIQUE(response_id, rater_key, version_key, rating_time_key)
);

CREATE TABLE IF NOT EXISTS operation_locks (
    lock_key TEXT PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    subject_ref TEXT,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_profiles_user_active ON profiles(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_simulations_user ON simulations(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_deliveries_simulation ON event_deliveries(simulation_id, turn_no);
CREATE INDEX IF NOT EXISTS idx_llm_calls_user ON llm_calls(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ratings_response ON rating_observations(response_id);
CREATE INDEX IF NOT EXISTS idx_ratings_rater ON rating_observations(rater_key, version_key);
CREATE INDEX IF NOT EXISTS idx_operation_locks_expiry ON operation_locks(expires_at);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_RATING_VERSION_DEFAULTS = {
    "provider": "unknown-provider",
    "model": "unknown-model",
    "api_version": "unspecified",
    "prompt_version": "unknown-prompt-version",
    "prompt_template_hash": "legacy-unavailable",
    "schema_hash": "legacy-unavailable",
    "build_id": "unversioned",
}


def rating_version_facets(
    *,
    provider: str | None,
    model: str | None,
    api_version: str | None,
    prompt_version: str | None,
    prompt_template_hash: str | None,
    schema_hash: str | None,
    build_id: str | None,
) -> dict[str, str]:
    """Return the normalized, input-independent facets of a rating version.

    ``prompt_hash`` is intentionally absent: it fingerprints one learner-specific
    request and therefore cannot define the reusable rater/version facet.
    Legacy rows may lack newer audit fields, so normalization uses explicit
    sentinel values rather than borrowing the per-input prompt hash.
    """

    raw = {
        "provider": provider,
        "model": model,
        "api_version": api_version,
        "prompt_version": prompt_version,
        "prompt_template_hash": prompt_template_hash,
        "schema_hash": schema_hash,
        "build_id": build_id,
    }
    return {
        name: value.strip()
        if isinstance(value, str) and value.strip()
        else _RATING_VERSION_DEFAULTS[name]
        for name, value in raw.items()
    }


def rating_version_key(
    *,
    provider: str | None,
    model: str | None,
    api_version: str | None,
    prompt_version: str | None,
    prompt_template_hash: str | None,
    schema_hash: str | None,
    build_id: str | None,
) -> str:
    """Build a stable key from the seven version facets, never exact input."""

    facets = rating_version_facets(
        provider=provider,
        model=model,
        api_version=api_version,
        prompt_version=prompt_version,
        prompt_template_hash=prompt_template_hash,
        schema_hash=schema_hash,
        build_id=build_id,
    )
    digest = hashlib.sha256(canonical_json(facets).encode("utf-8")).hexdigest()
    return f"rating-version-v1:{digest}"


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        database = Path(current_app.config["DATABASE"])
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        g.db = connection
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    database = get_db()
    database.executescript(SCHEMA)
    _migrate_schema(database)
    _backfill_primary_ratings(database)
    database.commit()


def _column_names(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


def _rating_schema_is_current(db: sqlite3.Connection) -> bool:
    required = {
        "id",
        "response_id",
        "rater_key",
        "rater_type",
        "version_key",
        "provider",
        "model",
        "api_version",
        "prompt_version",
        "prompt_template_hash",
        "schema_hash",
        "build_id",
        "prompt_hash",
        "rating_time_key",
        "evidence_json",
        "source_call_id",
        "created_at",
    }
    if not required.issubset(_column_names(db, "rating_observations")):
        return False
    expected_unique = [
        "response_id",
        "rater_key",
        "version_key",
        "rating_time_key",
    ]
    for index in db.execute("PRAGMA index_list(rating_observations)").fetchall():
        if not int(index["unique"]):
            continue
        index_name = str(index["name"]).replace('"', '""')
        columns = [
            str(row["name"])
            for row in db.execute(f'PRAGMA index_info("{index_name}")').fetchall()
        ]
        if columns == expected_unique:
            return True
    return False


def _create_rating_observations_table(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS rating_observations (
            id TEXT PRIMARY KEY,
            response_id TEXT NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
            rater_key TEXT NOT NULL,
            rater_type TEXT NOT NULL
                CHECK (rater_type IN ('llm', 'human', 'rule', 'other')),
            version_key TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            api_version TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prompt_template_hash TEXT NOT NULL,
            schema_hash TEXT NOT NULL,
            build_id TEXT NOT NULL,
            prompt_hash TEXT,
            rating_time_key TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            source_call_id TEXT REFERENCES llm_calls(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL,
            UNIQUE(response_id, rater_key, version_key, rating_time_key)
        )
        """
    )


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _migrate_rating_observations(db: sqlite3.Connection) -> None:
    """Rebuild the rating table with reusable versions and rating occasions.

    The original uniqueness constraint cannot be dropped in place in SQLite.
    A savepoint keeps the table rename/copy/drop atomic, while deterministic
    legacy occasion keys make repeated ``init_db`` calls idempotent.
    """

    if _rating_schema_is_current(db):
        return

    legacy_table = "rating_observations_legacy_v1"
    db.execute("SAVEPOINT migrate_rating_observations_v2")
    try:
        db.execute(f"ALTER TABLE rating_observations RENAME TO {legacy_table}")
        _create_rating_observations_table(db)
        legacy_rows = db.execute(f"SELECT * FROM {legacy_table} ORDER BY id").fetchall()
        seen_occasions: set[tuple[str, str, str, str]] = set()
        for legacy_row in legacy_rows:
            row = dict(legacy_row)
            source_call = None
            if row.get("source_call_id"):
                source_call = db.execute(
                    "SELECT * FROM llm_calls WHERE id = ?",
                    (row["source_call_id"],),
                ).fetchone()
            call = dict(source_call) if source_call is not None else {}
            parameters = _json_object(call.get("parameters_json"))
            facets = rating_version_facets(
                provider=row.get("provider") or call.get("provider"),
                model=row.get("model") or call.get("model"),
                api_version=row.get("api_version") or call.get("api_version"),
                prompt_version=row.get("prompt_version") or call.get("prompt_version"),
                prompt_template_hash=(
                    row.get("prompt_template_hash")
                    or parameters.get("prompt_template_hash")
                ),
                schema_hash=row.get("schema_hash") or parameters.get("schema_hash"),
                build_id=row.get("build_id") or parameters.get("build_id"),
            )
            version_key = rating_version_key(**facets)
            rating_time_key = str(row.get("rating_time_key") or "").strip()
            if not rating_time_key:
                rating_time_key = f"legacy:{row['id']}"
            occasion = (
                str(row["response_id"]),
                str(row["rater_key"]),
                version_key,
                rating_time_key,
            )
            if occasion in seen_occasions:
                rating_time_key = f"{rating_time_key}:legacy:{row['id']}"
                occasion = (*occasion[:3], rating_time_key)
            seen_occasions.add(occasion)
            db.execute(
                """
                INSERT INTO rating_observations(
                    id, response_id, rater_key, rater_type, version_key,
                    provider, model, api_version, prompt_version,
                    prompt_template_hash, schema_hash, build_id, prompt_hash,
                    rating_time_key, evidence_json, source_call_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["response_id"],
                    row["rater_key"],
                    row["rater_type"],
                    version_key,
                    facets["provider"],
                    facets["model"],
                    facets["api_version"],
                    facets["prompt_version"],
                    facets["prompt_template_hash"],
                    facets["schema_hash"],
                    facets["build_id"],
                    row.get("prompt_hash") or call.get("prompt_hash"),
                    rating_time_key,
                    row["evidence_json"],
                    row.get("source_call_id"),
                    row["created_at"],
                ),
            )
        db.execute(f"DROP TABLE {legacy_table}")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ratings_response "
            "ON rating_observations(response_id)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ratings_rater "
            "ON rating_observations(rater_key, version_key)"
        )
        db.execute("RELEASE SAVEPOINT migrate_rating_observations_v2")
    except Exception:
        db.execute("ROLLBACK TO SAVEPOINT migrate_rating_observations_v2")
        db.execute("RELEASE SAVEPOINT migrate_rating_observations_v2")
        raise


def _migrate_schema(db: sqlite3.Connection) -> None:
    """Apply small idempotent forward migrations for existing SQLite files.

    SQLite cannot add a non-null column without a default to a populated table.
    Existing deliveries are therefore marked as legacy and intentionally lack
    a snapshot hash; repository code will allow them to be viewed but will not
    settle a pending legacy event against an unversioned rule library.
    """

    delivery_columns = _column_names(db, "event_deliveries")
    if "library_version" not in delivery_columns:
        db.execute(
            "ALTER TABLE event_deliveries ADD COLUMN library_version TEXT "
            "NOT NULL DEFAULT 'legacy-unversioned'"
        )
    if "domain_snapshot_hash" not in delivery_columns:
        db.execute("ALTER TABLE event_deliveries ADD COLUMN domain_snapshot_hash TEXT")
    _migrate_rating_observations(db)


def _backfill_primary_ratings(db: sqlite3.Connection) -> None:
    """Represent pre-migration primary LLM scores as rating-level rows."""

    rows = db.execute(
        """
        SELECT r.id AS response_id, r.evidence_json, r.scoring_call_id, r.created_at,
               c.provider, c.model, c.api_version, c.prompt_version, c.prompt_hash,
               c.parameters_json
        FROM responses r
        JOIN llm_calls c ON c.id = r.scoring_call_id
        LEFT JOIN rating_observations q
               ON q.response_id = r.id AND q.source_call_id = r.scoring_call_id
        WHERE q.id IS NULL AND c.status = 'success'
        """
    ).fetchall()
    for row in rows:
        parameters = _json_object(row["parameters_json"])
        facets = rating_version_facets(
            provider=row["provider"],
            model=row["model"],
            api_version=row["api_version"],
            prompt_version=row["prompt_version"],
            prompt_template_hash=parameters.get("prompt_template_hash"),
            schema_hash=parameters.get("schema_hash"),
            build_id=parameters.get("build_id"),
        )
        rater_key = f"llm:{facets['provider']}:{facets['model']}"
        version_key = rating_version_key(**facets)
        db.execute(
            """
            INSERT OR IGNORE INTO rating_observations(
                id, response_id, rater_key, rater_type, version_key,
                provider, model, api_version, prompt_version,
                prompt_template_hash, schema_hash, build_id, prompt_hash,
                rating_time_key, evidence_json, source_call_id, created_at
            ) VALUES (?, ?, ?, 'llm', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("rating"),
                row["response_id"],
                rater_key,
                version_key,
                facets["provider"],
                facets["model"],
                facets["api_version"],
                facets["prompt_version"],
                facets["prompt_template_hash"],
                facets["schema_hash"],
                facets["build_id"],
                row["prompt_hash"],
                f"call:{row['scoring_call_id']}",
                row["evidence_json"],
                row["scoring_call_id"],
                row["created_at"],
            ),
        )


@contextmanager
def transaction(connection: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    db = connection or get_db()
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
    except Exception:
        db.rollback()
        raise
    else:
        db.commit()


def fetch_one(query: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    return get_db().execute(query, parameters).fetchone()


def fetch_all(query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return get_db().execute(query, parameters).fetchall()


def append_audit(
    db: sqlite3.Connection,
    *,
    subject_ref: str | None,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Append a chained audit entry; payloads must already be data-minimized."""

    previous = db.execute(
        "SELECT entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    previous_hash = previous["entry_hash"] if previous else "0" * 64
    occurred_at = utc_now()
    payload_json = canonical_json(dict(payload or {}))
    material = canonical_json(
        {
            "occurred_at": occurred_at,
            "subject_ref": subject_ref,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "payload": json.loads(payload_json),
            "previous_hash": previous_hash,
        }
    )
    entry_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()
    db.execute(
        """
        INSERT INTO audit_log(
            occurred_at, subject_ref, action, entity_type, entity_id,
            payload_json, previous_hash, entry_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            occurred_at,
            subject_ref,
            action,
            entity_type,
            entity_id,
            payload_json,
            previous_hash,
            entry_hash,
        ),
    )
    return entry_hash


def verify_audit_chain(db: sqlite3.Connection | None = None) -> tuple[bool, int | None]:
    connection = db or get_db()
    previous_hash = "0" * 64
    rows = connection.execute("SELECT * FROM audit_log ORDER BY sequence").fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"])
        material = canonical_json(
            {
                "occurred_at": row["occurred_at"],
                "subject_ref": row["subject_ref"],
                "action": row["action"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "payload": payload,
                "previous_hash": previous_hash,
            }
        )
        expected = hashlib.sha256(material.encode("utf-8")).hexdigest()
        if row["previous_hash"] != previous_hash or row["entry_hash"] != expected:
            return False, int(row["sequence"])
        previous_hash = row["entry_hash"]
    return True, None


def init_app(app: Any) -> None:
    app.teardown_appcontext(close_db)

    @app.cli.command("init-db")
    def init_db_command() -> None:
        """Initialize the local SQLite database."""
        init_db()
        print("Database initialized.")

    @app.cli.command("verify-audit")
    def verify_audit_command() -> None:
        """Verify the append-only audit hash chain."""
        valid, failed_at = verify_audit_chain()
        if not valid:
            raise SystemExit(f"Audit chain verification failed at sequence {failed_at}.")
        print("Audit chain is valid.")
