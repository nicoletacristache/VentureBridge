"""Small persistence layer used by the web workflow.

The repository stores only already-sanitized learner text.  API credentials and
full prompts are never accepted by these functions, which reduces the chance
that a later logging change exposes them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from .db import (
    append_audit,
    canonical_json,
    fetch_all,
    fetch_one,
    new_id,
    rating_version_facets,
    rating_version_key,
    transaction,
    utc_now,
)
from .domain import (
    BlueprintLibrary,
    EntrepreneurshipState,
    ScheduledEvent,
    StateTransitionError,
    apply_classified_action,
)
from .schemas import SchemaValidationError, validate_structured_output


class DeliveryIntegrityError(ValueError):
    """A persisted delivery no longer matches its canonical domain snapshot."""


def _require_successful_llm_call(
    db: sqlite3.Connection,
    *,
    call_id: str,
    user_id: str,
    purpose: str,
) -> sqlite3.Row:
    row = db.execute(
        """
        SELECT * FROM llm_calls
        WHERE id = ? AND user_id = ? AND purpose = ? AND status = 'success'
        """,
        (call_id, user_id, purpose),
    ).fetchone()
    if row is None:
        raise DeliveryIntegrityError(
            f"{purpose} output must reference a successful learner-owned LLM call"
        )
    return row


def _llm_call_output(call: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    """Return one successful call's structured output or fail closed."""

    try:
        raw_output = call["output_json"]
        parsed = json.loads(str(raw_output))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DeliveryIntegrityError(
            f"{purpose} LLM call is missing a valid structured output"
        ) from exc
    if not isinstance(parsed, dict):
        raise DeliveryIntegrityError(
            f"{purpose} LLM call output must be a JSON object"
        )
    return parsed


def _require_exact_llm_output(
    call: Mapping[str, Any], *, purpose: str, expected: Mapping[str, Any]
) -> None:
    actual = _llm_call_output(call, purpose=purpose)
    if not hmac.compare_digest(
        canonical_json(actual).encode("utf-8"),
        canonical_json(dict(expected)).encode("utf-8"),
    ):
        raise DeliveryIntegrityError(
            f"persisted {purpose} output differs from its successful LLM call"
        )


def _rating_facets_from_call(call: Mapping[str, Any]) -> dict[str, str]:
    """Read reusable version facets from one persisted LLM call.

    The exact ``prompt_hash`` remains on the call/rating for audit only.  It is
    deliberately excluded from the version key because it changes with each
    learner input.
    """

    try:
        parameters = json.loads(str(call["parameters_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DeliveryIntegrityError(
            "LLM call is missing valid version-facet parameters."
        ) from exc
    if not isinstance(parameters, dict):
        raise DeliveryIntegrityError("LLM call version-facet parameters must be an object.")
    raw = {
        "provider": call.get("provider"),
        "model": call.get("model"),
        "api_version": call.get("api_version"),
        "prompt_version": call.get("prompt_version"),
        "prompt_template_hash": parameters.get("prompt_template_hash"),
        "schema_hash": parameters.get("schema_hash"),
        "build_id": parameters.get("build_id"),
    }
    required = (
        "provider",
        "model",
        "prompt_version",
        "prompt_template_hash",
        "schema_hash",
        "build_id",
    )
    missing = [
        name
        for name in required
        if not isinstance(raw[name], str) or not str(raw[name]).strip()
    ]
    if missing:
        raise DeliveryIntegrityError(
            "LLM call lacks rating version facets: " + ", ".join(missing)
        )
    return rating_version_facets(**raw)


def _manual_rating_facets(
    *,
    provider: str | None,
    model: str | None,
    api_version: str | None,
    prompt_version: str | None,
    prompt_template_hash: str | None,
    schema_hash: str | None,
    build_id: str | None,
) -> dict[str, str]:
    raw = {
        "provider": provider,
        "model": model,
        "api_version": api_version,
        "prompt_version": prompt_version,
        "prompt_template_hash": prompt_template_hash,
        "schema_hash": schema_hash,
        "build_id": build_id,
    }
    required = (
        "provider",
        "model",
        "prompt_version",
        "prompt_template_hash",
        "schema_hash",
        "build_id",
    )
    missing = [
        name
        for name in required
        if not isinstance(raw[name], str) or not str(raw[name]).strip()
    ]
    if missing:
        raise ValueError("Missing rating version facets: " + ", ".join(missing))
    return rating_version_facets(**raw)


def _loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def row_to_dict(row: sqlite3.Row | None, json_fields: tuple[str, ...] = ()) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for field in json_fields:
        if field in result:
            result[field] = _loads(result[field])
    return result


def subject_ref(user_id: str) -> str:
    return hashlib.sha256(f"venturebridge:{user_id}".encode("utf-8")).hexdigest()[:20]


def trajectory_ref(user_id: str, simulation_id: str) -> str:
    """Pseudonymous trajectory key; separate simulations are never merged."""

    return subject_ref(f"{user_id}:{simulation_id}")


def acquire_operation_lock(lock_key: str, *, ttl_seconds: int = 120) -> str | None:
    """Atomically acquire a short-lived cross-request operation lock.

    ``None`` means another unexpired holder owns the key.  The opaque token is
    required for release, so an expired holder cannot accidentally release a
    newer acquisition.  Locks are coordination primitives only and do not
    replace the compare-and-swap checks on persisted domain state.
    """

    normalized_key = lock_key.strip()
    if not normalized_key or len(normalized_key) > 255:
        raise ValueError("lock_key must contain 1 to 255 non-whitespace characters")
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or not 1 <= ttl_seconds <= 3600
    ):
        raise ValueError("ttl_seconds must be an integer from 1 through 3600")

    acquired = datetime.now(UTC)
    acquired_at = acquired.isoformat(timespec="milliseconds")
    expires_at = (acquired + timedelta(seconds=ttl_seconds)).isoformat(
        timespec="milliseconds"
    )
    token = new_id("lock")
    with transaction() as db:
        db.execute(
            "DELETE FROM operation_locks WHERE lock_key = ? AND expires_at <= ?",
            (normalized_key, acquired_at),
        )
        inserted = db.execute(
            """
            INSERT OR IGNORE INTO operation_locks(lock_key, token, acquired_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (normalized_key, token, acquired_at, expires_at),
        )
        did_acquire = inserted.rowcount == 1
    return token if did_acquire else None


def release_operation_lock(lock_key: str, token: str) -> bool:
    """Release only the acquisition identified by ``token``."""

    normalized_key = lock_key.strip()
    normalized_token = token.strip()
    if not normalized_key or not normalized_token:
        return False
    with transaction() as db:
        deleted = db.execute(
            "DELETE FROM operation_locks WHERE lock_key = ? AND token = ?",
            (normalized_key, normalized_token),
        )
        did_release = deleted.rowcount == 1
    return did_release


def domain_snapshot_hash(domain_event: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(domain_event)).encode("utf-8")).hexdigest()


def _state_snapshot_fingerprint(state: EntrepreneurshipState) -> str:
    # Keep this representation aligned with EventScheduler's audit fingerprint.
    material = json.dumps(state.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def _scheduled_event_from_snapshot(domain_event: Mapping[str, Any]) -> ScheduledEvent:
    """Rebuild the immutable domain event used for persistence validation."""

    try:
        return ScheduledEvent(
            event_kind=str(domain_event["event_kind"]),
            blueprint_id=str(domain_event["blueprint_id"]),
            family_id=str(domain_event["family_id"]),
            form_id=str(domain_event["form_id"]),
            title=str(domain_event["title"]),
            target_dimensions=tuple(domain_event["target_dimensions"]),
            difficulty_level=int(domain_event["difficulty_level"]),
            state_variant_id=str(domain_event["state_variant_id"]),
            visible_state=EntrepreneurshipState.from_dict(
                domain_event["visible_state"]
            ),
            event_text=str(domain_event["event_text"]),
            form_fixed=bool(domain_event["form_fixed"]),
            allowed_actions=tuple(domain_event["allowed_actions"]),
            evidence_requirements=dict(domain_event["evidence_requirements"]),
            selection_probability=float(domain_event["selection_probability"]),
            scheduler_audit=dict(domain_event["scheduler_audit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeliveryIntegrityError(
            "persisted domain event cannot be reconstructed"
        ) from exc


def _validate_allocation_binding(
    *,
    direction: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    event_kind: str,
) -> None:
    """Keep persisted deliveries on the simulation's registered allocation path."""

    registered_mode = direction.get("anchor_allocation_mode")
    delivery_mode = scheduler.get("allocation_mode")
    if registered_mode is None:
        # Legacy/non-web fixtures may predate explicit development registration,
        # but a planned delivery must never attach to an unbound simulation.
        if delivery_mode == "precomputed_queue_balanced":
            raise DeliveryIntegrityError(
                "planned delivery requires a simulation-level anchor registration"
            )
        return
    if registered_mode == "development_hash_fallback":
        if delivery_mode != registered_mode:
            raise DeliveryIntegrityError(
                "development simulation cannot accept a planned anchor delivery"
            )
        if (
            direction.get("anchor_plan_hash") is not None
            or direction.get("allocation_unit_id") is not None
        ):
            raise DeliveryIntegrityError(
                "development allocation registration contains unexpected plan metadata"
            )
        if any(
            scheduler.get(field) is not None
            for field in (
                "allocation_plan_hash",
                "allocation_unit_id",
                "assignment_hash",
                "checkpoint_index",
            )
        ):
            raise DeliveryIntegrityError(
                "development delivery contains unexpected frozen-plan metadata"
            )
    elif registered_mode == "precomputed_queue_balanced":
        plan_hash = direction.get("anchor_plan_hash")
        unit_id = direction.get("allocation_unit_id")
        if not isinstance(plan_hash, str) or not plan_hash:
            raise DeliveryIntegrityError("planned simulation is missing anchor_plan_hash")
        if not isinstance(unit_id, str) or not unit_id:
            raise DeliveryIntegrityError("planned simulation is missing allocation_unit_id")
        if (
            delivery_mode != registered_mode
            or scheduler.get("allocation_plan_hash") != plan_hash
            or scheduler.get("allocation_unit_id") != unit_id
        ):
            raise DeliveryIntegrityError(
                "delivery plan hash or allocation unit differs from simulation registration"
            )
        if event_kind in {"exact_anchor", "surface_anchor"} and (
            not isinstance(scheduler.get("assignment_hash"), str)
            or not isinstance(scheduler.get("checkpoint_index"), int)
        ):
            raise DeliveryIntegrityError(
                "planned anchor delivery is missing assignment hash or checkpoint"
            )
        if event_kind == "main" and (
            scheduler.get("assignment_hash") is not None
            or scheduler.get("checkpoint_index") is not None
        ):
            raise DeliveryIntegrityError(
                "planned main delivery contains anchor-assignment metadata"
            )
    else:
        raise DeliveryIntegrityError("simulation has an unsupported anchor allocation mode")

    registered_interval = direction.get("anchor_every_turns")
    if (
        isinstance(registered_interval, bool)
        or not isinstance(registered_interval, int)
        or scheduler.get("anchor_every_turns") != registered_interval
    ):
        raise DeliveryIntegrityError(
            "delivery anchor interval differs from simulation registration"
        )


def _validate_delivery_components(
    *,
    event_kind: str,
    blueprint_id: str,
    form_id: str,
    selection_probability: float,
    scheduler: Mapping[str, Any],
    visible_state: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[str, str]:
    """Validate duplicated audit fields and return library version and hash."""

    domain_event = event.get("domain_event")
    if not isinstance(domain_event, Mapping):
        raise DeliveryIntegrityError("event payload is missing domain_event")
    event_scheduler = domain_event.get("scheduler_audit")
    if not isinstance(event_scheduler, Mapping):
        raise DeliveryIntegrityError("domain event is missing scheduler audit")
    library_version = event_scheduler.get("library_version")
    if not isinstance(library_version, str) or not library_version.strip():
        raise DeliveryIntegrityError("domain event is missing an immutable library version")

    expected_scalars = {
        "event_kind": event_kind,
        "blueprint_id": blueprint_id,
        "form_id": form_id,
    }
    for field, expected in expected_scalars.items():
        if domain_event.get(field) != expected:
            raise DeliveryIntegrityError(
                f"outer {field} does not match canonical domain snapshot"
            )
    try:
        nested_probability = float(domain_event["selection_probability"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeliveryIntegrityError("domain event has invalid selection_probability") from exc
    if not math.isclose(
        nested_probability, float(selection_probability), rel_tol=0.0, abs_tol=1e-12
    ):
        raise DeliveryIntegrityError(
            "outer selection_probability does not match canonical domain snapshot"
        )
    try:
        outer_state = EntrepreneurshipState.from_dict(visible_state).to_dict()
        nested_state = EntrepreneurshipState.from_dict(domain_event["visible_state"]).to_dict()
    except (KeyError, ValueError) as exc:
        raise DeliveryIntegrityError(f"invalid visible state: {exc}") from exc
    if outer_state != nested_state:
        raise DeliveryIntegrityError(
            "outer visible_state does not match canonical domain snapshot"
        )
    if canonical_json(dict(scheduler)) != canonical_json(dict(event_scheduler)):
        raise DeliveryIntegrityError(
            "outer scheduler audit does not match canonical domain snapshot"
        )
    if event_kind in {"exact_anchor", "surface_anchor"}:
        content = event.get("content")
        if not isinstance(content, Mapping):
            raise DeliveryIntegrityError("frozen anchor is missing learner-visible content")
        if content.get("narrative") != domain_event.get("event_text"):
            raise DeliveryIntegrityError(
                "frozen anchor narrative differs from its canonical event text"
            )
        raw_actions = domain_event.get("allowed_actions")
        if not isinstance(raw_actions, list) or any(
            not isinstance(item, Mapping) or not isinstance(item.get("action_code"), str)
            for item in raw_actions
        ):
            raise DeliveryIntegrityError("frozen anchor has invalid canonical actions")
        expected_codes = [str(item["action_code"]) for item in raw_actions]
        if content.get("allowed_action_codes") != expected_codes:
            raise DeliveryIntegrityError(
                "frozen anchor action whitelist differs from its canonical snapshot"
            )
        if content.get("learning_focus") != domain_event.get("target_dimensions"):
            raise DeliveryIntegrityError(
                "frozen anchor learning focus differs from its canonical snapshot"
            )
    return library_version, domain_snapshot_hash(domain_event)


def validate_delivery_snapshot(
    delivery: Mapping[str, Any], *, allow_legacy: bool = False
) -> None:
    """Recompute and verify all duplicated delivery fields after loading."""

    stored_library = str(delivery.get("library_version") or "")
    stored_hash = delivery.get("domain_snapshot_hash")
    if allow_legacy and stored_library == "legacy-unversioned" and not stored_hash:
        return
    library_version, snapshot_hash = _validate_delivery_components(
        event_kind=str(delivery["event_kind"]),
        blueprint_id=str(delivery["blueprint_id"]),
        form_id=str(delivery["form_id"]),
        selection_probability=float(delivery["selection_probability"]),
        scheduler=delivery["scheduler_json"],
        visible_state=delivery["visible_state_json"],
        event=delivery["event_json"],
    )
    if stored_library != library_version:
        raise DeliveryIntegrityError("stored library version does not match domain snapshot")
    if not isinstance(stored_hash, str) or not hmac.compare_digest(stored_hash, snapshot_hash):
        raise DeliveryIntegrityError("stored domain snapshot hash verification failed")


def create_user(alias: str, access_code_hash: str, consent_accepted: bool) -> str:
    user_id = new_id("usr")
    now = utc_now()
    with transaction() as db:
        db.execute(
            """
            INSERT INTO users(id, alias, access_code_hash, consent_accepted, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, alias, access_code_hash, int(consent_accepted), now, now),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="workspace.created",
            entity_type="user",
            entity_id=user_id,
            payload={"consent_accepted": bool(consent_accepted)},
        )
    return user_id


def get_user(user_id: str) -> dict[str, Any] | None:
    return row_to_dict(fetch_one("SELECT * FROM users WHERE id = ?", (user_id,)))


def users_by_alias(alias: str) -> list[dict[str, Any]]:
    return [dict(row) for row in fetch_all("SELECT * FROM users WHERE alias = ?", (alias,))]


def record_llm_call(
    *,
    user_id: str | None,
    purpose: str,
    provider: str,
    model: str,
    api_version: str | None,
    prompt_version: str,
    prompt_hash: str,
    parameters: Mapping[str, Any],
    input_hash: str,
    output: Mapping[str, Any] | None,
    status: str,
    error_code: str | None,
    latency_ms: int,
    attempts: int,
) -> str:
    if status == "success" and (output is None or error_code is not None):
        raise ValueError("A successful LLM call requires output and no error_code.")
    if status == "failed" and (output is not None or not error_code):
        raise ValueError("A failed LLM call requires an error_code and no output.")
    if status not in {"success", "failed"}:
        raise ValueError("Unsupported LLM call status.")
    call_id = new_id("call")
    with transaction() as db:
        if user_id is not None and db.execute(
            "SELECT 1 FROM users WHERE id = ?", (user_id,)
        ).fetchone() is None:
            raise LookupError("Cannot record an LLM call for an unknown learner.")
        db.execute(
            """
            INSERT INTO llm_calls(
                id, user_id, purpose, provider, model, api_version,
                prompt_version, prompt_hash, parameters_json, input_hash,
                output_json, status, error_code, latency_ms, attempts, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call_id,
                user_id,
                purpose,
                provider,
                model,
                api_version,
                prompt_version,
                prompt_hash,
                canonical_json(dict(parameters)),
                input_hash,
                canonical_json(dict(output)) if output is not None else None,
                status,
                error_code,
                int(latency_ms),
                int(attempts),
                utc_now(),
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id) if user_id else None,
            action=f"llm.{status}",
            entity_type="llm_call",
            entity_id=call_id,
            payload={
                "purpose": purpose,
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
                "error_code": error_code,
                "attempts": int(attempts),
            },
        )
    return call_id


def save_profile(
    user_id: str,
    source: Mapping[str, Any],
    profile: Mapping[str, Any],
    prompt_version: str,
    llm_call_id: str | None,
) -> str:
    profile_id = new_id("profile")
    with transaction() as db:
        if llm_call_id is None:
            if not prompt_version.startswith("learner-correction-"):
                raise DeliveryIntegrityError(
                    "Only an explicit learner correction may create a profile without an LLM call."
                )
        else:
            profile_call = _require_successful_llm_call(
                db, call_id=llm_call_id, user_id=user_id, purpose="profile"
            )
            if str(profile_call["prompt_version"]) != prompt_version:
                raise DeliveryIntegrityError(
                    "profile prompt_version differs from its successful LLM call"
                )
            _require_exact_llm_output(
                dict(profile_call), purpose="profile", expected=profile
            )
        current = db.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        version = int(current["version"]) + 1
        db.execute("UPDATE profiles SET is_active = 0 WHERE user_id = ?", (user_id,))
        db.execute(
            """
            INSERT INTO profiles(
                id, user_id, version, source_json, profile_json, prompt_version,
                llm_call_id, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                profile_id,
                user_id,
                version,
                canonical_json(dict(source)),
                canonical_json(dict(profile)),
                prompt_version,
                llm_call_id,
                utc_now(),
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="profile.version_created",
            entity_type="profile",
            entity_id=profile_id,
            payload={"version": version, "prompt_version": prompt_version},
        )
    return profile_id


def active_profile(user_id: str) -> dict[str, Any] | None:
    return row_to_dict(
        fetch_one(
            "SELECT * FROM profiles WHERE user_id = ? AND is_active = 1 ORDER BY version DESC LIMIT 1",
            (user_id,),
        ),
        ("source_json", "profile_json"),
    )


def save_recommendations(
    user_id: str,
    profile_id: str,
    recommendations: Mapping[str, Any],
    prompt_version: str,
    llm_call_id: str,
) -> str:
    recommendation_id = new_id("rec")
    with transaction() as db:
        recommendation_call = _require_successful_llm_call(
            db, call_id=llm_call_id, user_id=user_id, purpose="recommendation"
        )
        if str(recommendation_call["prompt_version"]) != prompt_version:
            raise DeliveryIntegrityError(
                "recommendation prompt_version differs from its successful LLM call"
            )
        _require_exact_llm_output(
            dict(recommendation_call),
            purpose="recommendation",
            expected=recommendations,
        )
        profile_owner = db.execute(
            "SELECT user_id FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if profile_owner is None or profile_owner["user_id"] != user_id:
            raise DeliveryIntegrityError(
                "Recommendations must reference a profile owned by the learner."
            )
        db.execute(
            """
            INSERT INTO recommendations(
                id, user_id, profile_id, recommendations_json, prompt_version,
                llm_call_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recommendation_id,
                user_id,
                profile_id,
                canonical_json(dict(recommendations)),
                prompt_version,
                llm_call_id,
                utc_now(),
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="recommendations.created",
            entity_type="recommendations",
            entity_id=recommendation_id,
            payload={"profile_id": profile_id, "prompt_version": prompt_version},
        )
    return recommendation_id


def latest_recommendations(user_id: str) -> dict[str, Any] | None:
    return row_to_dict(
        fetch_one(
            "SELECT * FROM recommendations WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ),
        ("recommendations_json",),
    )


def create_simulation(
    user_id: str,
    recommendation_id: str | None,
    direction: Mapping[str, Any],
    initial_state: Mapping[str, Any],
) -> str:
    simulation_id = new_id("sim")
    now = utc_now()
    with transaction() as db:
        db.execute(
            """
            INSERT INTO simulations(
                id, user_id, recommendation_id, direction_json, main_state_json,
                turn_index, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, 'active', ?, ?)
            """,
            (
                simulation_id,
                user_id,
                recommendation_id,
                canonical_json(dict(direction)),
                canonical_json(dict(initial_state)),
                now,
                now,
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="simulation.started",
            entity_type="simulation",
            entity_id=simulation_id,
            payload={"recommendation_id": recommendation_id},
        )
    return simulation_id


def get_simulation(simulation_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    query = "SELECT * FROM simulations WHERE id = ?"
    parameters: tuple[Any, ...] = (simulation_id,)
    if user_id is not None:
        query += " AND user_id = ?"
        parameters += (user_id,)
    return row_to_dict(fetch_one(query, parameters), ("direction_json", "main_state_json"))


def active_simulation(user_id: str) -> dict[str, Any] | None:
    return row_to_dict(
        fetch_one(
            """
            SELECT * FROM simulations
            WHERE user_id = ? AND status = 'active'
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id,),
        ),
        ("direction_json", "main_state_json"),
    )


def pending_delivery(simulation_id: str) -> dict[str, Any] | None:
    delivery = row_to_dict(
        fetch_one(
            """
            SELECT * FROM event_deliveries
            WHERE simulation_id = ? AND status = 'pending'
            ORDER BY turn_no DESC LIMIT 1
            """,
            (simulation_id,),
        ),
        ("scheduler_json", "visible_state_json", "event_json"),
    )
    if delivery is not None:
        validate_delivery_snapshot(delivery)
    return delivery


def save_delivery(
    *,
    user_id: str,
    simulation_id: str,
    turn_no: int,
    event_kind: str,
    blueprint_id: str,
    form_id: str,
    selection_probability: float,
    scheduler: Mapping[str, Any],
    visible_state: Mapping[str, Any],
    event: Mapping[str, Any],
    generation_call_id: str | None,
) -> str:
    delivery_id = new_id("event")
    library_version, snapshot_hash = _validate_delivery_components(
        event_kind=event_kind,
        blueprint_id=blueprint_id,
        form_id=form_id,
        selection_probability=selection_probability,
        scheduler=scheduler,
        visible_state=visible_state,
        event=event,
    )
    with transaction() as db:
        simulation = db.execute(
            """
            SELECT user_id, turn_index, status, main_state_json, direction_json
            FROM simulations WHERE id = ?
            """,
            (simulation_id,),
        ).fetchone()
        if simulation is None or simulation["user_id"] != user_id:
            raise LookupError("Simulation was not found for this learner.")
        if simulation["status"] != "active":
            raise ValueError("Cannot deliver an event to an inactive simulation.")
        try:
            simulation_direction = json.loads(simulation["direction_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise DeliveryIntegrityError("simulation direction is invalid JSON") from exc
        if not isinstance(simulation_direction, Mapping):
            raise DeliveryIntegrityError("simulation direction must be an object")
        _validate_allocation_binding(
            direction=simulation_direction,
            scheduler=scheduler,
            event_kind=event_kind,
        )
        expected_turn = int(simulation["turn_index"]) + 1
        if int(turn_no) != expected_turn:
            raise ValueError(
                f"Delivery turn must be {expected_turn}; received {int(turn_no)}."
            )
        try:
            simulation_state = EntrepreneurshipState.from_dict(
                json.loads(simulation["main_state_json"])
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DeliveryIntegrityError(f"invalid simulation state: {exc}") from exc
        if event_kind == "main":
            if EntrepreneurshipState.from_dict(visible_state) != simulation_state:
                raise DeliveryIntegrityError(
                    "main delivery visible state does not match current simulation state"
                )
            generation_call = db.execute(
                """
                SELECT * FROM llm_calls
                WHERE id = ? AND user_id = ? AND purpose = 'event_narrative'
                      AND status = 'success'
                """,
                (generation_call_id, user_id),
            ).fetchone()
            if generation_call is None:
                raise DeliveryIntegrityError(
                    "main delivery generation_call_id is not a successful "
                    "event-narrative call for this learner"
                )
            content = event.get("content")
            if not isinstance(content, Mapping):
                raise DeliveryIntegrityError(
                    "main delivery is missing structured learner-visible content"
                )
            _require_exact_llm_output(
                dict(generation_call),
                purpose="event_narrative",
                expected=content,
            )
        elif event_kind in {"exact_anchor", "surface_anchor"}:
            if generation_call_id is not None:
                raise DeliveryIntegrityError(
                    "frozen anchor delivery cannot reference an LLM generation call"
                )
            expected_fingerprint = scheduler.get("main_state_snapshot_hash")
            if expected_fingerprint != _state_snapshot_fingerprint(simulation_state):
                raise DeliveryIntegrityError(
                    "anchor delivery main-state fingerprint does not match current simulation state"
                )
        else:
            raise DeliveryIntegrityError(f"unsupported delivery event kind {event_kind!r}")
        existing_pending = db.execute(
            "SELECT id FROM event_deliveries WHERE simulation_id = ? AND status = 'pending'",
            (simulation_id,),
        ).fetchone()
        if existing_pending is not None:
            raise ValueError("Simulation already has a pending event delivery.")
        db.execute(
            """
            INSERT INTO event_deliveries(
                id, simulation_id, turn_no, event_kind, blueprint_id, form_id,
                selection_probability, scheduler_json, visible_state_json,
                event_json, library_version, domain_snapshot_hash,
                generation_call_id, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                delivery_id,
                simulation_id,
                int(turn_no),
                event_kind,
                blueprint_id,
                form_id,
                float(selection_probability),
                canonical_json(dict(scheduler)),
                canonical_json(dict(visible_state)),
                canonical_json(dict(event)),
                library_version,
                snapshot_hash,
                generation_call_id,
                utc_now(),
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="event.delivered",
            entity_type="event_delivery",
            entity_id=delivery_id,
            payload={
                "simulation_id": simulation_id,
                "turn_no": int(turn_no),
                "event_kind": event_kind,
                "blueprint_id": blueprint_id,
                "form_id": form_id,
                "selection_probability": float(selection_probability),
                "library_version": library_version,
                "domain_snapshot_hash": snapshot_hash,
            },
        )
    return delivery_id


def get_delivery(delivery_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    query = """
        SELECT e.* FROM event_deliveries e
        JOIN simulations s ON s.id = e.simulation_id
        WHERE e.id = ?
    """
    parameters: tuple[Any, ...] = (delivery_id,)
    if user_id is not None:
        query += " AND s.user_id = ?"
        parameters += (user_id,)
    delivery = row_to_dict(
        fetch_one(query, parameters),
        ("scheduler_json", "visible_state_json", "event_json"),
    )
    if delivery is not None:
        validate_delivery_snapshot(delivery)
    return delivery


def complete_delivery(
    *,
    user_id: str,
    delivery_id: str,
    decision_text: str,
    rationale_text: str,
    information_requests: str,
    revision_text: str,
    action_code: str,
    classification_confidence: float,
    evidence: Mapping[str, Any],
    feedback: Mapping[str, Any],
    main_state_before: Mapping[str, Any],
    main_state_after: Mapping[str, Any],
    sandbox_state_after: Mapping[str, Any] | None,
    scoring_call_id: str,
) -> str:
    response_id = new_id("response")
    with transaction() as db:
        delivery = db.execute(
            """
            SELECT e.*, s.user_id,
                   s.main_state_json AS simulation_state_json,
                   s.direction_json AS simulation_direction_json,
                   s.turn_index AS simulation_turn_index,
                   s.status AS simulation_status
            FROM event_deliveries e
            JOIN simulations s ON s.id = e.simulation_id
            WHERE e.id = ?
            """,
            (delivery_id,),
        ).fetchone()
        if delivery is None or delivery["user_id"] != user_id:
            raise LookupError("Event delivery was not found for this learner.")
        if delivery["status"] != "pending":
            raise ValueError("This event has already been completed or cancelled.")

        parsed_delivery = dict(delivery)
        for field in ("scheduler_json", "visible_state_json", "event_json"):
            parsed_delivery[field] = json.loads(parsed_delivery[field])
        validate_delivery_snapshot(parsed_delivery)
        try:
            simulation_direction = json.loads(
                delivery["simulation_direction_json"]
            )
        except (json.JSONDecodeError, TypeError) as exc:
            raise DeliveryIntegrityError("simulation direction is invalid JSON") from exc
        if not isinstance(simulation_direction, Mapping):
            raise DeliveryIntegrityError("simulation direction must be an object")
        _validate_allocation_binding(
            direction=simulation_direction,
            scheduler=parsed_delivery["scheduler_json"],
            event_kind=str(delivery["event_kind"]),
        )

        if delivery["simulation_status"] != "active":
            raise ValueError("Cannot complete an event for an inactive simulation.")
        expected_turn = int(delivery["simulation_turn_index"]) + 1
        if int(delivery["turn_no"]) != expected_turn:
            raise DeliveryIntegrityError(
                f"stale delivery turn: expected {expected_turn}, got {delivery['turn_no']}"
            )
        try:
            current_state = EntrepreneurshipState.from_dict(
                json.loads(delivery["simulation_state_json"])
            )
            before_state = EntrepreneurshipState.from_dict(main_state_before)
            after_state = EntrepreneurshipState.from_dict(main_state_after)
        except (json.JSONDecodeError, ValueError) as exc:
            raise DeliveryIntegrityError(f"invalid transition state: {exc}") from exc
        if current_state != before_state:
            raise DeliveryIntegrityError(
                "simulation state changed after delivery; refusing stale state overwrite"
            )

        event_kind = str(delivery["event_kind"])
        provided_sandbox_after: EntrepreneurshipState | None = None
        if event_kind in {"exact_anchor", "surface_anchor"}:
            if before_state != after_state:
                raise DeliveryIntegrityError(
                    "anchor completion must restore the unchanged personal main state"
                )
            if sandbox_state_after is None:
                raise DeliveryIntegrityError("anchor completion requires a sandbox result")
            try:
                provided_sandbox_after = EntrepreneurshipState.from_dict(
                    sandbox_state_after
                )
            except ValueError as exc:
                raise DeliveryIntegrityError(f"invalid anchor sandbox result: {exc}") from exc
            expected_fingerprint = parsed_delivery["scheduler_json"].get(
                "main_state_snapshot_hash"
            )
            if expected_fingerprint != _state_snapshot_fingerprint(before_state):
                raise DeliveryIntegrityError(
                    "anchor completion state differs from its delivery-time main-state snapshot"
                )
        elif event_kind == "main":
            if sandbox_state_after is not None:
                raise DeliveryIntegrityError("main-path completion cannot store a sandbox result")
            visible_state = EntrepreneurshipState.from_dict(
                parsed_delivery["visible_state_json"]
            )
            if visible_state != before_state:
                raise DeliveryIntegrityError(
                    "main-path delivery snapshot does not match transition input state"
                )
        else:
            raise DeliveryIntegrityError(f"unsupported persisted event kind {event_kind!r}")

        # Re-run the only authorized numeric transition from the immutable
        # library and persisted event snapshot.  The route already performs
        # this operation for the learner response, but persistence must not
        # accept a caller-supplied action or after-state that bypasses the rule
        # engine.  A library upgrade also fails closed because the event keeps
        # its original library version in scheduler_audit.
        try:
            active_library = BlueprintLibrary.load_default()
            scheduled_event = _scheduled_event_from_snapshot(
                parsed_delivery["event_json"]["domain_event"]
            )
            expected_transition = apply_classified_action(
                library=active_library,
                event=scheduled_event,
                main_state=current_state,
                action_code=action_code,
            )
        except (KeyError, StateTransitionError, ValueError) as exc:
            raise DeliveryIntegrityError(
                "response action cannot be verified against the frozen rule library"
            ) from exc
        if (
            expected_transition.main_state_before != before_state
            or expected_transition.main_state_after != after_state
        ):
            raise DeliveryIntegrityError(
                "caller-supplied main state differs from the frozen rule transition"
            )
        if expected_transition.sandbox_state_after != provided_sandbox_after:
            raise DeliveryIntegrityError(
                "caller-supplied sandbox state differs from the frozen rule transition"
            )

        scoring_call = db.execute(
            """
            SELECT * FROM llm_calls
            WHERE id = ? AND user_id = ? AND purpose = 'decision_evidence'
                  AND status = 'success'
            """,
            (scoring_call_id, user_id),
        ).fetchone()
        if scoring_call is None:
            raise DeliveryIntegrityError(
                "scoring_call_id is not a successful decision-evidence call for this learner"
            )
        scoring_output = _llm_call_output(
            dict(scoring_call), purpose="decision_evidence"
        )
        try:
            validate_structured_output("decision_evidence", scoring_output)
        except SchemaValidationError as exc:
            raise DeliveryIntegrityError(
                "decision-evidence call output no longer satisfies its schema"
            ) from exc
        if scoring_output["action_code"] != action_code:
            raise DeliveryIntegrityError(
                "stored action_code differs from the decision-evidence call"
            )
        if not math.isclose(
            float(scoring_output["classification_confidence"]),
            float(classification_confidence),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DeliveryIntegrityError(
                "stored classification confidence differs from the decision-evidence call"
            )
        expected_evidence = {"evidence": scoring_output["evidence"]}
        if canonical_json(dict(evidence)) != canonical_json(expected_evidence):
            raise DeliveryIntegrityError(
                "stored ECD evidence differs from the decision-evidence call"
            )
        model_feedback = scoring_output["feedback"]
        if any(
            key not in feedback
            or canonical_json(feedback[key]) != canonical_json(value)
            for key, value in model_feedback.items()
        ):
            raise DeliveryIntegrityError(
                "stored feedback differs from the decision-evidence call"
            )
        if (
            feedback.get("mapping_uncertainty")
            != scoring_output["mapping_uncertainty"]
            or feedback.get("safety_flags") != scoring_output["safety_flags"]
        ):
            raise DeliveryIntegrityError(
                "stored mapping or safety metadata differs from the decision-evidence call"
            )
        if canonical_json(feedback.get("transition")) != canonical_json(
            expected_transition.to_dict()
        ):
            raise DeliveryIntegrityError(
                "stored feedback transition differs from the frozen rule transition"
            )
        if (
            scoring_output["mapping_uncertainty"]["level"] != "low"
            or float(scoring_output["classification_confidence"]) < 0.70
        ):
            raise DeliveryIntegrityError(
                "an uncertain action mapping cannot advance simulation state"
            )
        if scoring_output["safety_flags"] != ["none"]:
            raise DeliveryIntegrityError(
                "a safety-flagged decision cannot advance simulation state"
            )
        learner_text = "\n".join(
            (decision_text, rationale_text, information_requests, revision_text)
        )
        if any(
            quote not in learner_text
            for dimension in scoring_output["evidence"]
            for quote in dimension["quotes"]
        ):
            raise DeliveryIntegrityError(
                "decision-evidence call cites text outside the stored learner response"
            )

        now = utc_now()
        db.execute(
            """
            INSERT INTO responses(
                id, delivery_id, decision_text, rationale_text,
                information_requests, revision_text, action_code,
                classification_confidence, evidence_json, feedback_json,
                main_state_before_json, main_state_after_json,
                sandbox_state_after_json, scoring_call_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id,
                delivery_id,
                decision_text,
                rationale_text,
                information_requests,
                revision_text,
                action_code,
                float(classification_confidence),
                canonical_json(dict(evidence)),
                canonical_json(dict(feedback)),
                canonical_json(dict(main_state_before)),
                canonical_json(dict(main_state_after)),
                canonical_json(dict(sandbox_state_after)) if sandbox_state_after else None,
                scoring_call_id,
                now,
            ),
        )
        db.execute(
            """
            UPDATE event_deliveries SET status = 'completed', completed_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (now, delivery_id),
        )
        if db.execute("SELECT changes()").fetchone()[0] != 1:
            raise DeliveryIntegrityError("event completion lost a concurrent update race")
        state_before_json = canonical_json(before_state.to_dict())
        state_after_json = canonical_json(after_state.to_dict())
        updated = db.execute(
            """
            UPDATE simulations
            SET main_state_json = ?, turn_index = ?, updated_at = ?
            WHERE id = ? AND turn_index = ? AND main_state_json = ? AND status = 'active'
            """,
            (
                state_after_json,
                int(delivery["turn_no"]),
                now,
                delivery["simulation_id"],
                int(delivery["simulation_turn_index"]),
                state_before_json,
            ),
        )
        if updated.rowcount != 1:
            raise DeliveryIntegrityError("simulation state compare-and-swap failed")

        facets = _rating_facets_from_call(dict(scoring_call))
        rater_key = f"llm:{facets['provider']}:{facets['model']}"
        version_key = rating_version_key(**facets)
        rating_time_key = f"call:{scoring_call_id}"
        rating_id = new_id("rating")
        db.execute(
            """
            INSERT INTO rating_observations(
                id, response_id, rater_key, rater_type, version_key,
                provider, model, api_version, prompt_version,
                prompt_template_hash, schema_hash, build_id, prompt_hash,
                rating_time_key, evidence_json, source_call_id, created_at
            ) VALUES (?, ?, ?, 'llm', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rating_id,
                response_id,
                rater_key,
                version_key,
                facets["provider"],
                facets["model"],
                facets["api_version"],
                facets["prompt_version"],
                facets["prompt_template_hash"],
                facets["schema_hash"],
                facets["build_id"],
                scoring_call["prompt_hash"],
                rating_time_key,
                canonical_json(dict(evidence)),
                scoring_call_id,
                now,
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="event.response_recorded",
            entity_type="response",
            entity_id=response_id,
            payload={
                "delivery_id": delivery_id,
                "event_kind": delivery["event_kind"],
                "blueprint_id": delivery["blueprint_id"],
                "form_id": delivery["form_id"],
                "action_code": action_code,
                "scoring_call_id": scoring_call_id,
                "rating_id": rating_id,
            },
        )
    return response_id


def save_rating_observation(
    *,
    user_id: str,
    response_id: str,
    rater_key: str,
    rater_type: str,
    rating_time_key: str,
    evidence: Mapping[str, Any],
    provider: str | None = None,
    model: str | None = None,
    api_version: str | None = None,
    prompt_version: str | None = None,
    prompt_template_hash: str | None = None,
    schema_hash: str | None = None,
    build_id: str | None = None,
    prompt_hash: str | None = None,
    source_call_id: str | None = None,
) -> str:
    """Append one independently attributable rating for a stored response.

    This is storage plumbing only.  It makes later crossed human/API ratings
    and repeated occasions representable; calling it does not establish
    rating quality or validity.  ``version_key`` is derived here from reusable
    model/prompt/schema/build facets and never from the exact ``prompt_hash``.
    """

    if rater_type not in {"llm", "human", "rule", "other"}:
        raise ValueError("Unsupported rater_type.")
    normalized_rater = rater_key.strip()
    normalized_time = rating_time_key.strip()
    if not normalized_rater or len(normalized_rater) > 255:
        raise ValueError("rater_key must contain 1 to 255 characters.")
    if not normalized_time or len(normalized_time) > 255:
        raise ValueError("rating_time_key must contain 1 to 255 characters.")
    rating_id = new_id("rating")
    with transaction() as db:
        owner = db.execute(
            """
            SELECT s.user_id, r.decision_text, r.rationale_text,
                   r.information_requests, r.revision_text
            FROM responses r
            JOIN event_deliveries e ON e.id = r.delivery_id
            JOIN simulations s ON s.id = e.simulation_id
            WHERE r.id = ?
            """,
            (response_id,),
        ).fetchone()
        if owner is None or owner["user_id"] != user_id:
            raise LookupError("Response was not found for this learner.")
        facets: dict[str, str]
        effective_prompt_hash = prompt_hash
        if source_call_id is not None:
            call = db.execute(
                "SELECT * FROM llm_calls WHERE id = ?",
                (source_call_id,),
            ).fetchone()
            if (
                call is None
                or call["user_id"] != user_id
                or call["status"] != "success"
                or call["purpose"] != "decision_evidence"
            ):
                raise ValueError("source_call_id must reference a successful learner-owned call.")
            facets = _rating_facets_from_call(dict(call))
            if rater_type != "llm":
                raise ValueError("A source LLM call requires rater_type='llm'.")
            expected_rater = f"llm:{facets['provider']}:{facets['model']}"
            if normalized_rater != expected_rater:
                raise ValueError(
                    "rater_key does not match the referenced LLM call provider/model."
                )
            call_output = _llm_call_output(
                dict(call), purpose="decision_evidence"
            )
            try:
                validate_structured_output("decision_evidence", call_output)
            except SchemaValidationError as exc:
                raise ValueError(
                    "Referenced decision-evidence output does not satisfy its schema."
                ) from exc
            expected_evidence = {"evidence": call_output["evidence"]}
            if canonical_json(dict(evidence)) != canonical_json(expected_evidence):
                raise ValueError(
                    "evidence does not match the referenced decision-evidence call."
                )
            response_text = "\n".join(
                str(owner[field] or "")
                for field in (
                    "decision_text",
                    "rationale_text",
                    "information_requests",
                    "revision_text",
                )
            )
            if any(
                quote not in response_text
                for dimension in call_output["evidence"]
                for quote in dimension["quotes"]
            ):
                raise ValueError(
                    "Referenced decision-evidence call cites text outside the stored response."
                )
            supplied = {
                "provider": provider,
                "model": model,
                "api_version": api_version,
                "prompt_version": prompt_version,
                "prompt_template_hash": prompt_template_hash,
                "schema_hash": schema_hash,
                "build_id": build_id,
            }
            for name, value in supplied.items():
                if value is None:
                    continue
                normalized_value = str(value).strip()
                if name == "api_version" and not normalized_value:
                    normalized_value = "unspecified"
                if normalized_value != facets[name]:
                    raise ValueError(
                        f"{name} does not match the referenced LLM call."
                    )
            if prompt_hash is not None and prompt_hash != call["prompt_hash"]:
                raise ValueError("prompt_hash does not match the referenced LLM call.")
            effective_prompt_hash = call["prompt_hash"]
        else:
            facets = _manual_rating_facets(
                provider=provider,
                model=model,
                api_version=api_version,
                prompt_version=prompt_version,
                prompt_template_hash=prompt_template_hash,
                schema_hash=schema_hash,
                build_id=build_id,
            )
        version_key = rating_version_key(**facets)
        duplicate = db.execute(
            """
            SELECT 1 FROM rating_observations
            WHERE response_id = ? AND rater_key = ? AND version_key = ?
                  AND rating_time_key = ?
            """,
            (response_id, normalized_rater, version_key, normalized_time),
        ).fetchone()
        if duplicate is not None:
            raise ValueError(
                "This response/rater/version/rating_time_key occasion already exists."
            )
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
                rating_id,
                response_id,
                normalized_rater,
                rater_type,
                version_key,
                facets["provider"],
                facets["model"],
                facets["api_version"],
                facets["prompt_version"],
                facets["prompt_template_hash"],
                facets["schema_hash"],
                facets["build_id"],
                effective_prompt_hash,
                normalized_time,
                canonical_json(dict(evidence)),
                source_call_id,
                utc_now(),
            ),
        )
        append_audit(
            db,
            subject_ref=subject_ref(user_id),
            action="rating.observation_added",
            entity_type="rating_observation",
            entity_id=rating_id,
            payload={
                "response_id": response_id,
                "rater_key": normalized_rater,
                "rater_type": rater_type,
                "version_key": version_key,
                "rating_time_key": normalized_time,
            },
        )
    return rating_id


def simulation_history(simulation_id: str, user_id: str) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT e.*, r.id AS response_id, r.decision_text, r.rationale_text,
               r.information_requests, r.revision_text, r.action_code,
               r.classification_confidence, r.evidence_json,
               r.feedback_json, r.main_state_before_json,
               r.main_state_after_json, r.sandbox_state_after_json,
               r.created_at AS response_created_at
        FROM event_deliveries e
        JOIN simulations s ON s.id = e.simulation_id
        LEFT JOIN responses r ON r.delivery_id = e.id
        WHERE e.simulation_id = ? AND s.user_id = ?
        ORDER BY e.turn_no
        """,
        (simulation_id, user_id),
    )
    json_fields = (
        "scheduler_json",
        "visible_state_json",
        "event_json",
        "evidence_json",
        "feedback_json",
        "main_state_before_json",
        "main_state_after_json",
        "sandbox_state_after_json",
    )
    labels = {
        "cash": "现金",
        "team": "团队",
        "product": "产品",
        "market_evidence": "市场证据",
        "reputation": "信誉",
        "compliance": "合规",
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        item = row_to_dict(row, json_fields) or {}
        validate_delivery_snapshot(item, allow_legacy=True)
        before = item.get("main_state_before_json") or {}
        after = item.get("main_state_after_json") or {}
        changes = []
        for key, label in labels.items():
            if key in before and key in after and before[key] != after[key]:
                changes.append(
                    {"key": key, "label": label, "before": before[key], "after": after[key]}
                )
        item["state_changes"] = changes
        if item.get("event_kind") in {"exact_anchor", "surface_anchor"}:
            item["restore_note"] = (
                "锚事件后果仅保存在沙盒记录；个性化主线已按进入前状态继续。"
            )
        result.append(item)
    return result


def export_user_data(user_id: str) -> dict[str, Any]:
    user = get_user(user_id)
    if user is None:
        raise LookupError("Workspace not found.")
    user.pop("access_code_hash", None)
    profiles = [
        row_to_dict(row, ("source_json", "profile_json"))
        for row in fetch_all("SELECT * FROM profiles WHERE user_id = ? ORDER BY version", (user_id,))
    ]
    recommendations = [
        row_to_dict(row, ("recommendations_json",))
        for row in fetch_all(
            "SELECT * FROM recommendations WHERE user_id = ? ORDER BY created_at", (user_id,)
        )
    ]
    simulations = [
        row_to_dict(row, ("direction_json", "main_state_json"))
        for row in fetch_all("SELECT * FROM simulations WHERE user_id = ? ORDER BY created_at", (user_id,))
    ]
    histories = {
        simulation["id"]: simulation_history(simulation["id"], user_id)
        for simulation in simulations
        if simulation is not None
    }
    calls = [
        row_to_dict(row, ("parameters_json", "output_json"))
        for row in fetch_all(
            """
            SELECT id, purpose, provider, model, api_version, prompt_version,
                   prompt_hash, parameters_json, input_hash, output_json,
                   status, error_code, latency_ms, attempts, created_at
            FROM llm_calls WHERE user_id = ? ORDER BY created_at
            """,
            (user_id,),
        )
    ]
    ratings = [
        row_to_dict(row, ("evidence_json",))
        for row in fetch_all(
            """
            SELECT q.* FROM rating_observations q
            JOIN responses r ON r.id = q.response_id
            JOIN event_deliveries e ON e.id = r.delivery_id
            JOIN simulations s ON s.id = e.simulation_id
            WHERE s.user_id = ? ORDER BY q.created_at
            """,
            (user_id,),
        )
    ]
    return {
        "exported_at": utc_now(),
        "user": user,
        "profiles": profiles,
        "recommendations": recommendations,
        "simulations": simulations,
        "histories": histories,
        "rating_observations": ratings,
        "llm_call_metadata": calls,
    }


def delete_user_data(user_id: str) -> str:
    deletion_receipt = hashlib.sha256(
        f"{subject_ref(user_id)}:{utc_now()}".encode("utf-8")
    ).hexdigest()[:24]
    with transaction() as db:
        exists = db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
        if exists is None:
            raise LookupError("Workspace not found.")
        simulation_ids = [
            str(row["id"])
            for row in db.execute(
                "SELECT id FROM simulations WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        delivery_ids = [
            str(row["id"])
            for row in db.execute(
                """
                SELECT e.id FROM event_deliveries e
                JOIN simulations s ON s.id = e.simulation_id
                WHERE s.user_id = ?
                """,
                (user_id,),
            ).fetchall()
        ]
        lock_keys = [
            *(f"event-generation:{simulation_id}" for simulation_id in simulation_ids),
            *(f"response-scoring:{delivery_id}" for delivery_id in delivery_ids),
        ]
        if lock_keys:
            db.executemany(
                "DELETE FROM operation_locks WHERE lock_key = ?",
                ((lock_key,) for lock_key in lock_keys),
            )
        db.execute("DELETE FROM llm_calls WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        append_audit(
            db,
            subject_ref=None,
            action="workspace.deleted",
            entity_type="deletion_receipt",
            entity_id=deletion_receipt,
            payload={"scope": "all learner-owned records"},
        )
    return deletion_receipt
