"""Prepare recorded anchor evidence for the proposed psychometric model.

This module deliberately does not fit the model.  It turns operational records
into a transparent Stan data object and manifest so a separately governed
analysis can later decide whether the identification assumptions are met.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import rating_version_key, utc_now
from .domain import EntrepreneurshipState
from .repository import subject_ref, trajectory_ref, validate_delivery_snapshot


DIMENSIONS: tuple[str, ...] = (
    "opportunity_framing",
    "evidence_falsifiability",
    "resource_finance",
    "adaptive_action",
    "stakeholder_ethics",
)

STATE_FIELDS: tuple[str, ...] = (
    "cash",
    "team",
    "product",
    "market_evidence",
    "reputation",
    "compliance",
)


def _dimension_records(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the versioned schema plus two earlier development shapes."""

    for key in ("dimensions", "evidence", "ecd_evidence"):
        candidate = evidence.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
        if isinstance(candidate, dict):
            records: list[dict[str, Any]] = []
            for dimension, item in candidate.items():
                if isinstance(item, dict):
                    records.append({"dimension": dimension, **item})
                elif isinstance(item, (int, float)):
                    records.append({"dimension": dimension, "score": item})
            return records
    return []


def extract_scores(
    evidence: dict[str, Any], *, target_dimensions: tuple[str, ...] | None = None
) -> dict[str, int]:
    """Extract valid ordinal labels, optionally restricted to task targets.

    A target-dimension response with ``evidence_status='insufficient'`` is a
    legitimate observed zero.  A dimension that the task did not target is
    omitted entirely rather than turned into an artificial zero.
    """

    scores: dict[str, int] = {}
    aliases = {
        "opportunity_definition": "opportunity_framing",
        "opportunity_evaluation": "opportunity_framing",
        "evidence_reasoning": "evidence_falsifiability",
        "resource_allocation": "resource_finance",
        "uncertainty_adaptation": "adaptive_action",
        "team_stakeholder_ethics": "stakeholder_ethics",
    }
    allowed = set(target_dimensions or DIMENSIONS)
    seen: set[str] = set()
    for record in _dimension_records(evidence):
        code = record.get("dimension") or record.get("dimension_code") or record.get("code")
        code = aliases.get(str(code), str(code))
        if code not in allowed:
            continue
        if code in seen:
            raise ValueError(f"duplicate ECD dimension {code!r}")
        seen.add(code)
        score = record.get("score", record.get("level"))
        if isinstance(score, bool):
            continue
        if code in DIMENSIONS and isinstance(score, (int, float)):
            integer = int(score)
            if float(score) == integer and 0 <= integer <= 4:
                if record.get("evidence_status") == "insufficient" and integer != 0:
                    raise ValueError(
                        f"insufficient evidence for {code!r} must be encoded as zero"
                    )
                scores[code] = integer
    return scores


def _index(values: list[str]) -> tuple[dict[str, int], list[str]]:
    ordered = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(ordered)}, ordered


def _standardize_state(state: dict[str, int]) -> list[float]:
    """Return the frozen, interpretable covariate coding used by the model.

    ``cash`` is centered at the teaching world's starting balance and measured
    in starting-balance units.  The five bounded state indices are centered at
    50 and measured in 50-index-point units.  This is deliberately not a
    fitted normalization: the same coding is available before any analysis.
    """

    return [
        (float(state["cash"]) - 100_000.0) / 100_000.0,
        *[(float(state[field]) - 50.0) / 50.0 for field in STATE_FIELDS[1:]],
    ]


def _matrix_rank(rows: list[list[float]], *, tolerance: float = 1e-10) -> int:
    """Compute a small design-matrix rank without adding a numerical dependency.

    It is a diagnostic for whether controlled-state columns vary in this
    export.  It is not a claim that the ensuing psychometric parameters are
    identified.
    """

    if not rows:
        return 0
    matrix = [list(map(float, row)) for row in rows]
    width = max((len(row) for row in matrix), default=0)
    if width == 0 or any(len(row) != width for row in matrix):
        return 0
    rank = 0
    pivot_row = 0
    for column in range(width):
        pivot = max(range(pivot_row, len(matrix)), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        denominator = matrix[pivot_row][column]
        matrix[pivot_row] = [value / denominator for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if abs(factor) > tolerance:
                matrix[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(matrix[row], matrix[pivot_row])
                ]
        rank += 1
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return rank


def _bipartite_components(
    edges: list[tuple[tuple[str, str], tuple[str, str]]]
) -> list[set[tuple[str, str]]]:
    """Return connected components for a small undirected bipartite graph."""

    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components: list[set[tuple[str, str]]] = []
    unseen = set(adjacency)
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency.get(node, set()):
                if neighbour not in component:
                    component.add(neighbour)
                    unseen.discard(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _component_summary(
    components: list[set[tuple[str, str]]], *, left_kind: str, right_kind: str
) -> list[dict[str, int]]:
    return [
        {
            "left_node_count": sum(kind == left_kind for kind, _value in component),
            "right_node_count": sum(kind == right_kind for kind, _value in component),
        }
        for component in components
    ]


def build_measurement_export(
    db: sqlite3.Connection,
    *,
    include_training_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build Stan data, a mapping manifest and a human-readable long table.

    By default only randomized exact/surface anchors are eligible.  Main-line
    generated events remain available as training evidence, but are excluded
    because a one-off surface form cannot be assumed to have known difficulty.
    """

    where = (
        "e.event_kind IN ('main', 'exact_anchor', 'surface_anchor')"
        if include_training_only
        else "e.event_kind IN ('exact_anchor', 'surface_anchor')"
    )
    rows = db.execute(
        f"""
        SELECT s.user_id, s.id AS simulation_id, r.id AS response_id,
               q.id AS rating_id, e.turn_no, e.event_kind, e.blueprint_id, e.form_id,
               e.selection_probability, e.scheduler_json, e.visible_state_json,
               e.event_json, e.library_version, e.domain_snapshot_hash,
               q.evidence_json, q.rater_key, q.rater_type, q.version_key,
               q.provider, q.model, q.api_version, q.prompt_version,
               q.prompt_template_hash, q.schema_hash, q.build_id, q.prompt_hash,
               q.rating_time_key, q.created_at AS rating_created_at, q.source_call_id
        FROM rating_observations q
        JOIN responses r ON r.id = q.response_id
        JOIN event_deliveries e ON e.id = r.delivery_id
        JOIN simulations s ON s.id = e.simulation_id
        WHERE {where}
        ORDER BY s.user_id, s.id, e.turn_no, q.id
        """
    ).fetchall()

    observations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        # Verify every selected rating row before any ECD-completeness filtering.
        # A bad reusable version key is an audit-integrity failure, not merely a
        # reason to omit one response from the analysis set.
        rating_facets = {
            "provider": row["provider"],
            "model": row["model"],
            "api_version": row["api_version"],
            "prompt_version": row["prompt_version"],
            "prompt_template_hash": row["prompt_template_hash"],
            "schema_hash": row["schema_hash"],
            "build_id": row["build_id"],
        }
        expected_version_key = rating_version_key(**rating_facets)
        if str(row["version_key"]) != expected_version_key:
            raise ValueError(
                "Rating observation version_key does not match its stable seven-facet "
                f"definition: {row['rating_id']}"
            )
        try:
            evidence = json.loads(row["evidence_json"])
            scheduler = json.loads(row["scheduler_json"])
            visible_state = json.loads(row["visible_state_json"])
            state = EntrepreneurshipState.from_dict(visible_state).to_dict()
            event = json.loads(row["event_json"])
            validate_delivery_snapshot(
                {
                    "event_kind": row["event_kind"],
                    "blueprint_id": row["blueprint_id"],
                    "form_id": row["form_id"],
                    "selection_probability": row["selection_probability"],
                    "scheduler_json": scheduler,
                    "visible_state_json": visible_state,
                    "event_json": event,
                    "library_version": row["library_version"],
                    "domain_snapshot_hash": row["domain_snapshot_hash"],
                }
            )
            domain_event = event["domain_event"]
            target_dimensions = tuple(domain_event["target_dimensions"])
            if (
                not target_dimensions
                or len(set(target_dimensions)) != len(target_dimensions)
                or not set(target_dimensions).issubset(DIMENSIONS)
            ):
                raise ValueError("invalid target_dimensions in domain snapshot")
            scores = extract_scores(
                evidence, target_dimensions=target_dimensions
            )
            state_variant_id = domain_event.get(
                "state_variant_id", scheduler.get("state_variant_id", "legacy_single_state")
            )
            if not isinstance(state_variant_id, str) or not state_variant_id.strip():
                raise ValueError("invalid state_variant_id in domain snapshot")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            skipped.append(
                {
                    "trajectory": trajectory_ref(row["user_id"], row["simulation_id"]),
                    "turn": int(row["turn_no"]),
                    "blueprint": row["blueprint_id"],
                    "rating_id": row["rating_id"],
                    "reason": f"invalid_record:{type(exc).__name__}",
                }
            )
            continue
        missing = [dimension for dimension in target_dimensions if dimension not in scores]
        if missing:
            skipped.append(
                {
                    "trajectory": trajectory_ref(row["user_id"], row["simulation_id"]),
                    "turn": int(row["turn_no"]),
                    "blueprint": row["blueprint_id"],
                    "rating_id": row["rating_id"],
                    "missing_dimensions": missing,
                }
            )
            continue
        analysis_set = (
            "primary_anchor"
            if row["event_kind"] in {"exact_anchor", "surface_anchor"}
            else "training_only"
        )
        form_type = {
            "exact_anchor": "exact",
            "surface_anchor": "surface",
            "main": "main",
        }[row["event_kind"]]
        for dimension in target_dimensions:
            observations.append(
                {
                    "person": trajectory_ref(row["user_id"], row["simulation_id"]),
                    "learner": subject_ref(row["user_id"]),
                    "simulation_id_hash": subject_ref(str(row["simulation_id"])),
                    "response_id": row["response_id"],
                    "rating_id": row["rating_id"],
                    # Keep learning occasions separate from occasions on which
                    # the same response was scored or re-scored.
                    "learner_time": int(row["turn_no"]),
                    "rating_time_key": row["rating_time_key"],
                    "rating_created_at": row["rating_created_at"],
                    # This is an exact per-call audit fingerprint, deliberately
                    # separate from the reusable seven-facet version key.
                    "prompt_hash": row["prompt_hash"],
                    "dimension": dimension,
                    "blueprint": row["blueprint_id"],
                    "form": row["form_id"],
                    "form_type": form_type,
                    "rater": row["rater_key"],
                    "rater_type": row["rater_type"],
                    "version": row["version_key"],
                    "rating_version_facets": rating_facets,
                    "score": scores[dimension],
                    "state": {field: float(state[field]) for field in STATE_FIELDS},
                    "state_z": _standardize_state(state),
                    "state_variant_id": state_variant_id,
                    "event_kind": row["event_kind"],
                    "analysis_set": analysis_set,
                    "selection_probability": float(row["selection_probability"]),
                    "library_version": row["library_version"],
                    "domain_snapshot_hash": row["domain_snapshot_hash"],
                    "source_call_id": row["source_call_id"],
                }
            )

    primary = [item for item in observations if item["analysis_set"] == "primary_anchor"]
    exploratory = [item for item in observations if item["analysis_set"] == "training_only"]
    if not primary:
        raise ValueError("No complete eligible ECD anchor observations are available for export.")

    library_versions = sorted({str(item["library_version"]) for item in primary})
    if len(library_versions) != 1:
        raise ValueError(
            "Primary export mixes immutable blueprint library versions: "
            + ", ".join(library_versions)
        )

    forms_by_blueprint: dict[str, set[str]] = {}
    exact_blueprints: set[str] = set()
    for item in primary:
        forms_by_blueprint.setdefault(item["blueprint"], set()).add(item["form_type"])
        if item["form_type"] == "exact":
            exact_blueprints.add(item["blueprint"])
    missing_exact = sorted(
        blueprint
        for blueprint, form_types in forms_by_blueprint.items()
        if "surface" in form_types and blueprint not in exact_blueprints
    )
    if missing_exact:
        raise ValueError(
            "Surface forms lack an observed exact-form reference for blueprints: "
            + ", ".join(missing_exact)
        )

    observed_dimensions = {item["dimension"] for item in primary}
    missing_dimensions = [
        dimension for dimension in DIMENSIONS if dimension not in observed_dimensions
    ]
    if missing_dimensions:
        raise ValueError(
            "Primary anchor observations do not identify every configured dimension: "
            + ", ".join(missing_dimensions)
        )

    # Common support is a property of the observed deployment, not an assumption
    # supplied to the model.  For every dimension a learner must be connected to
    # the anchor blueprint cells through shared exposures.  Separate components
    # have no empirical route to a common scale, so fail closed before export.
    support_by_dimension: dict[str, dict[str, Any]] = {}
    disconnected_dimensions: list[str] = []
    for dimension in DIMENSIONS:
        dimension_items = [item for item in primary if item["dimension"] == dimension]
        support_edges = [
            (
                ("person", str(item["person"])),
                (
                    "blueprint",
                    f"{item['library_version']}::{item['blueprint']}",
                ),
            )
            for item in dimension_items
        ]
        components = _bipartite_components(support_edges)
        support_by_dimension[dimension] = {
            "component_count": len(components),
            "components": _component_summary(
                components, left_kind="person", right_kind="blueprint"
            ),
            "learner_count": len({item["person"] for item in dimension_items}),
            "blueprint_cell_count": len(
                {
                    f"{item['library_version']}::{item['blueprint']}"
                    for item in dimension_items
                }
            ),
        }
        if len(components) != 1:
            disconnected_dimensions.append(dimension)
    if disconnected_dimensions:
        raise ValueError(
            "Primary anchor learner-to-blueprint common support is disconnected for "
            "dimensions: " + ", ".join(disconnected_dimensions)
        )

    person_map, people = _index([item["person"] for item in primary])
    blueprint_map, blueprints = _index([item["blueprint"] for item in primary])
    blueprint_dimension_pairs = sorted(
        {(item["blueprint"], item["dimension"]) for item in primary},
        key=lambda pair: (DIMENSIONS.index(pair[1]), pair[0]),
    )
    blueprint_dimension_map = {
        pair: index + 1 for index, pair in enumerate(blueprint_dimension_pairs)
    }
    surface_map, surface_forms = _index(
        [item["form"] for item in primary if item["form_type"] == "surface"]
    )
    rater_map, raters = _index([item["rater"] for item in primary])
    version_map, versions = _index([item["version"] for item in primary])
    version_facets: dict[str, dict[str, Any]] = {}
    for item in primary:
        version = str(item["version"])
        facets = dict(item["rating_version_facets"])
        prior = version_facets.setdefault(version, facets)
        if prior != facets:
            raise ValueError(
                "One stable rating version key maps to inconsistent seven-facet metadata: "
                + version
            )
    learner_times = sorted({int(item["learner_time"]) for item in primary})
    # A rating key represents a declared scoring occasion/wave.  IDs follow its
    # first persisted timestamp (then the key as a deterministic tie-breaker),
    # never arbitrary lexical order of an API call identifier.
    rating_time_first_seen: dict[str, str] = {}
    for item in primary:
        key = str(item["rating_time_key"])
        created_at = str(item["rating_created_at"])
        prior = rating_time_first_seen.get(key)
        if prior is None or created_at < prior:
            rating_time_first_seen[key] = created_at
    rating_times = sorted(
        rating_time_first_seen,
        key=lambda key: (rating_time_first_seen[key], key),
    )
    rating_time_map = {value: index + 1 for index, value in enumerate(rating_times)}
    dimension_map = {dimension: index + 1 for index, dimension in enumerate(DIMENSIONS)}

    stan_data: dict[str, Any] = {
        "N_obs": len(primary),
        "N_person": len(people),
        # Preserve actual event-turn spacing.  For example, anchors at turns
        # 4 and 8 keep four latent transitions between them rather than being
        # compressed into adjacent time IDs 1 and 2.
        "N_time": max(learner_times),
        "N_rating_time": len(rating_times),
        "N_dim": len(DIMENSIONS),
        "N_blueprint_dim": len(blueprint_dimension_pairs),
        "N_surface_form": len(surface_forms),
        # Stan implementations differ in their support for a zero-row matrix.
        # Keep one prior-only row when no surface form is observed; observation
        # IDs remain zero, so this row can never enter the likelihood.
        "N_surface_param": max(1, len(surface_forms)),
        "N_rater": len(raters),
        "N_version": len(versions),
        "N_state": len(STATE_FIELDS),
        "K": 5,
        "person_id": [person_map[item["person"]] for item in primary],
        "learner_time_id": [int(item["learner_time"]) for item in primary],
        "rating_time_id": [
            rating_time_map[str(item["rating_time_key"])] for item in primary
        ],
        "dimension_id": [dimension_map[item["dimension"]] for item in primary],
        "blueprint_dim_id": [
            blueprint_dimension_map[(item["blueprint"], item["dimension"])]
            for item in primary
        ],
        "blueprint_dim_dimension": [
            dimension_map[dimension]
            for _blueprint, dimension in blueprint_dimension_pairs
        ],
        "surface_form_id": [
            surface_map[item["form"]] if item["form_type"] == "surface" else 0
            for item in primary
        ],
        "rater_id": [rater_map[item["rater"]] for item in primary],
        "version_id": [version_map[item["version"]] for item in primary],
        "state_z": [item["state_z"] for item in primary],
        "score": [item["score"] for item in primary],
    }

    response_ratings: dict[str, set[tuple[str, str]]] = {}
    for item in primary:
        response_ratings.setdefault(item["response_id"], set()).add(
            (str(item["rater"]), str(item["version"]))
        )
    cross_rated = sum(len(configurations) > 1 for configurations in response_ratings.values())

    rater_version_edges = [
        (
            ("response", str(item["response_id"])),
            ("rater_version", f"{item['rater']}::{item['version']}"),
        )
        for item in primary
    ]
    rater_version_components = _bipartite_components(rater_version_edges)
    rater_version_pairs = sorted(
        {f"{item['rater']}::{item['version']}" for item in primary}
    )

    blueprint_learners: dict[str, set[str]] = {}
    exact_surface: dict[str, dict[str, set[str] | set[tuple[str, str]]]] = {}
    state_form_coverage: dict[str, dict[str, dict[str, set[str] | set[tuple[str, str]]]]] = {}
    for item in primary:
        blueprint = str(item["blueprint"])
        learner = str(item["learner"])
        response = str(item["response_id"])
        form_type = str(item["form_type"])
        form = str(item["form"])
        variant = str(item["state_variant_id"])
        blueprint_learners.setdefault(blueprint, set()).add(learner)
        form_summary = exact_surface.setdefault(
            blueprint,
            {
                "exact_learners": set(),
                "surface_learners": set(),
                "exact_responses": set(),
                "surface_responses": set(),
            },
        )
        if form_type == "exact":
            form_summary["exact_learners"].add(learner)
            form_summary["exact_responses"].add(response)
        elif form_type == "surface":
            form_summary["surface_learners"].add(learner)
            form_summary["surface_responses"].add(response)
        cell = state_form_coverage.setdefault(blueprint, {}).setdefault(
            variant, {}
        ).setdefault(form, {"learners": set(), "responses": set()})
        cell["learners"].add(learner)
        cell["responses"].add(response)

    exact_surface_report = {
        blueprint: {
            "exact_learner_count": len(summary["exact_learners"]),
            "surface_learner_count": len(summary["surface_learners"]),
            "learner_crossover_count": len(
                summary["exact_learners"].intersection(summary["surface_learners"])
            ),
            "exact_response_count": len(summary["exact_responses"]),
            "surface_response_count": len(summary["surface_responses"]),
        }
        for blueprint, summary in sorted(exact_surface.items())
    }
    state_form_report = {
        blueprint: {
            variant: {
                form: {
                    "learner_count": len(cell["learners"]),
                    "response_count": len(cell["responses"]),
                }
                for form, cell in sorted(forms.items())
            }
            for variant, forms in sorted(variants.items())
        }
        for blueprint, variants in sorted(state_form_coverage.items())
    }
    state_rank = _matrix_rank([list(item["state_z"]) for item in primary])
    state_rank_with_intercept = _matrix_rank(
        [[1.0, *list(item["state_z"])] for item in primary]
    )
    state_increment_by_dimension: dict[str, dict[str, Any]] = {}
    insufficient_state_increment_dimensions: list[str] = []
    for dimension in DIMENSIONS:
        dimension_items = [item for item in primary if item["dimension"] == dimension]
        blueprint_cells = sorted(
            {
                f"{item['library_version']}::{item['blueprint']}"
                for item in dimension_items
            }
        )
        blueprint_dummy_rows = [
            [
                1.0
                if f"{item['library_version']}::{item['blueprint']}" == blueprint
                else 0.0
                for blueprint in blueprint_cells
            ]
            for item in dimension_items
        ]
        blueprint_rank = _matrix_rank(blueprint_dummy_rows)
        augmented_rank = _matrix_rank(
            [
                [*dummy_row, *list(item["state_z"])]
                for dummy_row, item in zip(blueprint_dummy_rows, dimension_items)
            ]
        )
        incremental_rank = augmented_rank - blueprint_rank
        has_full_state_variation = incremental_rank >= len(STATE_FIELDS)
        state_increment_by_dimension[dimension] = {
            "observation_count": len(dimension_items),
            "blueprint_dummy_column_count": len(blueprint_cells),
            "blueprint_dummy_rank": blueprint_rank,
            "blueprint_plus_state_rank": augmented_rank,
            "state_z_incremental_rank_after_blueprint_dummies": incremental_rank,
            "state_z_column_count": len(STATE_FIELDS),
            "full_six_column_increment_observed": has_full_state_variation,
            "warning": (
                None
                if has_full_state_variation
                else "State covariates add fewer than six linearly independent columns after "
                "blueprint dummies in this export."
            ),
        }
        if not has_full_state_variation:
            insufficient_state_increment_dimensions.append(dimension)

    identification_warnings: list[str] = []
    if len(raters) < 2:
        identification_warnings.append(
            "Only one rater is present; rater severity and drift are fixed by identification constraints."
        )
    if len(versions) > 1 and cross_rated == 0:
        identification_warnings.append(
            "Multiple versions are present without cross-rated responses; version effects may be time-confounded."
        )
    if len(rater_version_components) > 1:
        identification_warnings.append(
            "The response-to-rater/version network has multiple disconnected components; "
            "rater/version contrasts are not fully linked."
        )
    if len(rater_version_pairs) > 1 and cross_rated == 0:
        identification_warnings.append(
            "More than one rater/version configuration is present but no response is cross-rated "
            "across configurations."
        )
    if insufficient_state_increment_dimensions:
        identification_warnings.append(
            "After absorbing blueprint dummies, state_z has fewer than six incremental columns "
            "for dimensions: " + ", ".join(insufficient_state_increment_dimensions)
        )

    manifest = {
        "created_at": utc_now(),
        "model_source": "analysis/longitudinal_multifacet_pcm.stan",
        "score_interpretation": "Uncalibrated ECD evidence labels; not ability estimates.",
        "include_training_only": include_training_only,
        "eligibility_rule": "Stan data always contains exact_anchor or surface_anchor only",
        "primary_observation_count": len(primary),
        "exploratory_training_observation_count": len(exploratory),
        "cross_rated_response_count": cross_rated,
        "library_version": library_versions[0],
        "support_diagnostics": {
            "rule": "Per dimension, learner ↔ (library_version, blueprint) must have one connected component.",
            "by_dimension": support_by_dimension,
            "status": "connected",
        },
        "blueprint_unique_learner_count": {
            blueprint: len(learners)
            for blueprint, learners in sorted(blueprint_learners.items())
        },
        "exact_surface_crossing": exact_surface_report,
        "state_variant_form_coverage": state_form_report,
        "response_rater_version_network": {
            "component_count": len(rater_version_components),
            "components": _component_summary(
                rater_version_components,
                left_kind="response",
                right_kind="rater_version",
            ),
            "rater_version_configuration_count": len(rater_version_pairs),
            "cross_rated_response_count": cross_rated,
            "cross_rating_status": (
                "not_required_single_configuration"
                if len(rater_version_pairs) == 1
                else "present"
                if cross_rated
                else "insufficient_for_configuration_linking"
            ),
        },
        "state_design_diagnostic": {
            "state_fields": list(STATE_FIELDS),
            "scaling": {
                "cash": "(cash - 100000) / 100000",
                "other_state_indices": "(value - 50) / 50",
            },
            "state_z_matrix_rank": state_rank,
            "state_z_with_intercept_matrix_rank": state_rank_with_intercept,
            "by_dimension_after_blueprint_dummies": state_increment_by_dimension,
            "row_count": len(primary),
            "note": "Rank is a deployment diagnostic, not evidence of parameter identification or validity.",
        },
        "mappings": {
            "person": people,
            "person_interpretation": "Pseudonymous simulation trajectory; separate simulations are not merged.",
            "learner_time": learner_times,
            "rating_time": [
                {
                    "rating_time_id": rating_time_map[key],
                    "rating_time_key": key,
                    "first_recorded_at": rating_time_first_seen[key],
                }
                for key in rating_times
            ],
            "dimension": list(DIMENSIONS),
            "blueprint": blueprints,
            "blueprint_dimension": [
                {"blueprint": blueprint, "dimension": dimension}
                for blueprint, dimension in blueprint_dimension_pairs
            ],
            "surface_form": surface_forms,
            "exact_form_reference": 0,
            "surface_parameter_rows": max(1, len(surface_forms)),
            "rater": raters,
            "version": versions,
            "rating_version_facets": version_facets,
        },
        "state_modeling": (
            "Visible controlled sandbox state is exported as fixed, interpretable state_z covariates; "
            "the manifest reports design rank rather than asserting that state effects are identified."
        ),
        "skipped_incomplete_records": skipped,
        "identification_warnings": identification_warnings,
        "warnings": [
            "Export does not establish anchor equivalence, common support, reliability or validity.",
            "Do not fit or interpret longitudinal scores until the pre-specified validation gates are met.",
            "Selection probabilities are conditional on the recorded scheduler candidate set and are not unconditional propensities.",
            "Training-only main-path records never enter Stan data, even when exported for exploration.",
            "Mixed blueprint-library versions and disconnected learner-to-blueprint support are rejected rather than pooled.",
        ],
    }
    return stan_data, manifest, observations
