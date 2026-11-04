"""Pre-compute auditable, queue-balanced common-anchor allocation plans.

This module is an implementation utility, not an experiment runner.  It takes
already-created anonymous allocation-unit identifiers and produces a frozen
JSON-compatible plan.  It never reads learner profiles, simulation states,
responses, outcomes, or measurement estimates.

The blueprint layout uses a cyclic incomplete-block construction.  With
``B`` anchor blueprints and ``K <= B`` checkpoints, each unit receives ``K``
different blueprints and the queue-wide blueprint counts differ by at most
one.  For the current (small) anchor library, every blueprint permutation is
examined and layouts are ranked by cross-unit ECD-dimension overlap.  Frozen
form kinds, concrete forms, and sandbox-state variants are then assigned with
integer quotas whose counts differ by at most one wherever that is
mathematically possible.

All choices are deterministic for the same library, unit identifiers,
checkpoint count, and seed.  The seed itself is deliberately omitted from the
result; only a one-way fingerprint is retained.  Reported probabilities are
conditional design probabilities under the documented constrained plan, not
independent draws and not probabilities of learner performance or outcomes.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .domain import BlueprintLibrary, ECD_DIMENSIONS, EventBlueprint


ALGORITHM_VERSION = "queue-balanced-common-anchor-v1"
PLAN_SCHEMA_VERSION = "1.0"
MAX_EXHAUSTIVE_BLUEPRINTS = 8
_UNIT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class AllocationPlanError(ValueError):
    """Raised when a safe plan cannot be created from the supplied inputs."""


class AllocationPlanIntegrityError(ValueError):
    """Raised when a generated plan or its assignment hash chain was changed."""


@dataclass(frozen=True, slots=True)
class _Slot:
    unit_id: str
    allocation_rank: int
    checkpoint_index: int
    blueprint_id: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.allocation_rank, self.checkpoint_index)

    @property
    def identity(self) -> str:
        return f"{self.unit_id}:{self.checkpoint_index}"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seed_digest(seed: str, *parts: object) -> str:
    material = "\x1f".join(
        str(part) for part in (ALGORITHM_VERSION, seed, *parts)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _probability(numerator: int, denominator: int, *, scope: str) -> dict[str, Any]:
    if denominator < 1 or numerator < 0 or numerator > denominator:
        raise AllocationPlanError("invalid conditional probability components")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
        "scope": scope,
    }


def _validate_inputs(
    allocation_unit_ids: Sequence[str],
    anchor_checkpoints: int,
    seed: str,
    *,
    anchor_count: int,
) -> tuple[str, ...]:
    if isinstance(allocation_unit_ids, (str, bytes)) or not isinstance(
        allocation_unit_ids, Sequence
    ):
        raise AllocationPlanError("allocation_unit_ids must be a sequence of anonymous IDs")
    unit_ids = tuple(allocation_unit_ids)
    if len(unit_ids) < 2:
        raise AllocationPlanError(
            "at least two allocation units are required for cross-unit dimension overlap"
        )
    for unit_id in unit_ids:
        if not isinstance(unit_id, str) or not _UNIT_ID_PATTERN.fullmatch(unit_id):
            raise AllocationPlanError(
                "allocation unit IDs must be 1-128 character ASCII pseudonymous tokens "
                "using letters, digits, '.', '_', ':' or '-'"
            )
    if len(unit_ids) != len(set(unit_ids)):
        raise AllocationPlanError("allocation unit IDs must be unique")
    if (
        isinstance(anchor_checkpoints, bool)
        or not isinstance(anchor_checkpoints, int)
        or anchor_checkpoints < 1
    ):
        raise AllocationPlanError("anchor_checkpoints must be a positive integer")
    if anchor_checkpoints > anchor_count:
        raise AllocationPlanError(
            f"anchor_checkpoints={anchor_checkpoints} exceeds the {anchor_count} available "
            "anchor blueprints; non-repetition within a unit would be impossible"
        )
    if not isinstance(seed, str) or not seed.strip():
        raise AllocationPlanError("seed must be a non-empty string")
    if len(seed) > 1024:
        raise AllocationPlanError("seed is unexpectedly long")
    return unit_ids


def _anchor_manifest(anchors: Sequence[EventBlueprint]) -> list[dict[str, Any]]:
    return [
        {
            "blueprint_id": blueprint.id,
            "family_id": blueprint.family_id,
            "kind": blueprint.kind,
            "title": blueprint.title,
            "target_dimensions": list(blueprint.target_dimensions),
            "difficulty_level": blueprint.difficulty_level,
            "tags": list(blueprint.tags),
            "preconditions": {
                field_name: {"min": bounds[0], "max": bounds[1]}
                for field_name, bounds in sorted(blueprint.preconditions.items())
            },
            "evidence_requirements": dict(sorted(blueprint.evidence_requirements.items())),
            "forms": [
                form.to_dict()
                for form in sorted(blueprint.forms, key=lambda item: item.id)
            ],
            "state_variants": [
                {"state_variant_id": item.id, "state": item.state.to_dict()}
                for item in sorted(blueprint.sandbox_states, key=lambda item: item.id)
            ],
            "actions": [
                {
                    "action_code": action.action_code,
                    "label": action.label,
                    "description": action.description,
                    "delta": dict(sorted(action.delta.items())),
                    "feedback_cue": action.feedback_cue,
                }
                for action in sorted(blueprint.actions, key=lambda item: item.action_code)
            ],
        }
        for blueprint in sorted(anchors, key=lambda item: item.id)
    ]


def _layout_for_permutation(
    permutation: Sequence[str], unit_count: int, checkpoints: int
) -> tuple[tuple[str, ...], ...]:
    blueprint_count = len(permutation)
    return tuple(
        tuple(
            permutation[(unit_rank * checkpoints + checkpoint) % blueprint_count]
            for checkpoint in range(checkpoints)
        )
        for unit_rank in range(unit_count)
    )


def _dimension_counts(
    layout: Sequence[Sequence[str]], by_id: Mapping[str, EventBlueprint]
) -> tuple[dict[str, int], dict[str, int]]:
    unit_overlap: dict[str, int] = {}
    assignment_exposure: dict[str, int] = {}
    for dimension in ECD_DIMENSIONS:
        unit_overlap[dimension] = sum(
            any(dimension in by_id[blueprint_id].target_dimensions for blueprint_id in row)
            for row in layout
        )
        assignment_exposure[dimension] = sum(
            dimension in by_id[blueprint_id].target_dimensions
            for row in layout
            for blueprint_id in row
        )
    return unit_overlap, assignment_exposure


def _dispersion(values: Iterable[int]) -> int:
    values_tuple = tuple(values)
    total = sum(values_tuple)
    count = len(values_tuple)
    # Integer-scaled squared deviations avoid floating-point ranking changes.
    return sum((value * count - total) ** 2 for value in values_tuple)


def _layout_score(
    layout: Sequence[Sequence[str]], by_id: Mapping[str, EventBlueprint]
) -> tuple[int, ...]:
    unit_overlap, exposure = _dimension_counts(layout, by_id)
    unit_values = tuple(unit_overlap[dimension] for dimension in ECD_DIMENSIONS)
    exposure_values = tuple(exposure[dimension] for dimension in ECD_DIMENSIONS)
    return (
        -min(unit_values),
        -sum(unit_values),
        max(unit_values) - min(unit_values),
        _dispersion(unit_values),
        -min(exposure_values),
        -sum(exposure_values),
        max(exposure_values) - min(exposure_values),
        _dispersion(exposure_values),
    )


def _select_layout(
    anchors: Sequence[EventBlueprint],
    *,
    unit_count: int,
    checkpoints: int,
    seed: str,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[str, ...], ...], ...],
    tuple[int, ...],
]:
    blueprint_ids = tuple(sorted(blueprint.id for blueprint in anchors))
    if len(blueprint_ids) > MAX_EXHAUSTIVE_BLUEPRINTS:
        raise AllocationPlanError(
            f"{ALGORITHM_VERSION} exhaustively supports at most "
            f"{MAX_EXHAUSTIVE_BLUEPRINTS} anchor blueprints; found {len(blueprint_ids)}"
        )
    by_id = {blueprint.id: blueprint for blueprint in anchors}
    unique_layouts: dict[tuple[tuple[str, ...], ...], tuple[int, ...]] = {}
    for permutation in itertools.permutations(blueprint_ids):
        layout = _layout_for_permutation(permutation, unit_count, checkpoints)
        unique_layouts.setdefault(layout, _layout_score(layout, by_id))
    if not unique_layouts:  # guarded by the validated library
        raise AllocationPlanError("the anchor library produced no candidate layouts")
    best_score = min(unique_layouts.values())
    best_layouts = tuple(
        sorted(layout for layout, score in unique_layouts.items() if score == best_score)
    )
    selected = min(
        best_layouts,
        key=lambda layout: _seed_digest(seed, "layout-tie", _object_hash(layout)),
    )
    unit_overlap, _ = _dimension_counts(selected, by_id)
    if min(unit_overlap.values()) < 2:
        raise AllocationPlanError(
            "the requested queue/checkpoint structure cannot give every ECD dimension "
            "to at least two distinct allocation units under strict blueprint balance; "
            "increase the number of units or anchor checkpoints"
        )
    return selected, best_layouts, best_score


def _balanced_token_assignment(
    occurrences: Sequence[_Slot],
    token_ids: Sequence[str],
    *,
    seed: str,
    domain: str,
) -> tuple[dict[tuple[int, int], str], dict[str, int]]:
    """Assign tokens with marginal count gap <= 1 using independent hash orders."""

    if not occurrences:
        return {}, {token_id: 0 for token_id in token_ids}
    if not token_ids:
        raise AllocationPlanError(f"no tokens available for {domain}")
    ordered_tokens = sorted(
        token_ids, key=lambda token_id: (_seed_digest(seed, domain, "token", token_id), token_id)
    )
    base, remainder = divmod(len(occurrences), len(ordered_tokens))
    quotas = {
        token_id: base + (1 if index < remainder else 0)
        for index, token_id in enumerate(ordered_tokens)
    }
    token_copies = [
        (token_id, copy_index)
        for token_id in ordered_tokens
        for copy_index in range(quotas[token_id])
    ]
    token_copies.sort(
        key=lambda item: (
            _seed_digest(seed, domain, "copy", item[0], item[1]),
            item,
        )
    )
    ordered_occurrences = sorted(
        occurrences,
        key=lambda slot: (
            _seed_digest(seed, domain, "occurrence", slot.identity),
            slot.key,
        ),
    )
    assignments = {
        occurrence.key: token_copy[0]
        for occurrence, token_copy in zip(ordered_occurrences, token_copies, strict=True)
    }
    return assignments, quotas


def _assign_form_kinds(
    occurrences_by_blueprint: Mapping[str, Sequence[_Slot]],
    *,
    seed: str,
) -> tuple[dict[tuple[int, int], str], dict[str, dict[str, int]]]:
    total = sum(len(items) for items in occurrences_by_blueprint.values())
    target_exact = total // 2
    if total % 2 and int(_seed_digest(seed, "global-form-kind-target")[:2], 16) % 2:
        target_exact += 1
    base_exact = sum(len(items) // 2 for items in occurrences_by_blueprint.values())
    extra_exact_needed = target_exact - base_exact
    odd_blueprints = [
        blueprint_id
        for blueprint_id, items in occurrences_by_blueprint.items()
        if len(items) % 2
    ]
    odd_blueprints.sort(
        key=lambda blueprint_id: (
            _seed_digest(seed, "form-kind-extra", blueprint_id),
            blueprint_id,
        )
    )
    extra_exact = set(odd_blueprints[:extra_exact_needed])

    assignment: dict[tuple[int, int], str] = {}
    quotas: dict[str, dict[str, int]] = {}
    for blueprint_id in sorted(occurrences_by_blueprint):
        occurrences = occurrences_by_blueprint[blueprint_id]
        exact_count = len(occurrences) // 2 + (blueprint_id in extra_exact)
        ordered = sorted(
            occurrences,
            key=lambda slot: (
                _seed_digest(seed, "form-kind-occurrence", blueprint_id, slot.identity),
                slot.key,
            ),
        )
        exact_keys = {slot.key for slot in ordered[:exact_count]}
        for slot in occurrences:
            assignment[slot.key] = "exact" if slot.key in exact_keys else "surface"
        quotas[blueprint_id] = {
            "exact": exact_count,
            "surface": len(occurrences) - exact_count,
        }
    return assignment, quotas


def _count_gap(values: Iterable[int]) -> int:
    values_tuple = tuple(values)
    return max(values_tuple) - min(values_tuple) if values_tuple else 0


def generate_anchor_plan(
    allocation_unit_ids: Sequence[str],
    anchor_checkpoints: int,
    seed: str,
    *,
    library: BlueprintLibrary | None = None,
) -> dict[str, Any]:
    """Create a deterministic, JSON-compatible queue-level anchor plan.

    The caller is responsible for supplying identifiers that were already
    pseudonymized outside this function.  The generator cannot determine
    whether a syntactically valid token embeds a real-world identity.
    """

    library = library or BlueprintLibrary.load_default()
    anchors = tuple(sorted(library.anchor_blueprints, key=lambda item: item.id))
    unit_ids = _validate_inputs(
        allocation_unit_ids,
        anchor_checkpoints,
        seed,
        anchor_count=len(anchors),
    )
    ordered_units = tuple(
        sorted(
            unit_ids,
            key=lambda unit_id: (
                _seed_digest(seed, "unit-order", unit_id),
                unit_id,
            ),
        )
    )
    layout, best_layouts, layout_score = _select_layout(
        anchors,
        unit_count=len(ordered_units),
        checkpoints=anchor_checkpoints,
        seed=seed,
    )
    by_id = {blueprint.id: blueprint for blueprint in anchors}
    slots = tuple(
        _Slot(
            unit_id=unit_id,
            allocation_rank=rank,
            checkpoint_index=checkpoint_index + 1,
            blueprint_id=layout[rank - 1][checkpoint_index],
        )
        for rank, unit_id in enumerate(ordered_units, start=1)
        for checkpoint_index in range(anchor_checkpoints)
    )
    occurrences_by_blueprint: dict[str, list[_Slot]] = defaultdict(list)
    for slot in slots:
        occurrences_by_blueprint[slot.blueprint_id].append(slot)

    kind_by_slot, form_kind_quotas = _assign_form_kinds(
        occurrences_by_blueprint, seed=seed
    )
    form_by_slot: dict[tuple[int, int], str] = {}
    form_quotas: dict[str, dict[str, dict[str, int]]] = {}
    for blueprint_id in sorted(occurrences_by_blueprint):
        blueprint = by_id[blueprint_id]
        form_quotas[blueprint_id] = {}
        for form_kind in ("exact", "surface"):
            kind_occurrences = [
                slot
                for slot in occurrences_by_blueprint[blueprint_id]
                if kind_by_slot[slot.key] == form_kind
            ]
            form_ids = tuple(
                sorted(form.id for form in blueprint.forms if form.form_kind == form_kind)
            )
            assigned, quotas = _balanced_token_assignment(
                kind_occurrences,
                form_ids,
                seed=seed,
                domain=f"form:{blueprint_id}:{form_kind}",
            )
            form_by_slot.update(assigned)
            form_quotas[blueprint_id][form_kind] = quotas

    state_by_slot: dict[tuple[int, int], str] = {}
    state_quotas: dict[str, dict[str, int]] = {}
    for blueprint_id in sorted(occurrences_by_blueprint):
        state_ids = tuple(
            sorted(item.id for item in by_id[blueprint_id].sandbox_states)
        )
        assigned, quotas = _balanced_token_assignment(
            occurrences_by_blueprint[blueprint_id],
            state_ids,
            seed=seed,
            domain=f"state:{blueprint_id}",
        )
        state_by_slot.update(assigned)
        state_quotas[blueprint_id] = quotas

    best_layout_count = len(best_layouts)
    blueprint_probability_by_slot: dict[tuple[int, int], tuple[int, int]] = {}
    for rank in range(1, len(ordered_units) + 1):
        for checkpoint_index in range(1, anchor_checkpoints + 1):
            selected_blueprint = layout[rank - 1][checkpoint_index - 1]
            numerator = sum(
                candidate[rank - 1][checkpoint_index - 1] == selected_blueprint
                for candidate in best_layouts
            )
            blueprint_probability_by_slot[(rank, checkpoint_index)] = (
                numerator,
                best_layout_count,
            )

    previous_hash = "0" * 64
    assignments_by_rank: dict[int, dict[str, Any]] = {}
    for rank, unit_id in enumerate(ordered_units, start=1):
        assignments_by_rank[rank] = {
            "allocation_unit_id": unit_id,
            "allocation_rank": rank,
            "unit_order_fingerprint": _seed_digest(
                seed, "unit-order", unit_id
            )[:16],
            "checkpoints": [],
        }
    for slot in slots:
        blueprint = by_id[slot.blueprint_id]
        form_kind = kind_by_slot[slot.key]
        form_id = form_by_slot[slot.key]
        state_variant_id = state_by_slot[slot.key]
        blueprint_numerator, blueprint_denominator = blueprint_probability_by_slot[
            slot.key
        ]
        blueprint_occurrences = len(occurrences_by_blueprint[slot.blueprint_id])
        kind_count = form_kind_quotas[slot.blueprint_id][form_kind]
        checkpoint: dict[str, Any] = {
            "checkpoint_index": slot.checkpoint_index,
            "blueprint_id": slot.blueprint_id,
            "family_id": blueprint.family_id,
            "target_dimensions": list(blueprint.target_dimensions),
            "difficulty_level": blueprint.difficulty_level,
            "form_kind": form_kind,
            "form_id": form_id,
            "state_variant_id": state_variant_id,
            "conditional_probabilities": {
                "blueprint": _probability(
                    blueprint_numerator,
                    blueprint_denominator,
                    scope=(
                        "conditional on the realized seeded unit order and the set of "
                        "equally optimal strict-balance cyclic layouts"
                    ),
                ),
                "form_kind": _probability(
                    kind_count,
                    blueprint_occurrences,
                    scope=(
                        "conditional on this blueprint's queue-wide exact/surface quota"
                    ),
                ),
                "form_id_given_kind": _probability(
                    form_quotas[slot.blueprint_id][form_kind][form_id],
                    kind_count,
                    scope=(
                        "conditional on blueprint, form kind and the balanced concrete-form quota"
                    ),
                ),
                "state_variant_given_blueprint": _probability(
                    state_quotas[slot.blueprint_id][state_variant_id],
                    blueprint_occurrences,
                    scope=(
                        "conditional on blueprint and the balanced sandbox-state quota"
                    ),
                ),
            },
            "previous_assignment_hash": previous_hash,
        }
        checkpoint["assignment_hash"] = _object_hash(checkpoint)
        previous_hash = checkpoint["assignment_hash"]
        assignments_by_rank[slot.allocation_rank]["checkpoints"].append(checkpoint)

    blueprint_counts = {
        blueprint.id: len(occurrences_by_blueprint.get(blueprint.id, ()))
        for blueprint in anchors
    }
    global_form_kind_counts = dict(Counter(kind_by_slot.values()))
    global_form_kind_counts = {
        kind: global_form_kind_counts.get(kind, 0) for kind in ("exact", "surface")
    }
    unit_overlap, assignment_exposure = _dimension_counts(layout, by_id)
    form_id_counts = {
        blueprint_id: {
            form_kind: dict(sorted(quotas.items()))
            for form_kind, quotas in form_quotas[blueprint_id].items()
        }
        for blueprint_id in sorted(form_quotas)
    }
    state_variant_counts = {
        blueprint_id: dict(sorted(quotas.items()))
        for blueprint_id, quotas in sorted(state_quotas.items())
    }
    balance_summary = {
        "blueprint_counts": blueprint_counts,
        "blueprint_max_min_gap": _count_gap(blueprint_counts.values()),
        "form_kind_counts_global": global_form_kind_counts,
        "form_kind_global_max_min_gap": _count_gap(
            global_form_kind_counts.values()
        ),
        "form_kind_counts_by_blueprint": {
            blueprint_id: dict(counts)
            for blueprint_id, counts in sorted(form_kind_quotas.items())
        },
        "form_id_counts_by_blueprint": form_id_counts,
        "state_variant_counts_by_blueprint": state_variant_counts,
        "dimension_distinct_unit_counts": unit_overlap,
        "dimension_assignment_exposure_counts": assignment_exposure,
        "minimum_dimension_distinct_unit_count": min(unit_overlap.values()),
        "layout_score": list(layout_score),
        "equally_optimal_layout_count": best_layout_count,
    }
    manifest = _anchor_manifest(anchors)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_kind": "precomputed_common_anchor_allocation",
        "algorithm_version": ALGORITHM_VERSION,
        "library_version": library.library_version,
        "anchor_library_hash": _object_hash(manifest),
        "seed_fingerprint": _seed_digest(seed, "seed-fingerprint")[:24],
        "raw_seed_included": False,
        "input": {
            "allocation_unit_count": len(ordered_units),
            "anchor_checkpoints_per_unit": anchor_checkpoints,
            "allocation_unit_ids_hash": _object_hash(sorted(unit_ids)),
            "unit_order_method": "SHA-256 seed-indexed pseudonymous-ID order",
            "input_order_ignored": True,
        },
        "design_probability_note": (
            "Component probabilities are conditional design probabilities under "
            "queue-wide quotas. Assignments are coupled, so components must not be "
            "multiplied as if they were independent; they are not outcome probabilities."
        ),
        "guarantees": {
            "blueprints_do_not_repeat_within_unit": True,
            "blueprint_count_gap_at_most_one": True,
            "global_exact_surface_count_gap_at_most_one": True,
            "within_blueprint_exact_surface_count_gap_at_most_one": True,
            "within_kind_concrete_form_count_gap_at_most_one": True,
            "within_blueprint_state_variant_count_gap_at_most_one": True,
            "each_dimension_shared_by_at_least_two_units": True,
            "profile_or_outcome_data_used": False,
        },
        "balance_summary": balance_summary,
        "assignments": [
            assignments_by_rank[rank] for rank in sorted(assignments_by_rank)
        ],
        "assignment_chain_root": previous_hash,
        "hash_algorithm": "SHA-256 over canonical UTF-8 JSON",
        "hash_limit": (
            "The embedded hashes are not digital signatures. Detecting intentional "
            "replacement requires comparing plan_hash with a trusted separately "
            "registered value."
        ),
    }
    plan["plan_hash"] = _object_hash(plan)
    return plan


def verify_anchor_plan_integrity(plan: Mapping[str, Any]) -> None:
    """Verify the top-level hash and ordered assignment hash chain.

    This check detects corruption relative to the embedded hash.  Because the
    hash is not a digital signature, an actor who can replace both content and
    hash can evade it; production registration should retain a trusted
    ``plan_hash`` separately.  The check does not validate scientific
    effectiveness, psychometric fit, or fairness.
    """

    if not isinstance(plan, Mapping):
        raise AllocationPlanIntegrityError("plan must be a mapping")
    supplied_plan_hash = plan.get("plan_hash")
    if not isinstance(supplied_plan_hash, str):
        raise AllocationPlanIntegrityError("plan_hash is missing")
    unhashed = dict(plan)
    unhashed.pop("plan_hash", None)
    if _object_hash(unhashed) != supplied_plan_hash:
        raise AllocationPlanIntegrityError("plan_hash does not match plan content")

    previous_hash = "0" * 64
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise AllocationPlanIntegrityError("assignments must be a list")
    for unit in assignments:
        if not isinstance(unit, Mapping) or not isinstance(unit.get("checkpoints"), list):
            raise AllocationPlanIntegrityError("invalid assignment unit structure")
        for checkpoint in unit["checkpoints"]:
            if not isinstance(checkpoint, Mapping):
                raise AllocationPlanIntegrityError("invalid checkpoint structure")
            if checkpoint.get("previous_assignment_hash") != previous_hash:
                raise AllocationPlanIntegrityError("assignment hash chain is discontinuous")
            supplied_assignment_hash = checkpoint.get("assignment_hash")
            if not isinstance(supplied_assignment_hash, str):
                raise AllocationPlanIntegrityError("assignment_hash is missing")
            unhashed_checkpoint = dict(checkpoint)
            unhashed_checkpoint.pop("assignment_hash", None)
            expected = _object_hash(unhashed_checkpoint)
            if expected != supplied_assignment_hash:
                raise AllocationPlanIntegrityError("assignment_hash does not match checkpoint")
            previous_hash = supplied_assignment_hash
    if plan.get("assignment_chain_root") != previous_hash:
        raise AllocationPlanIntegrityError("assignment_chain_root does not match the chain")


class AnchorAllocationPlan:
    """Validated read-only-by-copy lookup interface for a frozen JSON plan.

    Constructing this object verifies both hash layers and checks every
    blueprint/form/state reference against the supplied versioned library.
    ``assignment_for`` returns a deep copy, so a caller cannot mutate the
    in-memory registered plan accidentally.
    """

    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        library: BlueprintLibrary | None = None,
        expected_plan_hash: str | None = None,
    ) -> None:
        verify_anchor_plan_integrity(plan)
        if expected_plan_hash is not None:
            if not isinstance(expected_plan_hash, str) or not hmac.compare_digest(
                str(plan.get("plan_hash", "")), expected_plan_hash
            ):
                raise AllocationPlanIntegrityError(
                    "plan_hash does not match the separately registered expected hash"
                )
        if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise AllocationPlanIntegrityError("unsupported allocation plan schema_version")
        if plan.get("algorithm_version") != ALGORITHM_VERSION:
            raise AllocationPlanIntegrityError("unsupported allocation algorithm_version")
        if plan.get("plan_kind") != "precomputed_common_anchor_allocation":
            raise AllocationPlanIntegrityError("unexpected allocation plan kind")

        library = library or BlueprintLibrary.load_default()
        anchors = tuple(sorted(library.anchor_blueprints, key=lambda item: item.id))
        if plan.get("library_version") != library.library_version:
            raise AllocationPlanIntegrityError(
                "allocation plan library_version does not match the runtime library"
            )
        if plan.get("anchor_library_hash") != _object_hash(_anchor_manifest(anchors)):
            raise AllocationPlanIntegrityError(
                "allocation plan anchor_library_hash does not match the runtime library"
            )
        by_id = {blueprint.id: blueprint for blueprint in anchors}

        raw_input = plan.get("input")
        if not isinstance(raw_input, Mapping):
            raise AllocationPlanIntegrityError("plan input metadata is missing")
        expected_unit_count = raw_input.get("allocation_unit_count")
        expected_checkpoint_count = raw_input.get("anchor_checkpoints_per_unit")
        if (
            isinstance(expected_unit_count, bool)
            or not isinstance(expected_unit_count, int)
            or expected_unit_count < 2
        ):
            raise AllocationPlanIntegrityError("invalid allocation_unit_count")
        if (
            isinstance(expected_checkpoint_count, bool)
            or not isinstance(expected_checkpoint_count, int)
            or expected_checkpoint_count < 1
        ):
            raise AllocationPlanIntegrityError("invalid anchor_checkpoints_per_unit")

        assignments = plan.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != expected_unit_count:
            raise AllocationPlanIntegrityError(
                "assignment-unit count does not match input metadata"
            )
        index: dict[tuple[str, int], dict[str, Any]] = {}
        unit_ids: list[str] = []
        seen_ranks: set[int] = set()
        for raw_unit in assignments:
            if not isinstance(raw_unit, Mapping):
                raise AllocationPlanIntegrityError("assignment unit must be an object")
            unit_id = raw_unit.get("allocation_unit_id")
            rank = raw_unit.get("allocation_rank")
            checkpoints = raw_unit.get("checkpoints")
            if not isinstance(unit_id, str) or not _UNIT_ID_PATTERN.fullmatch(unit_id):
                raise AllocationPlanIntegrityError("invalid allocation_unit_id in plan")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise AllocationPlanIntegrityError("invalid allocation_rank in plan")
            if unit_id in unit_ids or rank in seen_ranks:
                raise AllocationPlanIntegrityError("duplicate allocation unit or rank")
            if not isinstance(checkpoints, list) or len(checkpoints) != expected_checkpoint_count:
                raise AllocationPlanIntegrityError(
                    "unit checkpoint count does not match plan metadata"
                )
            unit_ids.append(unit_id)
            seen_ranks.add(rank)
            seen_blueprints: set[str] = set()
            seen_checkpoint_indices: set[int] = set()
            for checkpoint in checkpoints:
                if not isinstance(checkpoint, Mapping):
                    raise AllocationPlanIntegrityError("checkpoint must be an object")
                checkpoint_index = checkpoint.get("checkpoint_index")
                blueprint_id = checkpoint.get("blueprint_id")
                form_id = checkpoint.get("form_id")
                form_kind = checkpoint.get("form_kind")
                state_variant_id = checkpoint.get("state_variant_id")
                if (
                    isinstance(checkpoint_index, bool)
                    or not isinstance(checkpoint_index, int)
                    or not 1 <= checkpoint_index <= expected_checkpoint_count
                    or checkpoint_index in seen_checkpoint_indices
                ):
                    raise AllocationPlanIntegrityError(
                        "invalid or duplicate checkpoint_index in unit allocation"
                    )
                seen_checkpoint_indices.add(checkpoint_index)
                if not isinstance(blueprint_id, str) or blueprint_id not in by_id:
                    raise AllocationPlanIntegrityError(
                        "plan references an unknown anchor blueprint"
                    )
                if blueprint_id in seen_blueprints:
                    raise AllocationPlanIntegrityError(
                        "plan repeats an anchor blueprint within one allocation unit"
                    )
                seen_blueprints.add(blueprint_id)
                blueprint = by_id[blueprint_id]
                try:
                    form = blueprint.get_form(str(form_id))
                except KeyError as exc:
                    raise AllocationPlanIntegrityError(
                        "plan references an unknown frozen anchor form"
                    ) from exc
                if form.form_kind != form_kind or not form.fixed:
                    raise AllocationPlanIntegrityError(
                        "plan form kind/frozen status conflicts with the runtime library"
                    )
                if state_variant_id not in {
                    variant.id for variant in blueprint.sandbox_states
                }:
                    raise AllocationPlanIntegrityError(
                        "plan references an unknown anchor state variant"
                    )
                if checkpoint.get("family_id") != blueprint.family_id:
                    raise AllocationPlanIntegrityError("plan family_id conflicts with library")
                if checkpoint.get("target_dimensions") != list(
                    blueprint.target_dimensions
                ):
                    raise AllocationPlanIntegrityError(
                        "plan target_dimensions conflict with library"
                    )
                if checkpoint.get("difficulty_level") != blueprint.difficulty_level:
                    raise AllocationPlanIntegrityError(
                        "plan difficulty_level conflicts with library"
                    )
                record = {
                    "allocation_unit_id": unit_id,
                    "allocation_rank": rank,
                    **dict(checkpoint),
                    "plan_hash": plan["plan_hash"],
                    "algorithm_version": plan["algorithm_version"],
                    "library_version": plan["library_version"],
                }
                index[(unit_id, checkpoint_index)] = record
        if seen_ranks != set(range(1, expected_unit_count + 1)):
            raise AllocationPlanIntegrityError("allocation ranks must be contiguous from 1")

        self._plan = copy.deepcopy(dict(plan))
        self._index = index
        self._unit_ids = tuple(unit_ids)
        self._checkpoint_count = expected_checkpoint_count

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        library: BlueprintLibrary | None = None,
        expected_plan_hash: str | None = None,
    ) -> "AnchorAllocationPlan":
        """Load a plan file, rejecting invalid JSON, hashes, or library references."""

        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AllocationPlanIntegrityError(
                f"cannot load anchor allocation plan {source}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise AllocationPlanIntegrityError("allocation plan JSON must be an object")
        return cls(
            raw,
            library=library,
            expected_plan_hash=expected_plan_hash,
        )

    @property
    def plan_hash(self) -> str:
        return str(self._plan["plan_hash"])

    @property
    def allocation_unit_ids(self) -> tuple[str, ...]:
        return self._unit_ids

    @property
    def checkpoint_count(self) -> int:
        return self._checkpoint_count

    def assignment_for(
        self, allocation_unit_id: str, checkpoint_index: int
    ) -> dict[str, Any]:
        """Return one validated frozen assignment for an online scheduler to consume."""

        if not isinstance(allocation_unit_id, str):
            raise KeyError("allocation_unit_id is not present in this plan")
        if isinstance(checkpoint_index, bool) or not isinstance(checkpoint_index, int):
            raise KeyError("checkpoint_index is not present in this plan")
        try:
            record = self._index[(allocation_unit_id, checkpoint_index)]
        except KeyError as exc:
            raise KeyError(
                f"no anchor assignment for unit {allocation_unit_id!r} at "
                f"checkpoint {checkpoint_index}"
            ) from exc
        return copy.deepcopy(record)


__all__ = [
    "ALGORITHM_VERSION",
    "AnchorAllocationPlan",
    "AllocationPlanError",
    "AllocationPlanIntegrityError",
    "generate_anchor_plan",
    "verify_anchor_plan_integrity",
]
