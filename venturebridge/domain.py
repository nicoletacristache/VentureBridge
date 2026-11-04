"""Auditable domain rules for the VentureBridge entrepreneurship simulation.

This module deliberately contains no LLM calls.  A language model may turn a
blueprint into prose and may classify free text into one of the *offered*
``action_code`` values, but only this module is allowed to mutate numeric
simulation state.  In particular, callers cannot supply an arbitrary delta to
``apply_classified_action``.

The deterministic scheduler is pseudo-random in the allocation sense: a
stable SHA-256 draw selects among the currently eligible blueprints and forms.
The same inputs always produce the same event, which makes a delivery
reproducible and auditable without relying on Python's process-randomized
``hash`` implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


STATE_FIELDS = (
    "cash",
    "team",
    "product",
    "market_evidence",
    "reputation",
    "compliance",
)

STATE_BOUNDS: Mapping[str, tuple[int, int]] = {
    "cash": (0, 1_000_000),
    "team": (0, 100),
    "product": (0, 100),
    "market_evidence": (0, 100),
    "reputation": (0, 100),
    "compliance": (0, 100),
}

ECD_DIMENSIONS = (
    "opportunity_framing",
    "evidence_falsifiability",
    "resource_finance",
    "adaptive_action",
    "stakeholder_ethics",
)

EVENT_KINDS = ("main", "exact_anchor", "surface_anchor")
FORM_KINDS = ("main", "exact", "surface")


class BlueprintValidationError(ValueError):
    """Raised when an event library violates the frozen blueprint contract."""


class SchedulingError(ValueError):
    """Raised when an event cannot be scheduled from the available library."""


class AnchorPlanLookup(Protocol):
    """Minimal interface supplied by a validated frozen allocation plan.

    Keeping the protocol in the domain layer avoids importing the concrete
    planner (which itself validates domain blueprints) and therefore avoids a
    circular dependency.
    """

    @property
    def checkpoint_count(self) -> int: ...

    @property
    def plan_hash(self) -> str: ...

    def assignment_for(
        self, allocation_unit_id: str, checkpoint_index: int
    ) -> dict[str, Any]: ...


class StateTransitionError(ValueError):
    """Raised when a classified action cannot safely be applied."""


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class EntrepreneurshipState:
    """The complete numeric state controlled by the external rule engine.

    ``cash`` is represented in whole teaching-simulation currency units and is
    bounded to one million.  The other dimensions are bounded teaching indices
    from 0 to 100; they are not claims about real business value.
    """

    cash: int = 100_000
    team: int = 50
    product: int = 25
    market_evidence: int = 20
    reputation: int = 50
    compliance: int = 60

    def __post_init__(self) -> None:
        for field_name in STATE_FIELDS:
            value = _strict_int(getattr(self, field_name), field=field_name)
            lower, upper = STATE_BOUNDS[field_name]
            if not lower <= value <= upper:
                raise ValueError(
                    f"{field_name} must be between {lower} and {upper}; got {value}"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntrepreneurshipState":
        """Deserialize a state while rejecting missing or unknown fields."""

        if not isinstance(value, Mapping):
            raise ValueError("state must be an object")
        supplied = set(value)
        required = set(STATE_FIELDS)
        missing = sorted(required - supplied)
        unknown = sorted(supplied - required)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ValueError("invalid state fields: " + ", ".join(details))
        return cls(**{name: _strict_int(value[name], field=name) for name in STATE_FIELDS})

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in STATE_FIELDS}

    def apply_delta(self, delta: Mapping[str, int]) -> "EntrepreneurshipState":
        """Apply an approved delta and clamp at the documented state bounds.

        Public callers normally use :func:`apply_classified_action`; this
        method is exposed for rule-engine composition and accepts only known
        state fields with integer values.
        """

        updated, _actual, _clamped = _apply_delta_with_audit(self, delta)
        return updated


@dataclass(frozen=True, slots=True)
class EventForm:
    id: str
    form_kind: str
    industry_context: str
    event_text: str
    fixed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "form_kind": self.form_kind,
            "industry_context": self.industry_context,
            "event_text": self.event_text,
            "fixed": self.fixed,
        }


@dataclass(frozen=True, slots=True)
class ActionOption:
    action_code: str
    label: str
    description: str
    delta: Mapping[str, int]
    feedback_cue: str

    def public_dict(self) -> dict[str, str]:
        """Return the classifier-facing action without numeric rule deltas."""

        return {
            "action_code": self.action_code,
            "label": self.label,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class AnchorStateVariant:
    """One frozen, independently allocated common state for an anchor."""

    id: str
    state: EntrepreneurshipState


@dataclass(frozen=True, slots=True)
class EventBlueprint:
    id: str
    family_id: str
    kind: str
    title: str
    target_dimensions: tuple[str, ...]
    difficulty_level: int
    tags: tuple[str, ...]
    preconditions: Mapping[str, tuple[int, int]]
    sandbox_states: tuple[AnchorStateVariant, ...]
    evidence_requirements: Mapping[str, str]
    forms: tuple[EventForm, ...]
    actions: tuple[ActionOption, ...]

    def is_eligible(self, state: EntrepreneurshipState) -> bool:
        for field_name, (minimum, maximum) in self.preconditions.items():
            value = getattr(state, field_name)
            if not minimum <= value <= maximum:
                return False
        return True

    def get_form(self, form_id: str) -> EventForm:
        for form in self.forms:
            if form.id == form_id:
                return form
        raise KeyError(f"unknown form {form_id!r} for blueprint {self.id!r}")

    def get_action(self, action_code: str) -> ActionOption:
        for action in self.actions:
            if action.action_code == action_code:
                return action
        raise KeyError(f"unknown action_code {action_code!r} for blueprint {self.id!r}")


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlueprintValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BlueprintValidationError(f"{field} must be a non-empty list")
    values = tuple(_require_string(item, field=field) for item in value)
    if len(values) != len(set(values)):
        raise BlueprintValidationError(f"{field} contains duplicates")
    return values


def _parse_preconditions(value: Any, *, blueprint_id: str) -> dict[str, tuple[int, int]]:
    if not isinstance(value, Mapping):
        raise BlueprintValidationError(f"{blueprint_id}.preconditions must be an object")
    parsed: dict[str, tuple[int, int]] = {}
    for field_name, condition in value.items():
        if field_name not in STATE_BOUNDS:
            raise BlueprintValidationError(
                f"{blueprint_id}.preconditions has unknown state field {field_name!r}"
            )
        if not isinstance(condition, Mapping) or set(condition) != {"min", "max"}:
            raise BlueprintValidationError(
                f"{blueprint_id}.preconditions.{field_name} requires min and max"
            )
        minimum = _strict_int(condition["min"], field=f"{field_name}.min")
        maximum = _strict_int(condition["max"], field=f"{field_name}.max")
        lower, upper = STATE_BOUNDS[field_name]
        if not lower <= minimum <= maximum <= upper:
            raise BlueprintValidationError(
                f"{blueprint_id}.preconditions.{field_name} lies outside {lower}..{upper}"
            )
        parsed[field_name] = (minimum, maximum)
    return parsed


def _parse_blueprint(raw: Any) -> EventBlueprint:
    if not isinstance(raw, Mapping):
        raise BlueprintValidationError("each blueprint must be an object")
    blueprint_id = _require_string(raw.get("id"), field="blueprint.id")
    family_id = _require_string(raw.get("family_id"), field=f"{blueprint_id}.family_id")
    kind = _require_string(raw.get("kind"), field=f"{blueprint_id}.kind")
    if kind not in {"main", "anchor"}:
        raise BlueprintValidationError(f"{blueprint_id}.kind must be main or anchor")
    title = _require_string(raw.get("title"), field=f"{blueprint_id}.title")

    target_dimensions = _require_string_list(
        raw.get("target_dimensions"), field=f"{blueprint_id}.target_dimensions"
    )
    unknown_dimensions = sorted(set(target_dimensions) - set(ECD_DIMENSIONS))
    if unknown_dimensions:
        raise BlueprintValidationError(
            f"{blueprint_id} has unknown ECD dimensions {unknown_dimensions}"
        )

    difficulty_level = _strict_int(
        raw.get("difficulty_level"), field=f"{blueprint_id}.difficulty_level"
    )
    if not 1 <= difficulty_level <= 5:
        raise BlueprintValidationError(f"{blueprint_id}.difficulty_level must be 1..5")

    raw_tags = raw.get("tags", [])
    if not isinstance(raw_tags, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in raw_tags
    ):
        raise BlueprintValidationError(f"{blueprint_id}.tags must be a string list")
    tags = tuple(dict.fromkeys(tag.strip() for tag in raw_tags))
    preconditions = _parse_preconditions(raw.get("preconditions", {}), blueprint_id=blueprint_id)

    raw_requirements = raw.get("evidence_requirements")
    if not isinstance(raw_requirements, Mapping) or not raw_requirements:
        raise BlueprintValidationError(
            f"{blueprint_id}.evidence_requirements must be a non-empty object"
        )
    if set(raw_requirements) != set(target_dimensions):
        raise BlueprintValidationError(
            f"{blueprint_id}.evidence_requirements must match target_dimensions"
        )
    evidence_requirements = {
        dimension: _require_string(text, field=f"{blueprint_id}.evidence_requirements")
        for dimension, text in raw_requirements.items()
    }

    raw_sandbox = raw.get("sandbox_state")
    raw_sandbox_variants = raw.get("sandbox_states")
    if kind == "anchor":
        sandbox_states: list[AnchorStateVariant] = []
        if raw_sandbox_variants is not None:
            if not isinstance(raw_sandbox_variants, list) or len(raw_sandbox_variants) < 2:
                raise BlueprintValidationError(
                    f"{blueprint_id}.sandbox_states requires at least two controlled variants"
                )
            variant_ids: set[str] = set()
            for raw_variant in raw_sandbox_variants:
                if not isinstance(raw_variant, Mapping):
                    raise BlueprintValidationError(
                        f"{blueprint_id}.sandbox_states entries must be objects"
                    )
                variant_id = _require_string(
                    raw_variant.get("id"), field=f"{blueprint_id}.sandbox_states.id"
                )
                if variant_id in variant_ids:
                    raise BlueprintValidationError(
                        f"duplicate sandbox state variant {variant_id!r} in {blueprint_id}"
                    )
                variant_ids.add(variant_id)
                try:
                    variant_state = EntrepreneurshipState.from_dict(
                        raw_variant.get("state")
                    )
                except ValueError as exc:
                    raise BlueprintValidationError(
                        f"{blueprint_id}.sandbox_states.{variant_id}: {exc}"
                    ) from exc
                sandbox_states.append(AnchorStateVariant(variant_id, variant_state))
        elif raw_sandbox is not None:
            # Compatibility for an old frozen library. Such a library can be
            # rendered, but a single state cannot identify the state facet.
            try:
                sandbox_states.append(
                    AnchorStateVariant(
                        "legacy_single_state",
                        EntrepreneurshipState.from_dict(raw_sandbox),
                    )
                )
            except ValueError as exc:
                raise BlueprintValidationError(f"{blueprint_id}.sandbox_state: {exc}") from exc
        else:
            raise BlueprintValidationError(
                f"{blueprint_id} anchor requires sandbox_states"
            )
    else:
        if raw_sandbox is not None or raw_sandbox_variants is not None:
            raise BlueprintValidationError(
                f"{blueprint_id} main blueprint cannot use sandbox states"
            )
        sandbox_states = []

    raw_forms = raw.get("forms")
    if not isinstance(raw_forms, list) or not raw_forms:
        raise BlueprintValidationError(f"{blueprint_id}.forms must be a non-empty list")
    forms: list[EventForm] = []
    form_ids: set[str] = set()
    for raw_form in raw_forms:
        if not isinstance(raw_form, Mapping):
            raise BlueprintValidationError(f"{blueprint_id}.forms entries must be objects")
        form_id = _require_string(raw_form.get("id"), field=f"{blueprint_id}.form.id")
        if form_id in form_ids:
            raise BlueprintValidationError(f"duplicate form id {form_id!r} in {blueprint_id}")
        form_ids.add(form_id)
        form_kind = _require_string(
            raw_form.get("form_kind"), field=f"{blueprint_id}.{form_id}.form_kind"
        )
        if form_kind not in FORM_KINDS:
            raise BlueprintValidationError(f"{form_id}.form_kind must be one of {FORM_KINDS}")
        fixed = raw_form.get("fixed")
        if not isinstance(fixed, bool):
            raise BlueprintValidationError(f"{form_id}.fixed must be boolean")
        forms.append(
            EventForm(
                id=form_id,
                form_kind=form_kind,
                industry_context=_require_string(
                    raw_form.get("industry_context"), field=f"{form_id}.industry_context"
                ),
                event_text=_require_string(raw_form.get("event_text"), field=f"{form_id}.event_text"),
                fixed=fixed,
            )
        )

    form_kinds = {form.form_kind for form in forms}
    if kind == "main":
        if form_kinds != {"main"} or any(form.fixed for form in forms):
            raise BlueprintValidationError(
                f"{blueprint_id} main forms must have form_kind=main and fixed=false"
            )
    else:
        if "exact" not in form_kinds or "surface" not in form_kinds:
            raise BlueprintValidationError(
                f"{blueprint_id} anchor requires exact and surface forms"
            )
        if form_kinds - {"exact", "surface"} or any(not form.fixed for form in forms):
            raise BlueprintValidationError(
                f"{blueprint_id} anchor forms must be frozen exact/surface forms"
            )

    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or len(raw_actions) < 2:
        raise BlueprintValidationError(f"{blueprint_id}.actions requires at least two options")
    actions: list[ActionOption] = []
    action_codes: set[str] = set()
    for raw_action in raw_actions:
        if not isinstance(raw_action, Mapping):
            raise BlueprintValidationError(f"{blueprint_id}.actions entries must be objects")
        action_code = _require_string(
            raw_action.get("action_code"), field=f"{blueprint_id}.action_code"
        )
        if action_code in action_codes:
            raise BlueprintValidationError(
                f"duplicate action_code {action_code!r} in {blueprint_id}"
            )
        action_codes.add(action_code)
        raw_delta = raw_action.get("delta")
        if not isinstance(raw_delta, Mapping) or not raw_delta:
            raise BlueprintValidationError(f"{blueprint_id}.{action_code}.delta must be an object")
        delta: dict[str, int] = {}
        for field_name, amount in raw_delta.items():
            if field_name not in STATE_BOUNDS:
                raise BlueprintValidationError(
                    f"{blueprint_id}.{action_code}.delta has unknown field {field_name!r}"
                )
            delta[field_name] = _strict_int(amount, field=f"{action_code}.{field_name}")
        action = ActionOption(
            action_code=action_code,
            label=_require_string(raw_action.get("label"), field=f"{action_code}.label"),
            description=_require_string(
                raw_action.get("description"), field=f"{action_code}.description"
            ),
            delta=delta,
            feedback_cue=_require_string(
                raw_action.get("feedback_cue"), field=f"{action_code}.feedback_cue"
            ),
        )
        actions.append(action)

    for state_variant in sandbox_states:
        for action in actions:
            for field_name, amount in action.delta.items():
                resulting = getattr(state_variant.state, field_name) + amount
                lower, upper = STATE_BOUNDS[field_name]
                if not lower <= resulting <= upper:
                    raise BlueprintValidationError(
                        f"{blueprint_id}.{action.action_code} leaves anchor sandbox bounds "
                        f"for state variant {state_variant.id}"
                    )

    return EventBlueprint(
        id=blueprint_id,
        family_id=family_id,
        kind=kind,
        title=title,
        target_dimensions=target_dimensions,
        difficulty_level=difficulty_level,
        tags=tags,
        preconditions=preconditions,
        sandbox_states=tuple(sandbox_states),
        evidence_requirements=evidence_requirements,
        forms=tuple(forms),
        actions=tuple(actions),
    )


class BlueprintLibrary:
    """Validated, reusable event blueprint library."""

    def __init__(self, *, schema_version: str, library_version: str, blueprints: Sequence[EventBlueprint]):
        self.schema_version = schema_version
        self.library_version = library_version
        self._blueprints = tuple(blueprints)
        self._by_id = {blueprint.id: blueprint for blueprint in self._blueprints}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BlueprintLibrary":
        if not isinstance(raw, Mapping):
            raise BlueprintValidationError("blueprint library must be an object")
        schema_version = _require_string(raw.get("schema_version"), field="schema_version")
        if schema_version != "1.0":
            raise BlueprintValidationError(f"unsupported schema_version {schema_version!r}")
        library_version = _require_string(raw.get("library_version"), field="library_version")
        dimensions = _require_string_list(raw.get("ecd_dimensions"), field="ecd_dimensions")
        if dimensions != ECD_DIMENSIONS:
            raise BlueprintValidationError(
                "ecd_dimensions must contain the five canonical dimensions in canonical order"
            )
        raw_blueprints = raw.get("blueprints")
        if not isinstance(raw_blueprints, list):
            raise BlueprintValidationError("blueprints must be a list")
        blueprints = tuple(_parse_blueprint(item) for item in raw_blueprints)
        ids = [blueprint.id for blueprint in blueprints]
        if len(ids) != len(set(ids)):
            raise BlueprintValidationError("blueprint ids must be globally unique")
        form_ids = [form.id for blueprint in blueprints for form in blueprint.forms]
        if len(form_ids) != len(set(form_ids)):
            raise BlueprintValidationError("form ids must be globally unique")

        mains = [blueprint for blueprint in blueprints if blueprint.kind == "main"]
        anchors = [blueprint for blueprint in blueprints if blueprint.kind == "anchor"]
        if len(mains) < 2:
            raise BlueprintValidationError("library requires multiple main-path blueprints")
        if len({blueprint.family_id for blueprint in anchors}) < 3:
            raise BlueprintValidationError("library requires at least three anchor families")
        covered = {dimension for blueprint in anchors for dimension in blueprint.target_dimensions}
        missing = sorted(set(ECD_DIMENSIONS) - covered)
        if missing:
            raise BlueprintValidationError(
                f"anchor blueprints do not cover all ECD dimensions; missing {missing}"
            )
        return cls(
            schema_version=schema_version,
            library_version=library_version,
            blueprints=blueprints,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BlueprintLibrary":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BlueprintValidationError(f"cannot load blueprint library {source}: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def load_default(cls) -> "BlueprintLibrary":
        return cls.load(Path(__file__).with_name("data") / "blueprints.json")

    @property
    def blueprints(self) -> tuple[EventBlueprint, ...]:
        return self._blueprints

    @property
    def main_blueprints(self) -> tuple[EventBlueprint, ...]:
        return tuple(item for item in self._blueprints if item.kind == "main")

    @property
    def anchor_blueprints(self) -> tuple[EventBlueprint, ...]:
        return tuple(item for item in self._blueprints if item.kind == "anchor")

    def get(self, blueprint_id: str) -> EventBlueprint:
        try:
            return self._by_id[blueprint_id]
        except KeyError as exc:
            raise KeyError(f"unknown blueprint {blueprint_id!r}") from exc


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    event_kind: str
    blueprint_id: str
    family_id: str
    form_id: str
    title: str
    target_dimensions: tuple[str, ...]
    difficulty_level: int
    state_variant_id: str
    visible_state: EntrepreneurshipState
    event_text: str
    form_fixed: bool
    allowed_actions: tuple[Mapping[str, str], ...]
    evidence_requirements: Mapping[str, str]
    selection_probability: float
    scheduler_audit: Mapping[str, Any]

    @property
    def allowed_action_codes(self) -> tuple[str, ...]:
        return tuple(action["action_code"] for action in self.allowed_actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_kind": self.event_kind,
            "blueprint_id": self.blueprint_id,
            "family_id": self.family_id,
            "form_id": self.form_id,
            "title": self.title,
            "target_dimensions": list(self.target_dimensions),
            "difficulty_level": self.difficulty_level,
            "state_variant_id": self.state_variant_id,
            "visible_state": self.visible_state.to_dict(),
            "event_text": self.event_text,
            "form_fixed": self.form_fixed,
            "allowed_actions": [dict(action) for action in self.allowed_actions],
            "evidence_requirements": dict(self.evidence_requirements),
            "selection_probability": self.selection_probability,
            "scheduler_audit": dict(self.scheduler_audit),
        }


class EventScheduler:
    """Schedule main and common-anchor events with reproducible allocation."""

    VERSION = "sha256-checkpoint-v1"

    def __init__(
        self,
        library: BlueprintLibrary,
        *,
        anchor_every: int = 4,
        exact_anchor_probability: float = 0.5,
        allocation_namespace: str = "venturebridge-common-anchor-v1",
        anchor_plan: AnchorPlanLookup | None = None,
        allow_development_fallback: bool = True,
    ) -> None:
        if isinstance(anchor_every, bool) or not isinstance(anchor_every, int) or anchor_every < 2:
            raise ValueError("anchor_every must be an integer of at least 2")
        if not 0.0 < exact_anchor_probability < 1.0:
            raise ValueError("exact_anchor_probability must lie strictly between 0 and 1")
        if not allocation_namespace:
            raise ValueError("allocation_namespace cannot be empty")
        if not isinstance(allow_development_fallback, bool):
            raise ValueError("allow_development_fallback must be a boolean")
        self.library = library
        self.anchor_every = anchor_every
        self.exact_anchor_probability = exact_anchor_probability
        self.allocation_namespace = allocation_namespace
        self.anchor_plan = anchor_plan
        self.allow_development_fallback = allow_development_fallback

    def _digest(self, *parts: object) -> bytes:
        material = "\x1f".join(str(part) for part in (self.allocation_namespace, *parts))
        return hashlib.sha256(material.encode("utf-8")).digest()

    def _index(self, candidates: Sequence[Any], *parts: object) -> tuple[int, str]:
        if not candidates:
            raise SchedulingError("cannot draw from an empty candidate set")
        digest = self._digest(*parts)
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        return index, digest.hex()[:16]

    def _unit_draw(self, *parts: object) -> tuple[float, str]:
        digest = self._digest(*parts)
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        return value, digest.hex()[:16]

    def schedule(
        self,
        *,
        simulation_id: str,
        turn_no: int,
        main_state: EntrepreneurshipState,
        used_blueprint_ids: Iterable[str] = (),
        direction_tags: Iterable[str] = (),
        allocation_unit_id: str | None = None,
    ) -> ScheduledEvent:
        """Select one event and expose the conditional allocation probability.

        ``direction_tags`` may narrow *main-path* content.  It is intentionally
        ignored at anchor checkpoints: common-anchor delivery depends only on
        the checkpoint, prior anchor exposure and deterministic allocation key,
        never on learner profile or personality fields.
        """

        if not isinstance(simulation_id, str) or not simulation_id.strip():
            raise SchedulingError("simulation_id must be a non-empty string")
        if isinstance(turn_no, bool) or not isinstance(turn_no, int) or turn_no < 1:
            raise SchedulingError("turn_no must be a positive integer")
        if not isinstance(main_state, EntrepreneurshipState):
            raise SchedulingError("main_state must be EntrepreneurshipState")
        if self.anchor_plan is None and not self.allow_development_fallback:
            raise SchedulingError(
                "common-anchor delivery is blocked until a registered queue plan is active"
            )
        used = frozenset(used_blueprint_ids)
        tags = frozenset(tag.strip() for tag in direction_tags if isinstance(tag, str) and tag.strip())
        normalized_unit_id: str | None = None
        if self.anchor_plan is not None:
            if not isinstance(allocation_unit_id, str) or not allocation_unit_id.strip():
                raise SchedulingError(
                    "a configured queue anchor plan requires an allocation_unit_id"
                )
            normalized_unit_id = allocation_unit_id.strip()
            try:
                self.anchor_plan.assignment_for(normalized_unit_id, 1)
            except KeyError as exc:
                raise SchedulingError(
                    "the frozen queue plan has no assignment for this "
                    "allocation_unit_id"
                ) from exc
        if turn_no % self.anchor_every == 0:
            if self.anchor_plan is not None:
                return self._schedule_planned_anchor(
                    allocation_unit_id=str(normalized_unit_id),
                    turn_no=turn_no,
                    main_state=main_state,
                    used=used,
                )
            return self._schedule_anchor(
                simulation_id=simulation_id,
                turn_no=turn_no,
                main_state=main_state,
                used=used,
            )
        return self._schedule_main(
            simulation_id=simulation_id,
            turn_no=turn_no,
            main_state=main_state,
            used=used,
            direction_tags=tags,
            allocation_unit_id=normalized_unit_id,
        )

    def _schedule_planned_anchor(
        self,
        *,
        allocation_unit_id: str,
        turn_no: int,
        main_state: EntrepreneurshipState,
        used: frozenset[str],
    ) -> ScheduledEvent:
        """Resolve one fail-closed assignment from a queue-balanced plan."""

        if self.anchor_plan is None:  # guarded by the caller
            raise SchedulingError("no queue anchor plan is configured")
        checkpoint_index = turn_no // self.anchor_every
        try:
            assignment = self.anchor_plan.assignment_for(
                allocation_unit_id, checkpoint_index
            )
        except KeyError as exc:
            raise SchedulingError(
                "the frozen queue plan has no assignment for this unit and checkpoint"
            ) from exc

        try:
            blueprint = self.library.get(str(assignment["blueprint_id"]))
            form = blueprint.get_form(str(assignment["form_id"]))
            state_variant = next(
                variant
                for variant in blueprint.sandbox_states
                if variant.id == str(assignment["state_variant_id"])
            )
            probabilities = assignment["conditional_probabilities"]
            blueprint_probability = float(probabilities["blueprint"]["value"])
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise SchedulingError(
                "the frozen queue plan assignment conflicts with the runtime library"
            ) from exc
        if blueprint.kind != "anchor" or form.form_kind not in {"exact", "surface"}:
            raise SchedulingError("the queue plan must reference a frozen anchor form")
        if form.form_kind != assignment.get("form_kind") or not form.fixed:
            raise SchedulingError("the queue plan form metadata is inconsistent")
        if blueprint.id in used:
            raise SchedulingError(
                "the queue plan would repeat an anchor blueprint for this trajectory"
            )
        if not 0.0 < blueprint_probability <= 1.0:
            raise SchedulingError("the queue plan contains an invalid design probability")

        audit = {
            "scheduler_version": f"{self.VERSION}+{assignment['algorithm_version']}",
            "library_version": self.library.library_version,
            "deterministic_allocation": True,
            "allocation_mode": "precomputed_queue_balanced",
            "anchor_checkpoint": True,
            "checkpoint_rule": f"every {self.anchor_every}th turn",
            "anchor_every_turns": self.anchor_every,
            "checkpoint_index": checkpoint_index,
            "selection_basis": (
                "frozen queue-level common-anchor plan independent of learner profile, "
                "main-path state, responses and outcomes"
            ),
            "profile_features_used_for_anchor": False,
            "main_state_used_for_anchor_selection": False,
            "allocation_unit_id": allocation_unit_id,
            "allocation_rank": assignment["allocation_rank"],
            "allocation_plan_hash": assignment["plan_hash"],
            "allocation_algorithm_version": assignment["algorithm_version"],
            "assignment_hash": assignment["assignment_hash"],
            "state_variant_id": state_variant.id,
            "conditional_design_probabilities": probabilities,
            "blueprint_probability": blueprint_probability,
            "selection_probability_compatibility_field": (
                "blueprint component only; coupled form/state quota probabilities remain "
                "in conditional_design_probabilities and must not be multiplied"
            ),
            "probability_scope": (
                "conditional design probability under the registered queue plan; "
                "not an outcome probability or unconditional propensity"
            ),
            "reason": "precomputed_common_anchor_checkpoint",
            "main_state_snapshot_hash": hashlib.sha256(
                json.dumps(main_state.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
        }
        return self._build_event(
            blueprint,
            form,
            state_variant.state,
            blueprint_probability,
            audit,
            state_variant_id=state_variant.id,
        )

    def _schedule_main(
        self,
        *,
        simulation_id: str,
        turn_no: int,
        main_state: EntrepreneurshipState,
        used: frozenset[str],
        direction_tags: frozenset[str],
        allocation_unit_id: str | None,
    ) -> ScheduledEvent:
        eligible = [item for item in self.library.main_blueprints if item.is_eligible(main_state)]
        if not eligible:
            raise SchedulingError("no main-path blueprint is eligible for the current state")
        tag_matched = [item for item in eligible if direction_tags.intersection(item.tags)]
        tag_filter_used = bool(tag_matched)
        candidates = tag_matched or eligible
        unused = [item for item in candidates if item.id not in used]
        pool_reset = not unused
        candidates = unused or candidates
        candidates.sort(key=lambda item: item.id)
        index, blueprint_draw = self._index(
            candidates, simulation_id, turn_no, "main-blueprint", *(item.id for item in candidates)
        )
        blueprint = candidates[index]
        forms = sorted(blueprint.forms, key=lambda item: item.id)
        form_index, form_draw = self._index(
            forms, simulation_id, turn_no, "main-form", blueprint.id, *(item.id for item in forms)
        )
        form = forms[form_index]
        probability = (1.0 / len(candidates)) * (1.0 / len(forms))
        audit = {
            "scheduler_version": self.VERSION,
            "library_version": self.library.library_version,
            "deterministic_allocation": True,
            "anchor_checkpoint": False,
            "checkpoint_rule": f"every {self.anchor_every}th turn",
            "anchor_every_turns": self.anchor_every,
            "selection_basis": "eligible main-path blueprint; venture direction tags when available",
            "profile_features_used_for_anchor": False,
            "direction_tag_filter_used": tag_filter_used,
            "candidate_blueprint_ids": [item.id for item in candidates],
            "used_pool_reset": pool_reset,
            "blueprint_draw_fingerprint": blueprint_draw,
            "form_draw_fingerprint": form_draw,
            "blueprint_probability": 1.0 / len(candidates),
            "form_probability": 1.0 / len(forms),
            "probability_scope": "conditional on current state, main checkpoint and candidate pool",
            "reason": "main_path_turn",
        }
        if self.anchor_plan is not None:
            audit.update(
                {
                    "allocation_mode": "precomputed_queue_balanced",
                    "allocation_plan_hash": self.anchor_plan.plan_hash,
                    "allocation_unit_id": allocation_unit_id,
                }
            )
        else:
            audit["allocation_mode"] = "development_hash_fallback"
        return self._build_event(
            blueprint,
            form,
            main_state,
            probability,
            audit,
            state_variant_id="main_path_state",
        )

    def _schedule_anchor(
        self,
        *,
        simulation_id: str,
        turn_no: int,
        main_state: EntrepreneurshipState,
        used: frozenset[str],
    ) -> ScheduledEvent:
        anchors = sorted(self.library.anchor_blueprints, key=lambda item: item.id)
        unused = [item for item in anchors if item.id not in used]
        pool_reset = not unused
        candidates = unused or anchors
        index, blueprint_draw = self._index(
            candidates, simulation_id, turn_no, "anchor-blueprint", *(item.id for item in candidates)
        )
        blueprint = candidates[index]

        state_variants = sorted(blueprint.sandbox_states, key=lambda item: item.id)
        if not state_variants:  # guarded by library validation
            raise SchedulingError(f"anchor {blueprint.id} has no sandbox state")
        state_index, state_draw = self._index(
            state_variants,
            simulation_id,
            turn_no,
            "anchor-state",
            blueprint.id,
            *(item.id for item in state_variants),
        )
        state_variant = state_variants[state_index]

        exact_forms = sorted(
            (form for form in blueprint.forms if form.form_kind == "exact"), key=lambda item: item.id
        )
        surface_forms = sorted(
            (form for form in blueprint.forms if form.form_kind == "surface"), key=lambda item: item.id
        )
        kind_draw, kind_fingerprint = self._unit_draw(
            simulation_id, turn_no, "anchor-form-kind", blueprint.id
        )
        if kind_draw < self.exact_anchor_probability:
            form_pool = exact_forms
            kind_probability = self.exact_anchor_probability
            selected_kind = "exact"
        else:
            form_pool = surface_forms
            kind_probability = 1.0 - self.exact_anchor_probability
            selected_kind = "surface"
        form_index, form_draw = self._index(
            form_pool,
            simulation_id,
            turn_no,
            "anchor-form",
            blueprint.id,
            selected_kind,
            *(item.id for item in form_pool),
        )
        form = form_pool[form_index]
        form_probability = kind_probability / len(form_pool)
        state_probability = 1.0 / len(state_variants)
        probability = (1.0 / len(candidates)) * state_probability * form_probability
        audit = {
            "scheduler_version": self.VERSION,
            "library_version": self.library.library_version,
            "deterministic_allocation": True,
            "anchor_checkpoint": True,
            "checkpoint_rule": f"every {self.anchor_every}th turn",
            "anchor_every_turns": self.anchor_every,
            "selection_basis": "common anchor pool independent of learner profile and main-path state",
            "profile_features_used_for_anchor": False,
            "main_state_used_for_anchor_selection": False,
            "candidate_blueprint_ids": [item.id for item in candidates],
            "used_pool_reset": pool_reset,
            "blueprint_draw_fingerprint": blueprint_draw,
            "state_variant_ids": [item.id for item in state_variants],
            "state_variant_id": state_variant.id,
            "state_variant_draw_fingerprint": state_draw,
            "state_variant_probability": state_probability,
            "form_kind_draw_fingerprint": kind_fingerprint,
            "form_draw_fingerprint": form_draw,
            "blueprint_probability": 1.0 / len(candidates),
            "form_kind_probability": kind_probability,
            "within_kind_form_probability": 1.0 / len(form_pool),
            "form_probability": form_probability,
            "probability_scope": "conditional on anchor checkpoint and prior anchor exposure",
            "reason": "common_anchor_checkpoint",
            "allocation_mode": "development_hash_fallback",
            "main_state_snapshot_hash": hashlib.sha256(
                json.dumps(main_state.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
        }
        return self._build_event(
            blueprint,
            form,
            state_variant.state,
            probability,
            audit,
            state_variant_id=state_variant.id,
        )

    @staticmethod
    def _build_event(
        blueprint: EventBlueprint,
        form: EventForm,
        visible_state: EntrepreneurshipState,
        probability: float,
        audit: Mapping[str, Any],
        *,
        state_variant_id: str,
    ) -> ScheduledEvent:
        if form.form_kind == "main":
            event_kind = "main"
        elif form.form_kind == "exact":
            event_kind = "exact_anchor"
        else:
            event_kind = "surface_anchor"
        return ScheduledEvent(
            event_kind=event_kind,
            blueprint_id=blueprint.id,
            family_id=blueprint.family_id,
            form_id=form.id,
            title=blueprint.title,
            target_dimensions=blueprint.target_dimensions,
            difficulty_level=blueprint.difficulty_level,
            state_variant_id=state_variant_id,
            visible_state=visible_state,
            event_text=form.event_text,
            form_fixed=form.fixed,
            allowed_actions=tuple(action.public_dict() for action in blueprint.actions),
            evidence_requirements=dict(blueprint.evidence_requirements),
            selection_probability=probability,
            scheduler_audit=dict(audit),
        )


@dataclass(frozen=True, slots=True)
class TransitionResult:
    event_kind: str
    blueprint_id: str
    action_code: str
    feedback_cue: str
    main_state_before: EntrepreneurshipState
    main_state_after: EntrepreneurshipState
    sandbox_state_before: EntrepreneurshipState | None
    sandbox_state_after: EntrepreneurshipState | None
    requested_delta: Mapping[str, int]
    applied_delta: Mapping[str, int]
    clamped_fields: tuple[str, ...]
    main_state_restored: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_kind": self.event_kind,
            "blueprint_id": self.blueprint_id,
            "action_code": self.action_code,
            "feedback_cue": self.feedback_cue,
            "main_state_before": self.main_state_before.to_dict(),
            "main_state_after": self.main_state_after.to_dict(),
            "sandbox_state_before": (
                self.sandbox_state_before.to_dict() if self.sandbox_state_before else None
            ),
            "sandbox_state_after": (
                self.sandbox_state_after.to_dict() if self.sandbox_state_after else None
            ),
            "requested_delta": dict(self.requested_delta),
            "applied_delta": dict(self.applied_delta),
            "clamped_fields": list(self.clamped_fields),
            "main_state_restored": self.main_state_restored,
        }


def _apply_delta_with_audit(
    state: EntrepreneurshipState, delta: Mapping[str, int]
) -> tuple[EntrepreneurshipState, dict[str, int], tuple[str, ...]]:
    if not isinstance(delta, Mapping) or not delta:
        raise StateTransitionError("delta must be a non-empty mapping")
    values = state.to_dict()
    actual: dict[str, int] = {}
    clamped: list[str] = []
    for field_name, amount in delta.items():
        if field_name not in STATE_BOUNDS:
            raise StateTransitionError(f"delta contains unknown state field {field_name!r}")
        try:
            amount = _strict_int(amount, field=f"delta.{field_name}")
        except ValueError as exc:
            raise StateTransitionError(str(exc)) from exc
        before = values[field_name]
        lower, upper = STATE_BOUNDS[field_name]
        after = min(upper, max(lower, before + amount))
        values[field_name] = after
        actual[field_name] = after - before
        if actual[field_name] != amount:
            clamped.append(field_name)
    return EntrepreneurshipState.from_dict(values), actual, tuple(sorted(clamped))


def apply_classified_action(
    *,
    library: BlueprintLibrary,
    event: ScheduledEvent,
    main_state: EntrepreneurshipState,
    action_code: str,
) -> TransitionResult:
    """Apply a classifier-selected code using only the frozen blueprint delta.

    The function accepts no caller-provided numeric delta.  Unknown codes are
    rejected instead of being repaired or interpreted.  Anchor decisions are
    applied to their shared sandbox and the personal main state is returned
    unchanged.
    """

    if event.event_kind not in EVENT_KINDS:
        raise StateTransitionError(f"unsupported event kind {event.event_kind!r}")
    if not isinstance(action_code, str) or not action_code:
        raise StateTransitionError("action_code must be a non-empty string")
    blueprint = library.get(event.blueprint_id)
    event_library_version = event.scheduler_audit.get("library_version")
    if event_library_version != library.library_version:
        raise StateTransitionError(
            "event blueprint library version does not match the active immutable library: "
            f"event={event_library_version!r}, active={library.library_version!r}"
        )
    try:
        form = blueprint.get_form(event.form_id)
    except KeyError as exc:
        raise StateTransitionError(str(exc)) from exc
    expected_kind = {
        "main": "main",
        "exact": "exact_anchor",
        "surface": "surface_anchor",
    }[form.form_kind]
    if event.event_kind != expected_kind:
        raise StateTransitionError("event kind does not match its frozen blueprint form")
    try:
        action = blueprint.get_action(action_code)
    except KeyError as exc:
        raise StateTransitionError(
            f"action_code {action_code!r} is not allowed; expected one of "
            f"{[item.action_code for item in blueprint.actions]}"
        ) from exc

    if event.event_kind == "main":
        if event.visible_state != main_state:
            raise StateTransitionError(
                "stale main-path event: current state differs from its scheduled snapshot"
            )
        main_after, applied, clamped = _apply_delta_with_audit(main_state, action.delta)
        return TransitionResult(
            event_kind=event.event_kind,
            blueprint_id=blueprint.id,
            action_code=action.action_code,
            feedback_cue=action.feedback_cue,
            main_state_before=main_state,
            main_state_after=main_after,
            sandbox_state_before=None,
            sandbox_state_after=None,
            requested_delta=dict(action.delta),
            applied_delta=applied,
            clamped_fields=clamped,
            main_state_restored=False,
        )

    if not blueprint.sandbox_states:
        raise StateTransitionError("anchor blueprint is missing its common sandbox state")
    matching_state = next(
        (
            state_variant.state
            for state_variant in blueprint.sandbox_states
            if state_variant.id == event.state_variant_id
        ),
        None,
    )
    if matching_state is None:
        raise StateTransitionError("anchor event references an unknown state variant")
    if event.visible_state != matching_state:
        raise StateTransitionError("anchor event does not expose the frozen common sandbox state")
    sandbox_after, applied, clamped = _apply_delta_with_audit(
        matching_state, action.delta
    )
    return TransitionResult(
        event_kind=event.event_kind,
        blueprint_id=blueprint.id,
        action_code=action.action_code,
        feedback_cue=action.feedback_cue,
        main_state_before=main_state,
        main_state_after=main_state,
        sandbox_state_before=matching_state,
        sandbox_state_after=sandbox_after,
        requested_delta=dict(action.delta),
        applied_delta=applied,
        clamped_fields=clamped,
        main_state_restored=True,
    )


__all__ = [
    "ActionOption",
    "AnchorStateVariant",
    "BlueprintLibrary",
    "BlueprintValidationError",
    "ECD_DIMENSIONS",
    "EntrepreneurshipState",
    "EventBlueprint",
    "EventForm",
    "EventScheduler",
    "ScheduledEvent",
    "SchedulingError",
    "STATE_BOUNDS",
    "STATE_FIELDS",
    "StateTransitionError",
    "TransitionResult",
    "apply_classified_action",
]
