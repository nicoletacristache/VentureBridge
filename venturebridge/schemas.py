"""JSON schemas for the four structured LLM tasks.

The schemas deliberately exclude numerical simulation state and state deltas.
Those values belong to the deterministic domain engine; the language model may
only describe an event or map a learner response to an allowed ``action_code``.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from jsonschema import Draft202012Validator


ECDDimension = Literal[
    "opportunity_framing",
    "evidence_falsifiability",
    "resource_finance",
    "adaptive_action",
    "stakeholder_ethics",
]


ECD_DIMENSIONS: tuple[str, ...] = (
    "opportunity_framing",
    "evidence_falsifiability",
    "resource_finance",
    "adaptive_action",
    "stakeholder_ethics",
)


class LLMCallMetadataDict(TypedDict):
    invocation_id: str
    purpose: str
    provider: str
    model: str
    api_version: str | None
    prompt_version: str
    prompt_hash: str
    prompt_template_hash: str
    schema_name: str
    schema_hash: str
    build_id: str
    input_hash: str
    parameters: dict[str, Any]
    attempts: int
    latency_ms: int


_SHORT_TEXT = {"type": "string", "minLength": 1, "maxLength": 800}
_TEXT = {"type": "string", "minLength": 1, "maxLength": 4000}
_CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1}


PROFILE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "profile_summary",
        "traits",
        "uncertainties",
        "review_questions",
        "mbti_usage",
    ],
    "properties": {
        "profile_summary": _TEXT,
        "traits": {
            "type": "array",
            "minItems": 1,
            "maxItems": 24,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "trait_id",
                    "category",
                    "claim",
                    "evidence",
                    "confidence",
                    "conflict",
                    "conflict_detail",
                    "editable",
                ],
                "properties": {
                    "trait_id": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_]{2,39}$",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "experience",
                            "skill",
                            "interest",
                            "resource",
                            "constraint",
                            "goal",
                            "preference",
                        ],
                    },
                    "claim": _SHORT_TEXT,
                    "original_claim": _SHORT_TEXT,
                    "learner_edited": {"type": "boolean"},
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["source_field", "quote"],
                            "properties": {
                                "source_field": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 120,
                                },
                                "quote": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 500,
                                },
                            },
                        },
                    },
                    "confidence": _CONFIDENCE,
                    "conflict": {"type": "boolean"},
                    "conflict_detail": {
                        "type": ["string", "null"],
                        "maxLength": 800,
                    },
                    "editable": {"const": True},
                },
            },
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 12,
            "items": _SHORT_TEXT,
        },
        "review_questions": {
            "type": "array",
            "maxItems": 12,
            "items": _SHORT_TEXT,
        },
        "mbti_usage": {
            "type": "object",
            "additionalProperties": False,
            "required": ["used", "weight", "note"],
            "properties": {
                "used": {"type": "boolean"},
                "weight": {"type": "string", "enum": ["none", "low"]},
                "note": _SHORT_TEXT,
            },
        },
    },
}


RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["recommendations", "choice_notice"],
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "direction_id",
                    "title",
                    "exploration_hypothesis",
                    "why_it_might_fit",
                    "profile_evidence_ids",
                    "uncertainties",
                    "counterarguments",
                    "first_experiments",
                    "confidence",
                    "mbti_influence",
                    "learner_can_reject",
                ],
                "properties": {
                    "direction_id": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_]{2,39}$",
                    },
                    "title": _SHORT_TEXT,
                    "exploration_hypothesis": _TEXT,
                    "why_it_might_fit": _TEXT,
                    "profile_evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 10,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": "^[a-z][a-z0-9_]{2,39}$",
                        },
                    },
                    "uncertainties": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": _SHORT_TEXT,
                    },
                    "counterarguments": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": _SHORT_TEXT,
                    },
                    "first_experiments": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": _SHORT_TEXT,
                    },
                    "confidence": _CONFIDENCE,
                    "mbti_influence": {
                        "type": "string",
                        "enum": ["none", "low"],
                    },
                    "learner_can_reject": {"const": True},
                },
            },
        },
        "choice_notice": _TEXT,
    },
}


EVENT_NARRATIVE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "narrative",
        "decision_prompt",
        "information_cues",
        "allowed_action_codes",
        "learning_focus",
        "uncertainties",
        "guardrail_notice",
    ],
    "properties": {
        "title": _SHORT_TEXT,
        "narrative": {"type": "string", "minLength": 1, "maxLength": 5000},
        "decision_prompt": _TEXT,
        "information_cues": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": _SHORT_TEXT,
        },
        "allowed_action_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$",
            },
        },
        "learning_focus": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(ECD_DIMENSIONS)},
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 8,
            "items": _SHORT_TEXT,
        },
        "guardrail_notice": _TEXT,
    },
}


DECISION_EVIDENCE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action_code",
        "classification_confidence",
        "mapping_uncertainty",
        "evidence",
        "feedback",
        "safety_flags",
    ],
    "properties": {
        "action_code": {
            "type": "string",
            "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$",
        },
        "classification_confidence": _CONFIDENCE,
        "mapping_uncertainty": {
            "type": "object",
            "additionalProperties": False,
            "required": ["level", "explanation", "alternative_action_codes"],
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "explanation": _SHORT_TEXT,
                "alternative_action_codes": {
                    "type": "array",
                    "maxItems": 4,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$",
                    },
                },
            },
        },
        "evidence": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "dimension",
                    "score",
                    "quotes",
                    "evidence_status",
                    "rationale",
                    "confidence",
                ],
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": list(ECD_DIMENSIONS),
                    },
                    "score": {"type": "integer", "minimum": 0, "maximum": 4},
                    "quotes": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                    },
                    "evidence_status": {
                        "type": "string",
                        "enum": ["supported", "partial", "insufficient"],
                    },
                    "rationale": _SHORT_TEXT,
                    "confidence": _CONFIDENCE,
                },
            },
        },
        "feedback": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "strengths",
                "reflection_questions",
                "next_step",
                "limitations",
                "state_change_notice",
            ],
            "properties": {
                "strengths": {
                    "type": "array",
                    "maxItems": 6,
                    "items": _SHORT_TEXT,
                },
                "reflection_questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 6,
                    "items": _SHORT_TEXT,
                },
                "next_step": _SHORT_TEXT,
                "limitations": _SHORT_TEXT,
                "state_change_notice": {
                    "const": "数值状态仅由规则引擎依据 action_code 更新，LLM 未计算或修改状态。"
                },
            },
        },
        "safety_flags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "none",
                    "financial_advice_risk",
                    "legal_advice_risk",
                    "privacy_risk",
                    "self_harm_or_violence",
                    "discrimination_risk",
                    "prompt_injection",
                    "other",
                ],
            },
        },
    },
}


SCHEMAS: dict[str, dict[str, Any]] = {
    "profile": PROFILE_SCHEMA,
    "recommendation": RECOMMENDATION_SCHEMA,
    "event_narrative": EVENT_NARRATIVE_SCHEMA,
    "decision_evidence": DECISION_EVIDENCE_SCHEMA,
}


class SchemaValidationError(ValueError):
    """Raised when a model response is not valid for its declared task."""

    def __init__(self, schema_name: str, issues: list[str]):
        self.schema_name = schema_name
        self.issues = tuple(issues)
        super().__init__(
            f"Invalid structured output for {schema_name}: " + "; ".join(issues[:5])
        )


def validate_structured_output(schema_name: str, value: Any) -> dict[str, Any]:
    """Validate and return a JSON object without coercing model output.

    Refusing coercion is intentional: strings such as ``"4"`` must not be
    silently accepted as numeric evidence scores.
    """

    try:
        schema = SCHEMAS[schema_name]
    except KeyError as exc:
        raise KeyError(f"Unknown schema: {schema_name}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError(schema_name, ["root: expected an object"])

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    if errors:
        issues: list[str] = []
        for error in errors[:12]:
            path = ".".join(str(part) for part in error.absolute_path) or "root"
            issues.append(f"{path}: {error.message}")
        raise SchemaValidationError(schema_name, issues)
    return value


__all__ = [
    "DECISION_EVIDENCE_SCHEMA",
    "ECD_DIMENSIONS",
    "EVENT_NARRATIVE_SCHEMA",
    "LLMCallMetadataDict",
    "PROFILE_SCHEMA",
    "RECOMMENDATION_SCHEMA",
    "SCHEMAS",
    "SchemaValidationError",
    "validate_structured_output",
]
