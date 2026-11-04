"""Privacy-gated, structured LLM access for VentureBridge.

The default service always constructs a real OpenAI-compatible provider from
environment variables.  A fake provider exists only as an explicitly enabled
test double; there is no automatic fallback when credentials or the network
are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import time
import urllib.error
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from openai_messages_client import chat_completion

from .privacy import SensitiveDataError, assert_no_sensitive_data, sanitize_for_llm
from .prompts import (
    DECISION_PROMPT_VERSION,
    EVENT_PROMPT_VERSION,
    JSON_NORMALIZER_PROMPT_VERSION,
    PROFILE_PROMPT_VERSION,
    RECOMMENDATION_PROMPT_VERSION,
    build_decision_messages,
    build_event_messages,
    build_json_normalizer_messages,
    build_profile_messages,
    build_recommendation_messages,
)
from .schemas import (
    ECD_DIMENSIONS,
    SCHEMAS,
    SchemaValidationError,
    validate_structured_output,
)


Message = dict[str, str]
Recorder = Callable[[dict[str, Any]], None]
DECISION_REPAIR_POLICY_VERSION = "decision-evidence-repair-2026-11-en-v1"
DECISION_QUOTE_CANDIDATE_POLICY_VERSION = "decision-verbatim-candidates-2026-11-en-v1"
JSON_EXTRACTION_POLICY_VERSION = "json-regex-extraction-2026-11-v1"

_JSON_NORMALIZER_MAX_INPUT_CHARS = 100_000
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.I | re.S)
_JSON_TAG_RE = re.compile(
    r"<json(?:_output)?>\s*(.*?)\s*</json(?:_output)?>", re.I | re.S
)
_JSON_LABEL_RE = re.compile(
    r"(?:^|\b)(?:json|result|output)\s*[:=]\s*(\{.*\})", re.I | re.S
)
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.S)

_DECISION_SOURCE_FIELDS = (
    "decision_text",
    "rationale_text",
    "information_requests",
    "revision_text",
)
_DECISION_QUOTE_STRONG_BOUNDARY_PATTERN = r"[。！？!?；;\r\n]+"
_DECISION_QUOTE_SOFT_BOUNDARY_PATTERN = r"[，,:：、]+"
_DECISION_REDACTION_MARKER_PATTERN = r"\[(?:removed|\u5df2\u79fb\u9664):[^\]]+\]"
_DECISION_QUOTE_MAX_CHARS = 240
_DECISION_QUOTE_WINDOW_OVERLAP = 40
_DECISION_QUOTE_MAX_TOTAL = 160
_DECISION_QUOTE_STRONG_OR_REDACTION_RE = re.compile(
    rf"(?P<boundary>{_DECISION_QUOTE_STRONG_BOUNDARY_PATTERN})|"
    rf"(?P<redaction>{_DECISION_REDACTION_MARKER_PATTERN})"
)
_DECISION_QUOTE_SOFT_BOUNDARY_RE = re.compile(
    rf"(?P<boundary>{_DECISION_QUOTE_SOFT_BOUNDARY_PATTERN})"
)
_DECISION_REDACTION_MARKER_RE = re.compile(_DECISION_REDACTION_MARKER_PATTERN)
_DECISION_QUOTE_CANDIDATE_ID_RE = re.compile(r"^_*q\d{3}$")

_DECISION_QUOTE_CANDIDATE_POLICY = {
    "version": DECISION_QUOTE_CANDIDATE_POLICY_VERSION,
    "source_fields": list(_DECISION_SOURCE_FIELDS),
    "strong_boundary_pattern": _DECISION_QUOTE_STRONG_BOUNDARY_PATTERN,
    "soft_boundary_pattern": _DECISION_QUOTE_SOFT_BOUNDARY_PATTERN,
    "redaction_marker_pattern": _DECISION_REDACTION_MARKER_PATTERN,
    "candidate_variants": "core_and_original_trailing_boundary",
    "max_candidate_characters": _DECISION_QUOTE_MAX_CHARS,
    "fixed_window_overlap": _DECISION_QUOTE_WINDOW_OVERLAP,
    "max_total_candidates": _DECISION_QUOTE_MAX_TOTAL,
    "selection_order": "field_round_robin_then_source_order",
    "candidate_id_format": "collision_safe_qNNN",
    "model_quote_reference": "candidate_id_preferred_exact_text_accepted",
    "service_resolution": "valid_candidate_id_to_exact_text_before_persistence",
    "unknown_candidate_id": "reject",
}
DECISION_QUOTE_CANDIDATE_POLICY_HASH = hashlib.sha256(
    json.dumps(
        _DECISION_QUOTE_CANDIDATE_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _repair_instruction(schema_name: str, error_code: str) -> str:
    """Return a versioned, label-blind repair instruction for structured output."""

    prefix = "The previous output failed structural or constraint validation. "
    decision_repairs = {
        "unsupported_decision_quote": (
            "At least one quote is not a verbatim contiguous substring of the four response fields in this turn. Remove paraphrased quotes and select again from verbatim_quote_candidates. In quotes, preferably return only a valid candidate_id; the system will deterministically expand it to the original text. Do not copy or rewrite text yourself."
        ),
        "unlisted_decision_quote_candidate": (
            "At least one quote may occur in the response but does not exactly equal any complete verbatim_quote_candidates[].text value. Select again from the candidate list and return only the corresponding candidate_id in quotes. Do not shorten, extend, combine, or rewrite text."
        ),
        "unknown_decision_quote_candidate_id": (
            "At least one quote looks like a candidate ID but is absent from this turn's verbatim_quote_candidates. Copy an existing candidate_id from the current list exactly; do not guess, alter, or reuse an ID from another sample."
        ),
        "missing_decision_quote": (
            "At least one item with evidence_status supported or partial has no quote. If supporting evidence or an explicit contrary action exists, quotes must contain at least one candidate_id from the current candidates. If no supporting evidence exists, use score=0, quotes=[], and evidence_status=insufficient."
        ),
        "unexpected_decision_quote": (
            "At least one item with evidence_status=insufficient still contains a quote. Every insufficient item must also have score=0 and quotes=[]. To retain a verbatim quote of an explicit contrary action, change evidence_status to supported or partial and select a valid candidate_id in quotes."
        ),
        "unsupported_nonzero_score": (
            "At least one item with evidence_status=insufficient has a nonzero score. Every insufficient item must have score=0 and quotes=[]. If the source contains supporting evidence or an explicit contrary action, use supported or partial and select a valid candidate_id in quotes."
        ),
        "ecd_dimensions": (
            "evidence must contain exactly five items, with each of the five ECD dimensions appearing once and only once."
        ),
        "unknown_action_code": (
            "action_code must exactly match an action_code in input allowed_actions; do not invent, translate, or rename it."
        ),
        "unknown_alternative_action": (
            "alternative_action_codes may contain only exact action_code values from input allowed_actions; do not invent, translate, or rename them."
        ),
        "contradictory_safety_flags": (
            "If safety_flags contains none, none must be the only item."
        ),
        "invalid_structure": (
            "Rebuild the complete object according to the decision_evidence Schema. Keep only fields allowed by the Schema; evidence must have exactly five items with each dimension once; score must be an integer from 0 to 4, not a string; confidence must be numeric from 0 to 1; quotes must be an array of strings; evidence_status, mapping_uncertainty.level, and safety_flags must use Schema enum values; include all five required feedback fields; and copy the specified state_change_notice constant exactly."
        ),
    }
    common_repairs = {
        "unexpected_profile_provenance": (
            "For an initial profile, omit original_claim and learner_edited from every trait. "
            "Those fields are reserved for later learner-authored corrections. "
        ),
        "unsupported_profile_quote": (
            "Every profile evidence quote must be a verbatim contiguous substring of the "
            "sanitized field named by source_field. Correct the field path or quote; do not "
            "paraphrase or invent evidence. "
        ),
        "duplicate_trait_id": "Every trait_id must be unique. ",
        "missing_conflict_detail": (
            "When conflict is true, provide a nonempty conflict_detail. "
        ),
        "invented_mbti": "Do not use MBTI unless the original input supplied it. ",
        "mbti_weight": (
            "Use weight=low when MBTI is used and weight=none when it is not used. "
        ),
    }
    detail = (
        decision_repairs.get(error_code)
        if schema_name == "decision_evidence"
        else common_repairs.get(error_code)
    )
    if detail is None:
        detail = "Complete the original task again. "
    return (
        f"{prefix}{detail}"
        "Regenerate the complete JSON from the original input. Do not return only changed fields or repeat the previous output. Return exactly one JSON object that satisfies the JSON Schema and all constraints."
    )


class LLMConfigurationError(RuntimeError):
    """Required real-provider configuration is missing or invalid."""


class LLMProviderError(RuntimeError):
    """A safely classified provider failure (never contains response bodies)."""

    def __init__(self, code: str, message: str, *, retryable: bool):
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class StructuredOutputError(ValueError):
    """The returned text could not be accepted as the requested object."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class LLMRequestFailed(RuntimeError):
    """A high-level call failed after its finite retry policy."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        attempts: int,
        metadata: "LLMCallMetadata",
    ):
        self.code = code
        self.attempts = attempts
        self.metadata = metadata
        super().__init__(message)


class LLMRecorderError(RuntimeError):
    """The durable recorder failed after an LLM attempt completed.

    The exception deliberately hides database/transport details from its
    message.  ``invocation_id`` lets an operator correlate an internal error
    without placing prompts, model output or credentials in logs.
    """

    def __init__(self, invocation_id: str):
        self.invocation_id = invocation_id
        super().__init__(f"LLM audit recorder failed for invocation {invocation_id}.")


@dataclass(frozen=True)
class LLMRuntimeConfig:
    timeout_seconds: int
    max_retries: int
    retry_base_seconds: float
    temperature: float
    max_tokens: int

    def __post_init__(self) -> None:
        _validate_runtime_value(
            "timeout_seconds", self.timeout_seconds, int, minimum=1, maximum=600
        )
        _validate_runtime_value(
            "max_retries", self.max_retries, int, minimum=0, maximum=5
        )
        _validate_runtime_value(
            "retry_base_seconds",
            self.retry_base_seconds,
            (int, float),
            minimum=0.0,
            maximum=30.0,
        )
        _validate_runtime_value(
            "temperature", self.temperature, (int, float), minimum=0.0, maximum=2.0
        )
        _validate_runtime_value(
            "max_tokens", self.max_tokens, int, minimum=128, maximum=32000
        )

    @classmethod
    def from_env(cls) -> "LLMRuntimeConfig":
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, source: Mapping[str, str]) -> "LLMRuntimeConfig":
        timeout = _mapping_int(
            source, "LLM_TIMEOUT_SECONDS", 90, minimum=1, maximum=600
        )
        retries = _mapping_int(source, "LLM_MAX_RETRIES", 2, minimum=0, maximum=5)
        retry_base = _mapping_float(
            source, "LLM_RETRY_BASE_SECONDS", 0.5, minimum=0.0, maximum=30.0
        )
        temperature = _mapping_float(
            source, "LLM_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0
        )
        max_tokens = _mapping_int(
            source, "LLM_MAX_TOKENS", 2400, minimum=128, maximum=32000
        )
        return cls(
            timeout_seconds=timeout,
            max_retries=retries,
            retry_base_seconds=retry_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass(frozen=True)
class LLMCallMetadata:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResult:
    data: dict[str, Any]
    metadata: LLMCallMetadata


@dataclass(frozen=True)
class LLMReadiness:
    """Network-free configuration readiness result with no secret values."""

    ready: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    model: str | None
    api_version: str | None
    build_id: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "model": self.model,
            "api_version": self.api_version,
            "build_id": self.build_id,
        }


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str
    api_version: str | None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        """Return assistant text or raise an exception."""


@dataclass(frozen=True)
class _ProviderSettings:
    api_key: str
    base_url: str
    model: str
    api_version: str | None


def _first_nonblank(source: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _validate_base_url(base_url: str) -> str | None:
    if not base_url:
        return "missing_base_url"
    if any(character in base_url for character in "\r\n\x00"):
        return "invalid_base_url"
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return "invalid_base_url"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "invalid_base_url"
    if parsed.scheme != "https":
        return "https_required"
    if parsed.username is not None or parsed.password is not None:
        return "base_url_contains_credentials"
    if parsed.query or parsed.fragment or port is not None and not 1 <= port <= 65535:
        return "invalid_base_url"
    return None


def _provider_settings(
    source: Mapping[str, str], *, raise_on_error: bool
) -> tuple[_ProviderSettings | None, list[str], list[str]]:
    api_key = _first_nonblank(source, "V_API_KEY")
    base_url = _first_nonblank(source, "V_API_BASE_URL")
    model = _first_nonblank(source, "V_MODEL", "LLM_MODEL")
    api_version = _first_nonblank(source, "V_API_VERSION") or None
    errors: list[str] = []
    warnings: list[str] = []
    if not api_key:
        errors.append("missing_api_key")
    elif any(character in api_key for character in "\r\n\x00"):
        errors.append("invalid_api_key")
    base_error = _validate_base_url(base_url)
    if base_error:
        errors.append(base_error)
    if not model:
        errors.append("missing_model")
    elif len(model) > 200 or any(character.isspace() for character in model):
        errors.append("invalid_model")
    if not api_version:
        warnings.append("missing_api_version")
    elif len(api_version) > 200 or any(character in api_version for character in "\r\n\x00"):
        errors.append("invalid_api_version")
    settings = (
        _ProviderSettings(api_key, base_url, model, api_version) if not errors else None
    )
    if errors and raise_on_error:
        raise LLMConfigurationError(
            "Invalid real LLM configuration: " + ", ".join(errors)
        )
    return settings, errors, warnings


def _audit_build_id(source: Mapping[str, str] | None = None) -> str:
    environment = source if source is not None else os.environ
    build_id = _first_nonblank(environment, "APP_BUILD_VERSION", "APP_BUILD_ID")
    if not build_id:
        return "unversioned"
    if len(build_id) > 160 or any(character in build_id for character in "\r\n\x00"):
        raise LLMConfigurationError("APP_BUILD_VERSION is invalid")
    return build_id


def check_llm_readiness(
    source: Mapping[str, str] | None = None,
) -> LLMReadiness:
    """Validate real-provider and runtime configuration without network I/O.

    Returned errors are stable categories and never contain an API key, URL or
    other environment value.  A missing API/model version is a warning because
    some providers do not expose immutable revisions, but formal data
    collection should treat that warning as an audit-readiness blocker.
    """

    environment = source if source is not None else os.environ
    settings, errors, warnings = _provider_settings(environment, raise_on_error=False)
    try:
        LLMRuntimeConfig.from_mapping(environment)
    except LLMConfigurationError as exc:
        # Runtime parser messages contain only a variable name and allowed
        # range, never the supplied value.
        errors.append(str(exc))
    try:
        build_id = _audit_build_id(environment)
    except LLMConfigurationError:
        build_id = "invalid"
        errors.append("invalid_build_version")
    if build_id == "unversioned":
        warnings.append("missing_build_version")
    return LLMReadiness(
        ready=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        model=settings.model if settings else (_first_nonblank(environment, "V_MODEL", "LLM_MODEL") or None),
        api_version=(
            settings.api_version
            if settings
            else (_first_nonblank(environment, "V_API_VERSION") or None)
        ),
        build_id=build_id,
    )


@dataclass(repr=False)
class OpenAICompatibleProvider:
    """Real provider backed by the repository's OpenAI-compatible client."""

    base_url: str
    _api_key: str = field(repr=False)
    model: str
    api_version: str | None = None
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        source = {
            "V_API_KEY": self._api_key,
            "V_API_BASE_URL": self.base_url,
            "V_MODEL": self.model,
            "V_API_VERSION": self.api_version or "",
        }
        _settings, errors, _warnings = _provider_settings(source, raise_on_error=False)
        if errors:
            # API version is an audit warning when absent, not a provider
            # execution requirement.  All actual execution errors remain fatal.
            raise LLMConfigurationError(
                "Invalid OpenAI-compatible provider: " + ", ".join(errors)
            )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        settings, _errors, _warnings = _provider_settings(
            os.environ, raise_on_error=True
        )
        if settings is None:  # pragma: no cover - guaranteed by raise_on_error
            raise LLMConfigurationError("Invalid real LLM configuration")
        return cls(
            base_url=settings.base_url,
            _api_key=settings.api_key,
            model=settings.model,
            api_version=settings.api_version,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        try:
            result = chat_completion(
                base_url=self.base_url,
                api_key=self._api_key,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:  # provider library exposes several stdlib errors
            raise classify_provider_exception(exc) from exc
        if not isinstance(result, str) or not result.strip():
            raise LLMProviderError(
                "empty_provider_response",
                "The LLM provider returned no text.",
                retryable=True,
            )
        return result

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleProvider("
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"api_version={self.api_version!r}, name={self.name!r}, api_key=<redacted>)"
        )


class FakeProvider:
    """Finite scripted test double; construction requires explicit test opt-in."""

    name = "fake-test-provider"
    model = "fake-engineering-test-model"
    api_version = "engineering-test-only"
    is_test_double = True

    def __init__(
        self,
        scripted: Sequence[str | BaseException],
        *,
        allow_for_tests: bool = False,
    ):
        if not allow_for_tests:
            raise LLMConfigurationError(
                "FakeProvider is test-only; pass allow_for_tests=True explicitly."
            )
        self._scripted = list(scripted)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if not self._scripted:
            raise AssertionError("FakeProvider has no scripted response remaining")
        item = self._scripted.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def validate_llm_provider(provider: Any) -> LLMProvider:
    """Validate a dependency-injected provider without making a network call."""

    if not isinstance(provider, LLMProvider):
        raise TypeError("provider does not implement LLMProvider")
    if not isinstance(provider.name, str) or not provider.name.strip():
        raise TypeError("provider.name must be a non-empty string")
    if not isinstance(provider.model, str) or not provider.model.strip():
        raise TypeError("provider.model must be a non-empty string")
    if provider.api_version is not None and (
        not isinstance(provider.api_version, str) or not provider.api_version.strip()
    ):
        raise TypeError("provider.api_version must be None or a non-empty string")
    return provider


class LLMService:
    """High-level structured methods used by the application service layer."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        recorder: Recorder | None = None,
        *,
        runtime: LLMRuntimeConfig | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        # This is deliberately fail-closed: missing credentials never select a
        # fake implementation or a hard-coded response.
        selected_provider = (
            provider if provider is not None else OpenAICompatibleProvider.from_env()
        )
        self.provider = validate_llm_provider(selected_provider)
        if recorder is not None and not callable(recorder):
            raise TypeError("recorder must be callable")
        if runtime is not None and not isinstance(runtime, LLMRuntimeConfig):
            raise TypeError("runtime must be LLMRuntimeConfig")
        if not callable(sleep_fn):
            raise TypeError("sleep_fn must be callable")
        self.recorder = recorder
        self.runtime = runtime if runtime is not None else LLMRuntimeConfig.from_env()
        self._sleep = sleep_fn

    def build_profile(self, source: Mapping[str, Any]) -> LLMResult:
        safe = _safe_mapping(source, "source")
        return self._invoke(
            purpose="profile",
            prompt_version=PROFILE_PROMPT_VERSION,
            messages=build_profile_messages(safe),
            input_payload=safe,
            schema_name="profile",
            semantic_check=lambda data: _check_profile(data, safe),
        )

    def recommend(self, profile: Mapping[str, Any]) -> LLMResult:
        safe = _safe_mapping(profile, "profile")
        # A caller cannot pass an arbitrary pseudo-profile into recommendation.
        validate_structured_output("profile", safe)
        return self._invoke(
            purpose="recommendation",
            prompt_version=RECOMMENDATION_PROMPT_VERSION,
            messages=build_recommendation_messages(safe),
            input_payload=safe,
            schema_name="recommendation",
            semantic_check=lambda data: _check_recommendation(data, safe),
        )

    def generate_event(self, context: Mapping[str, Any]) -> LLMResult:
        """Generate a *main-path* narrative.

        Exact and surface anchors must be rendered directly from the frozen
        ``ScheduledEvent`` fields by the application service.  Sending either
        anchor kind through an LLM would introduce construct-irrelevant wording
        variation, so this boundary rejects it before any provider call.
        """

        safe = _safe_mapping(context, "context")
        if safe.get("event_kind") != "main":
            raise ValueError(
                "generate_event accepts only event_kind='main'; render frozen anchors "
                "directly from ScheduledEvent"
            )
        allowed = _allowed_action_codes(safe)
        if not allowed:
            raise ValueError("context.allowed_actions must contain at least one action_code")
        return self._invoke(
            purpose="event_narrative",
            prompt_version=EVENT_PROMPT_VERSION,
            messages=build_event_messages(safe),
            input_payload=safe,
            schema_name="event_narrative",
            semantic_check=lambda data: _check_event(data, safe, allowed),
        )

    def score_decision(self, context: Mapping[str, Any]) -> LLMResult:
        safe = _safe_mapping(context, "context")
        source_context = dict(safe)
        source_context.pop("verbatim_quote_candidates", None)
        source_context.pop("verbatim_quote_candidates_truncated", None)
        response = source_context.get("response")
        if isinstance(response, Mapping):
            clean_response = dict(response)
            clean_response.pop("verbatim_quote_candidates", None)
            clean_response.pop("verbatim_quote_candidates_truncated", None)
            source_context["response"] = clean_response

        allowed = _allowed_action_codes(source_context)
        if not allowed:
            raise ValueError("context.allowed_actions must contain at least one action_code")
        learner_text = _decision_source_text(source_context)
        if not learner_text.strip():
            raise ValueError("context must contain learner decision or rationale text")
        quote_candidates, candidates_truncated = _decision_quote_candidates(source_context)
        candidate_texts = frozenset(item["text"] for item in quote_candidates)
        candidate_text_by_id = {
            item["candidate_id"]: item["text"] for item in quote_candidates
        }
        prompt_context = {
            **source_context,
            "verbatim_quote_candidates": quote_candidates,
            "verbatim_quote_candidates_truncated": candidates_truncated,
        }
        return self._invoke(
            purpose="decision_evidence",
            prompt_version=DECISION_PROMPT_VERSION,
            messages=build_decision_messages(prompt_context),
            input_payload=prompt_context,
            schema_name="decision_evidence",
            semantic_check=lambda data: _check_decision(
                data,
                allowed,
                learner_text,
                candidate_texts,
                candidate_text_by_id,
            ),
            audit_parameters={
                "source_input_hash": _hash_json(source_context),
                "quote_candidates_hash": _hash_json(quote_candidates),
                "quote_candidate_count": len(quote_candidates),
                "quote_candidates_truncated": candidates_truncated,
                "quote_candidate_policy_version": (
                    DECISION_QUOTE_CANDIDATE_POLICY_VERSION
                ),
                "quote_candidate_policy_hash": DECISION_QUOTE_CANDIDATE_POLICY_HASH,
            },
        )

    def _invoke(
        self,
        *,
        purpose: str,
        prompt_version: str,
        messages: list[Message],
        input_payload: Mapping[str, Any],
        schema_name: str,
        semantic_check: Callable[[dict[str, Any]], None],
        audit_parameters: Mapping[str, Any] | None = None,
    ) -> LLMResult:
        invocation_id = f"llm_{uuid.uuid4().hex}"
        actual_prompt_hash = _hash_json(messages)
        prompt_template_hash = _prompt_template_hash(messages, input_payload)
        schema_hash = _hash_json(SCHEMAS[schema_name])
        build_id = _audit_build_id()
        input_hash = _hash_json(input_payload)
        parameters = {
            "temperature": self.runtime.temperature,
            "max_tokens": self.runtime.max_tokens,
            "timeout_seconds": self.runtime.timeout_seconds,
            "max_retries": self.runtime.max_retries,
            # These non-secret audit fields are duplicated in metadata so the
            # existing persistence schema records them immediately inside its
            # parameters JSON, before dedicated columns are added.
            "invocation_id": invocation_id,
            "prompt_template_hash": prompt_template_hash,
            "schema_name": schema_name,
            "schema_hash": schema_hash,
            "build_id": build_id,
            "json_extraction_policy_version": JSON_EXTRACTION_POLICY_VERSION,
            "json_normalizer_version": JSON_NORMALIZER_PROMPT_VERSION,
            "json_normalizer_calls": 0,
            "json_normalizer_input_truncated": False,
        }
        if purpose == "decision_evidence":
            parameters["repair_policy_version"] = DECISION_REPAIR_POLICY_VERSION
        if audit_parameters:
            overlap = set(parameters).intersection(audit_parameters)
            if overlap:
                raise LLMConfigurationError(
                    "Audit parameters overlap reserved metadata keys"
                )
            parameters.update(dict(audit_parameters))
        started = time.monotonic()
        attempt_messages = list(messages)
        attempts = 0
        provider_calls = 0
        normalizer_used = False
        final_code = "unknown"
        final_message = "LLM request failed."

        for attempt in range(1, self.runtime.max_retries + 2):
            # The exact audit hash must describe the messages actually sent on
            # this attempt, including the repair instruction added after a
            # structured-output failure.
            actual_prompt_hash = _hash_json(attempt_messages)
            try:
                provider_calls += 1
                attempts = provider_calls
                raw = self.provider.complete(
                    attempt_messages,
                    temperature=self.runtime.temperature,
                    max_tokens=self.runtime.max_tokens,
                    timeout=self.runtime.timeout_seconds,
                )
                try:
                    parsed = extract_json_object(raw)
                except StructuredOutputError as extraction_error:
                    if extraction_error.code != "invalid_json" or normalizer_used:
                        raise
                    normalizer_used = True
                    normalizer_input = raw[:_JSON_NORMALIZER_MAX_INPUT_CHARS]
                    parameters["json_normalizer_calls"] = 1
                    parameters["json_normalizer_input_truncated"] = (
                        len(normalizer_input) != len(raw)
                    )
                    normalizer_messages = build_json_normalizer_messages(
                        normalizer_input,
                        SCHEMAS[schema_name],
                    )
                    actual_prompt_hash = _hash_json(normalizer_messages)
                    provider_calls += 1
                    attempts = provider_calls
                    normalized_raw = self.provider.complete(
                        normalizer_messages,
                        temperature=0.0,
                        max_tokens=self.runtime.max_tokens,
                        timeout=self.runtime.timeout_seconds,
                    )
                    parsed = extract_json_object(normalized_raw)
                data = validate_structured_output(schema_name, parsed)
                # Do not persist or display newly hallucinated direct identifiers.
                assert_no_sensitive_data(data)
                semantic_check(data)
                # Decision quote IDs are expanded during the semantic check;
                # validate the final object that will actually be persisted.
                assert_no_sensitive_data(data)
            except (StructuredOutputError, SchemaValidationError, SensitiveDataError) as exc:
                final_code = getattr(exc, "code", "invalid_structure")
                final_message = "The LLM response did not satisfy the required structure."
                if attempt > self.runtime.max_retries:
                    break
                attempt_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": _repair_instruction(schema_name, final_code),
                    },
                ]
                self._backoff(attempt)
                continue
            except Exception as exc:
                provider_error = (
                    exc if isinstance(exc, LLMProviderError) else classify_provider_exception(exc)
                )
                final_code = provider_error.code
                final_message = str(provider_error)
                if not provider_error.retryable or attempt > self.runtime.max_retries:
                    break
                self._backoff(attempt)
                continue

            metadata = self._metadata(
                invocation_id=invocation_id,
                purpose=purpose,
                prompt_version=prompt_version,
                prompt_hash=actual_prompt_hash,
                prompt_template_hash=prompt_template_hash,
                schema_name=schema_name,
                schema_hash=schema_hash,
                build_id=build_id,
                input_hash=input_hash,
                parameters=parameters,
                attempts=attempts,
                started=started,
            )
            result = LLMResult(data=data, metadata=metadata)
            self._record(
                {
                    "status": "success",
                    "metadata": metadata.to_dict(),
                    "output": data,
                    "error_code": None,
                }
            )
            return result

        metadata = self._metadata(
            invocation_id=invocation_id,
            purpose=purpose,
            prompt_version=prompt_version,
            prompt_hash=actual_prompt_hash,
            prompt_template_hash=prompt_template_hash,
            schema_name=schema_name,
            schema_hash=schema_hash,
            build_id=build_id,
            input_hash=input_hash,
            parameters=parameters,
            attempts=attempts,
            started=started,
        )
        self._record(
            {
                "status": "failed",
                "metadata": metadata.to_dict(),
                "output": None,
                "error_code": final_code,
            }
        )
        raise LLMRequestFailed(
            final_code,
            final_message,
            attempts=attempts,
            metadata=metadata,
        )

    def _backoff(self, attempt: int) -> None:
        delay = self.runtime.retry_base_seconds * (2 ** (attempt - 1))
        if delay:
            self._sleep(delay)

    def _metadata(
        self,
        *,
        invocation_id: str,
        purpose: str,
        prompt_version: str,
        prompt_hash: str,
        prompt_template_hash: str,
        schema_name: str,
        schema_hash: str,
        build_id: str,
        input_hash: str,
        parameters: dict[str, Any],
        attempts: int,
        started: float,
    ) -> LLMCallMetadata:
        return LLMCallMetadata(
            invocation_id=invocation_id,
            purpose=purpose,
            provider=self.provider.name,
            model=self.provider.model,
            api_version=self.provider.api_version,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            prompt_template_hash=prompt_template_hash,
            schema_name=schema_name,
            schema_hash=schema_hash,
            build_id=build_id,
            input_hash=input_hash,
            parameters=dict(parameters),
            attempts=attempts,
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        )

    def _record(self, record: dict[str, Any]) -> None:
        if self.recorder is not None:
            try:
                self.recorder(record)
            except Exception:
                metadata = record.get("metadata", {})
                invocation_id = str(metadata.get("invocation_id", "unknown"))
                raise LLMRecorderError(invocation_id) from None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object using regex candidates plus strict decoding."""

    if not isinstance(text, str) or not text.strip():
        raise StructuredOutputError("empty_output", "The model returned no text.")
    source = text.strip().lstrip("\ufeff")
    candidates = [source]
    for pattern in (_JSON_FENCE_RE, _JSON_TAG_RE, _JSON_LABEL_RE, _JSON_OBJECT_RE):
        candidates.extend(match.group(1).strip() for match in pattern.finditer(source))
    candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except (JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict):
            return decoded
        for position, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                decoded, _end = decoder.raw_decode(candidate[position:])
            except JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    raise StructuredOutputError("invalid_json", "No JSON object was found.")


def classify_provider_exception(exc: BaseException) -> LLMProviderError:
    """Map provider/stdlib errors to non-sensitive retry policy categories."""

    if isinstance(exc, LLMProviderError):
        return exc
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return LLMProviderError("timeout", "The LLM request timed out.", retryable=True)
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        return LLMProviderError(
            "network_error", "The LLM provider could not be reached.", retryable=True
        )
    if isinstance(exc, (KeyError, IndexError, JSONDecodeError, TypeError)):
        return LLMProviderError(
            "invalid_provider_response",
            "The provider response envelope was invalid.",
            retryable=True,
        )

    # The supplied low-level client converts HTTP errors to RuntimeError.  Only
    # inspect the status code; never propagate its response body.
    text = str(exc)
    status_match = re.search(r"\bHTTP\s+(\d{3})\b", text, re.I)
    if status_match:
        status = int(status_match.group(1))
        if status == 429:
            return LLMProviderError(
                "rate_limited", "The LLM provider rate-limited the request.", retryable=True
            )
        if status in {408, 409, 425} or 500 <= status <= 599:
            return LLMProviderError(
                "provider_unavailable",
                "The LLM provider is temporarily unavailable.",
                retryable=True,
            )
        if status in {401, 403}:
            return LLMProviderError(
                "authentication_failed",
                "The LLM provider rejected its credentials.",
                retryable=False,
            )
        return LLMProviderError(
            "request_rejected", "The LLM provider rejected the request.", retryable=False
        )
    lowered = text.casefold()
    if "timeout" in lowered or "timed out" in lowered:
        return LLMProviderError("timeout", "The LLM request timed out.", retryable=True)
    return LLMProviderError(
        "provider_error", "The LLM provider call failed.", retryable=False
    )


def _safe_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    safe = sanitize_for_llm(value)
    if not isinstance(safe, dict):  # defensive; Mapping always sanitizes to dict
        raise TypeError(f"{name} must sanitize to an object")
    return safe


def _check_profile(data: dict[str, Any], source: Mapping[str, Any]) -> None:
    trait_ids: set[str] = set()
    for trait in data["traits"]:
        if "original_claim" in trait or "learner_edited" in trait:
            raise StructuredOutputError(
                "unexpected_profile_provenance",
                "Initial model output cannot set learner-edit provenance fields",
            )
        trait_id = trait["trait_id"]
        if trait_id in trait_ids:
            raise StructuredOutputError("duplicate_trait_id", "trait_id values must be unique")
        trait_ids.add(trait_id)
        if trait["conflict"] and not trait["conflict_detail"]:
            raise StructuredOutputError(
                "missing_conflict_detail", "Conflicting evidence requires an explanation"
            )
        for evidence in trait["evidence"]:
            quote = evidence["quote"]
            source_text = _profile_source_field_text(
                source, str(evidence["source_field"])
            )
            if _DECISION_REDACTION_MARKER_RE.search(quote) or not source_text or quote not in source_text:
                raise StructuredOutputError(
                    "unsupported_profile_quote",
                    "Profile evidence must quote its declared sanitized source field",
                )
    mbti = data["mbti_usage"]
    if mbti["used"] and not _contains_mbti_input(source):
        raise StructuredOutputError("invented_mbti", "MBTI use was not supported by input")
    if mbti["used"] and mbti["weight"] != "low":
        raise StructuredOutputError("mbti_weight", "Used MBTI must have low weight")
    if not mbti["used"] and mbti["weight"] != "none":
        raise StructuredOutputError("mbti_weight", "Unused MBTI must have no weight")


def _check_recommendation(data: dict[str, Any], profile: Mapping[str, Any]) -> None:
    trait_ids = {item["trait_id"] for item in profile["traits"]}
    direction_ids: set[str] = set()
    for recommendation in data["recommendations"]:
        direction_id = recommendation["direction_id"]
        if direction_id in direction_ids:
            raise StructuredOutputError(
                "duplicate_direction_id", "direction_id values must be unique"
            )
        direction_ids.add(direction_id)
        unknown = set(recommendation["profile_evidence_ids"]) - trait_ids
        if unknown:
            raise StructuredOutputError(
                "unknown_profile_evidence", "Recommendation references unknown trait_id"
            )
        if (
            not profile["mbti_usage"]["used"]
            and recommendation["mbti_influence"] != "none"
        ):
            raise StructuredOutputError(
                "unsupported_mbti_influence", "Recommendation used MBTI that was not provided"
            )


def _check_event(
    data: dict[str, Any], context: Mapping[str, Any], allowed: set[str]
) -> None:
    returned = set(data["allowed_action_codes"])
    if returned != allowed:
        raise StructuredOutputError(
            "action_whitelist_mismatch", "Event action codes must exactly match the blueprint"
        )
    target_dimensions = {
        item for item in context.get("target_dimensions", []) if item in ECD_DIMENSIONS
    }
    if target_dimensions and set(data["learning_focus"]) != target_dimensions:
        raise StructuredOutputError(
            "learning_focus_mismatch", "Event learning focus must match its blueprint"
        )


def _check_decision(
    data: dict[str, Any],
    allowed: set[str],
    learner_text: str,
    quote_candidates: frozenset[str],
    quote_candidate_by_id: Mapping[str, str],
) -> None:
    if data["action_code"] not in allowed:
        raise StructuredOutputError(
            "unknown_action_code", "Decision maps outside the blueprint action whitelist"
        )
    if not set(data["mapping_uncertainty"]["alternative_action_codes"]).issubset(allowed):
        raise StructuredOutputError(
            "unknown_alternative_action", "Alternative action is not in the whitelist"
        )
    dimensions = [item["dimension"] for item in data["evidence"]]
    if set(dimensions) != set(ECD_DIMENSIONS) or len(set(dimensions)) != len(ECD_DIMENSIONS):
        raise StructuredOutputError(
            "ecd_dimensions", "Evidence must cover each ECD dimension exactly once"
        )
    for item in data["evidence"]:
        resolved_quotes: list[str] = []
        for quote in item["quotes"]:
            if quote in quote_candidates:
                resolved_quotes.append(quote)
                continue
            candidate_text = quote_candidate_by_id.get(quote)
            if candidate_text is not None:
                resolved_quotes.append(candidate_text)
                continue
            if _DECISION_QUOTE_CANDIDATE_ID_RE.fullmatch(quote):
                raise StructuredOutputError(
                    "unknown_decision_quote_candidate_id",
                    "Evidence quote references an unknown candidate_id",
                )
            resolved_quotes.append(quote)
        item["quotes"] = resolved_quotes

        if item["evidence_status"] in {"supported", "partial"} and not item["quotes"]:
            raise StructuredOutputError(
                "missing_decision_quote", "Supported evidence requires a learner quote"
            )
        if item["evidence_status"] == "insufficient" and item["quotes"]:
            raise StructuredOutputError(
                "unexpected_decision_quote", "Insufficient evidence cannot cite support"
            )
        if item["evidence_status"] == "insufficient" and item["score"] != 0:
            raise StructuredOutputError(
                "unsupported_nonzero_score",
                "Insufficient evidence must receive score 0",
            )
        for quote in item["quotes"]:
            if _DECISION_REDACTION_MARKER_RE.search(quote) or quote not in learner_text:
                raise StructuredOutputError(
                    "unsupported_decision_quote", "Evidence quote is not in learner text"
                )
            if quote not in quote_candidates:
                raise StructuredOutputError(
                    "unlisted_decision_quote_candidate",
                    "Evidence quote is not an exact verbatim candidate",
                )
    flags = set(data["safety_flags"])
    if "none" in flags and len(flags) > 1:
        raise StructuredOutputError(
            "contradictory_safety_flags", "none cannot accompany another safety flag"
        )


def _allowed_action_codes(context: Mapping[str, Any]) -> set[str]:
    raw = context.get("allowed_actions", context.get("allowed_action_codes", []))
    codes: set[str] = set()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, str):
                code = item
            elif isinstance(item, Mapping):
                code = item.get("action_code")
            else:
                code = None
            if isinstance(code, str) and code:
                codes.add(code)
    return codes


def _flatten_strings_with_paths(value: Any, path: str) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, Mapping):
        output: list[tuple[str, str]] = []
        for key in sorted(value, key=str):
            output.extend(_flatten_strings_with_paths(value[key], f"{path}.{key}"))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output = []
        for index, item in enumerate(value):
            output.extend(_flatten_strings_with_paths(item, f"{path}[{index}]"))
        return output
    return []


def _decision_source_entries(
    context: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    """Return allowed decision strings with stable field and provenance paths."""

    containers: list[tuple[str, Mapping[str, Any]]] = []
    response = context.get("response")
    if isinstance(response, Mapping):
        containers.append(("response", response))
    containers.append(("", context))

    selected: list[tuple[str, str, str]] = []
    for prefix, container in containers:
        for field_name in _DECISION_SOURCE_FIELDS:
            if field_name not in container:
                continue
            field_path = f"{prefix}.{field_name}" if prefix else field_name
            for value_path, text in _flatten_strings_with_paths(
                container.get(field_name), field_path
            ):
                selected.append((field_name, value_path, text))
    return selected


def _decision_source_text(context: Mapping[str, Any]) -> str:
    return "\n".join(text for _field, _path, text in _decision_source_entries(context))


def _split_exact_segments(
    text: str,
    boundary_re: re.Pattern[str],
    *,
    keep_boundary_variant: bool,
) -> list[str]:
    """Split without normalising, returning only literal source substrings."""

    output: list[str] = []
    cursor = 0
    for match in boundary_re.finditer(text):
        raw_core = text[cursor : match.start()]
        core = raw_core.strip()
        if core:
            output.append(core)
            if keep_boundary_variant and match.lastgroup == "boundary":
                full = text[cursor : match.end()].strip()
                if full and full != core:
                    output.append(full)
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        output.append(tail)
    return output


def _fixed_quote_windows(text: str) -> list[str]:
    if len(text) <= _DECISION_QUOTE_MAX_CHARS:
        return [text]
    stride = _DECISION_QUOTE_MAX_CHARS - _DECISION_QUOTE_WINDOW_OVERLAP
    output: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + _DECISION_QUOTE_MAX_CHARS)
        window = text[start:end].strip()
        if window:
            output.append(window)
        if end == len(text):
            break
        start += stride
    return output


def _bounded_quote_fragments(text: str) -> list[str]:
    if len(text) <= _DECISION_QUOTE_MAX_CHARS:
        return [text]
    soft_segments = _split_exact_segments(
        text,
        _DECISION_QUOTE_SOFT_BOUNDARY_RE,
        keep_boundary_variant=True,
    )
    output: list[str] = []
    for segment in soft_segments:
        output.extend(_fixed_quote_windows(segment))
    return output


def _decision_quote_candidates(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, str]], bool]:
    """Build a deterministic, label-blind exact-quote menu from safe input."""

    by_field: dict[str, list[dict[str, str]]] = {
        field_name: [] for field_name in _DECISION_SOURCE_FIELDS
    }
    seen: set[tuple[str, str]] = set()
    for field_name, source_path, source_text in _decision_source_entries(context):
        strong_segments = _split_exact_segments(
            source_text,
            _DECISION_QUOTE_STRONG_OR_REDACTION_RE,
            keep_boundary_variant=True,
        )
        for strong_segment in strong_segments:
            for candidate_text in _bounded_quote_fragments(strong_segment):
                if (
                    not candidate_text
                    or len(candidate_text) > _DECISION_QUOTE_MAX_CHARS
                    or _DECISION_REDACTION_MARKER_RE.search(candidate_text)
                ):
                    continue
                key = (source_path, candidate_text)
                if key in seen:
                    continue
                seen.add(key)
                by_field[field_name].append(
                    {"source_field": source_path, "text": candidate_text}
                )

    selected: list[dict[str, str]] = []
    field_offsets = {field_name: 0 for field_name in _DECISION_SOURCE_FIELDS}
    while len(selected) < _DECISION_QUOTE_MAX_TOTAL:
        advanced = False
        for field_name in _DECISION_SOURCE_FIELDS:
            offset = field_offsets[field_name]
            values = by_field[field_name]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            field_offsets[field_name] = offset + 1
            advanced = True
            if len(selected) >= _DECISION_QUOTE_MAX_TOTAL:
                break
        if not advanced:
            break

    total_candidates = sum(len(values) for values in by_field.values())
    candidate_texts = {candidate["text"] for candidate in selected}
    candidate_ids: set[str] = set()
    candidates: list[dict[str, str]] = []
    for index, candidate in enumerate(selected, start=1):
        candidate_id = f"q{index:03d}"
        while candidate_id in candidate_texts or candidate_id in candidate_ids:
            candidate_id = "_" + candidate_id
        candidate_ids.add(candidate_id)
        candidates.append({"candidate_id": candidate_id, **candidate})
    return candidates, total_candidates > len(candidates)


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        output: list[str] = []
        for item in value.values():
            output.extend(_flatten_strings(item))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    return []


def _profile_source_field_text(
    source: Mapping[str, Any], source_field: str
) -> str:
    """Resolve a declared profile provenance path without fuzzy matching.

    The model must name the real input field that contains its quotation.  A
    quote appearing somewhere else in the profile is insufficient provenance.
    Dot-separated paths support the small nested objects used for optional
    preference notes while rejecting empty or non-object traversal.
    """

    parts = source_field.split(".")
    if not parts or any(not part for part in parts):
        return ""
    current: Any = source
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return ""
        current = current[part]
    return "\n".join(_flatten_strings(current))


def _contains_mbti_input(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[\s_.-]+", "", str(key)).casefold()
            if normalized in {"mbti", "mbtitype", "\u8fc8\u5c14\u65af\u5e03\u91cc\u683c\u65af", "\u4eba\u683c\u504f\u597d"}:
                return True
            if _contains_mbti_input(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_mbti_input(item) for item in value)
    if isinstance(value, str):
        return bool(
            re.search(
                r"(?i)\b(?:ISTJ|ISFJ|INFJ|INTJ|ISTP|ISFP|INFP|INTP|"
                r"ESTP|ESFP|ENFP|ENTP|ESTJ|ESFJ|ENFJ|ENTJ)\b",
                value,
            )
        )
    return False


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_template_hash(
    messages: Sequence[Message], input_payload: Mapping[str, Any]
) -> str:
    """Hash the prompt/schema template after replacing learner data.

    ``prompt_hash`` remains the fingerprint of the exact sent messages.  This
    second hash is stable across learners and therefore reveals prompt/schema
    drift without storing a prompt or any learner text.
    """

    serialized = json.dumps(
        input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    exact_block = (
        "<untrusted_input_json>\n" + serialized + "\n</untrusted_input_json>"
    )
    replacement = "<untrusted_input_json>\n[DATA_MINIMIZED_PAYLOAD]\n</untrusted_input_json>"
    template: list[Message] = []
    replaced = False
    for message in messages:
        content = message["content"]
        if not replaced and exact_block in content:
            content = content.replace(exact_block, replacement, 1)
            replaced = True
        template.append({"role": message["role"], "content": content})
    if not replaced:
        # Failing closed avoids accidentally calling a learner-specific hash a
        # template hash if a future prompt builder changes its delimiters.
        raise LLMConfigurationError("Prompt builder did not expose its data boundary")
    return _hash_json(template)


def _validate_runtime_value(
    name: str,
    value: Any,
    expected_type: type | tuple[type, ...],
    *,
    minimum: float,
    maximum: float,
) -> None:
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise LLMConfigurationError(f"{name} has an invalid type")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise LLMConfigurationError(f"{name} must be between {minimum} and {maximum}")


def _mapping_int(
    source: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(name)
    try:
        value = default if raw is None or not str(raw).strip() else int(str(raw).strip())
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise LLMConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _mapping_float(
    source: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = source.get(name)
    try:
        value = default if raw is None or not str(raw).strip() else float(str(raw).strip())
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise LLMConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


__all__ = [
    "DECISION_QUOTE_CANDIDATE_POLICY_HASH",
    "DECISION_QUOTE_CANDIDATE_POLICY_VERSION",
    "DECISION_REPAIR_POLICY_VERSION",
    "JSON_EXTRACTION_POLICY_VERSION",
    "FakeProvider",
    "LLMCallMetadata",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderError",
    "LLMReadiness",
    "LLMRecorderError",
    "LLMRequestFailed",
    "LLMResult",
    "LLMRuntimeConfig",
    "LLMService",
    "OpenAICompatibleProvider",
    "StructuredOutputError",
    "classify_provider_exception",
    "check_llm_readiness",
    "extract_json_object",
    "validate_llm_provider",
]
