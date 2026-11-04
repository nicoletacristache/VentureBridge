"""Data minimisation and deterministic redaction before external API calls.

This module intentionally does not attempt to infer sensitive attributes.  It
removes direct identifiers that can be detected reliably and entire values of
known identifier fields.  The application should still collect only the
minimum data needed for the teaching task.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SensitiveFinding:
    kind: str
    path: str


class SensitiveDataError(ValueError):
    """Raised when detectable direct identifiers remain in an API payload."""

    def __init__(self, findings: Sequence[SensitiveFinding]):
        self.findings = tuple(findings)
        summary = ", ".join(f"{item.kind}@{item.path}" for item in findings[:8])
        super().__init__(f"Detectable sensitive data remains in payload: {summary}")


# More specific patterns are applied before generic numeric patterns.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "api_credential",
        re.compile(r"(?i)\b(?:sk|api)[-_][A-Za-z0-9_-]{16,}\b"),
    ),
    (
        "email",
        re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"),
    ),
    (
        "url",
        re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'，。；;]+"),
    ),
    (
        "prc_identity_number",
        re.compile(
            r"(?<!\d)(?:[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]|[1-9]\d{14})(?!\d)"
        ),
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d(?:[-\s]?\d){8}(?!\d)"),
    ),
    (
        "telephone",
        re.compile(r"(?<!\d)(?:\+?86[-\s]?)?0\d{2,3}[-\s]?\d{7,8}(?!\d)"),
    ),
    (
        "bank_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)"),
    ),
    (
        "ipv4",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
    ),
    (
        "coordinates",
        re.compile(
            r"(?<!\d)(?:[-+]?\d{1,2}\.\d{4,})\s*[,，]\s*"
            r"(?:[-+]?\d{1,3}\.\d{4,})(?!\d)"
        ),
    ),
    (
        "precise_address_zh",
        re.compile(
            r"(?:[\u4e00-\u9fff]{2,}(?:省|自治区|市|自治州|县|区)){0,4}"
            r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:大道|大街|街道|路|街|巷|弄)"
            r"[\u4e00-\u9fffA-Za-z0-9-]{0,24}(?:号|栋|幢|座|单元|室)"
            r"[\u4e00-\u9fffA-Za-z0-9-]{0,12}"
        ),
    ),
    (
        "precise_address_en",
        re.compile(
            r"(?i)\b\d{1,6}\s+[A-Z0-9][A-Z0-9 .'-]{1,50}\s+"
            r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|boulevard|blvd\.?)\b"
        ),
    ),
)


_DIRECT_IDENTIFIER_KEYS = {
    "name",
    "fullname",
    "realname",
    "legalname",
    "alias",
    "email",
    "emailaddress",
    "phone",
    "phonenumber",
    "mobile",
    "telephone",
    "wechat",
    "weixin",
    "qq",
    "id",
    "idnumber",
    "identitynumber",
    "nationalid",
    "passport",
    "address",
    "homeaddress",
    "exactaddress",
    "url",
    "website",
    "school",
    "university",
    "employer",
    "company",
    "organization",
    "organisation",
    "workplace",
    "apikey",
    "accesskey",
    "secret",
    "token",
    # Internal/pseudonymous record identifiers are not direct real-world
    # identifiers, but an external model never needs them.  Keeping these
    # keys at the privacy boundary prevents a future caller from accidentally
    # sending database or allocation linkage keys in an otherwise safe
    # payload.  Content identifiers such as trait_id, blueprint_id and form_id
    # are deliberately not included because they are required for auditable
    # structured tasks and do not identify a learner.
    "userid",
    "workspaceid",
    "profileid",
    "recommendationid",
    "simulationid",
    "deliveryid",
    "responseid",
    "ratingid",
    "callid",
    "sourcecallid",
    "allocationunitid",
    "participantid",
    "learnerid",
    "subjectid",
    "recordid",
    "姓名",
    "真实姓名",
    "邮箱",
    "电子邮箱",
    "电话",
    "手机号",
    "手机号码",
    "身份证",
    "身份证号",
    "护照",
    "地址",
    "详细地址",
    "微信",
    "学校",
    "单位",
    "公司",
    "机构",
    "网址",
    "密钥",
}


def _normalize_key(key: str) -> str:
    return re.sub(r"[\s_.-]+", "", key).casefold()


def detect_sensitive_text(text: str) -> tuple[str, ...]:
    """Return detected identifier kinds without returning the identifier text."""

    return tuple(name for name, pattern in _PATTERNS if pattern.search(text))


def redact_text(text: str) -> str:
    """Replace detectable identifiers with labelled non-reversible markers."""

    redacted = text
    for name, pattern in _PATTERNS:
        redacted = pattern.sub(f"[已移除:{name}]", redacted)
    return redacted


def sanitize_for_llm(value: Any) -> Any:
    """Return a JSON-compatible, data-minimised copy safe for prompt building.

    Entire values of explicit identifier fields are removed even when the value
    does not match a regular expression.  Unknown object types are rejected so
    a custom ``repr`` cannot accidentally place secrets in a prompt.
    """

    sanitized = _sanitize(value, path="$", depth=0)
    assert_no_sensitive_data(sanitized)
    return sanitized


def _sanitize(value: Any, *, path: str, depth: int) -> Any:
    if depth > 20:
        raise ValueError(f"Payload nesting is too deep at {path}")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if not isinstance(raw_key, str):
                raise TypeError(f"Payload key at {path} must be a string")
            key = raw_key
            if detect_sensitive_text(key):
                key = f"redacted_key_{index}"
            item_path = f"{path}.{key}"
            if _normalize_key(key) in _DIRECT_IDENTIFIER_KEYS:
                output[key] = "[已移除:direct_identifier_field]"
            else:
                output[key] = _sanitize(item, path=item_path, depth=depth + 1)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _sanitize(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise TypeError(f"Unsupported payload type at {path}: {type(value).__name__}")


def find_sensitive_data(value: Any, *, path: str = "$") -> list[SensitiveFinding]:
    """Scan a JSON-like value, reporting paths but never returning secret text."""

    findings: list[SensitiveFinding] = []
    if isinstance(value, str):
        findings.extend(SensitiveFinding(kind, path) for kind in detect_sensitive_text(value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            findings.extend(
                SensitiveFinding(kind, f"{path}.<key>")
                for kind in detect_sensitive_text(key_text)
            )
            findings.extend(find_sensitive_data(item, path=f"{path}.{key_text}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            findings.extend(find_sensitive_data(item, path=f"{path}[{index}]"))
    return findings


def assert_no_sensitive_data(value: Any) -> None:
    """Fail closed if a supported direct identifier pattern remains."""

    findings = find_sensitive_data(value)
    if findings:
        raise SensitiveDataError(findings)


__all__ = [
    "SensitiveDataError",
    "SensitiveFinding",
    "assert_no_sensitive_data",
    "detect_sensitive_text",
    "find_sensitive_data",
    "redact_text",
    "sanitize_for_llm",
]
