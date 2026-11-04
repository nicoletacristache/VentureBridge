#!/usr/bin/env python3
"""Run the rule and structured LLM ECD encoders.

See ../PROVENANCE.md for data source, evaluation assumptions, and scope.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from venturebridge.llm import (  # noqa: E402
    DECISION_QUOTE_CANDIDATE_POLICY_HASH,
    DECISION_QUOTE_CANDIDATE_POLICY_VERSION,
    DECISION_REPAIR_POLICY_VERSION,
    LLMRequestFailed,
    LLMRuntimeConfig,
    LLMService,
    OpenAICompatibleProvider,
)
from venturebridge.prompts import DECISION_PROMPT_VERSION  # noqa: E402
from venturebridge.schemas import (  # noqa: E402
    DECISION_EVIDENCE_SCHEMA,
    ECD_DIMENSIONS,
    validate_structured_output,
)


EXPECTED_PROMPT_VERSION = "decision-evidence-2026-10-v6"
QWEN_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://run.v3.cm/v1"
RULE_VERSION = "ecd-keyword-structure-rule-2026-10-v1"
STATE_CHANGE_NOTICE = "数值状态仅由规则引擎依据 action_code 更新，LLM 未计算或修改状态。"

DEFAULT_INPUT = WORKSPACE_ROOT / "method_validation" / "data" / "ecd_audit_items.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "method_validation" / "predictions"

ID_ALIASES = ("audit_id", "sample_id", "item_id", "response_id", "case_id", "id")
TEXT_ALIASES = (
    "decision_text",
    "response_text",
    "text",
    "learner_text",
    "synthetic_text",
    "response",
)
TARGET_DIMENSION_ALIASES = ("target_dimension", "ecd_dimension", "dimension")
RATIONALE_ALIASES = ("rationale_text", "rationale", "reasoning")
INFORMATION_ALIASES = (
    "information_requests",
    "information_request_text",
    "information_request",
    "requested_information",
)
REVISION_ALIASES = (
    "revision_text",
    "revision_condition_text",
    "revision_condition",
    "revision",
    "update_text",
)
EVENT_ALIASES = ("scenario_text", "event_text", "scenario", "prompt_text", "case_text")
ALLOWED_ACTIONS_ALIASES = ("allowed_actions", "allowed_action_codes")

DIMENSION_ALIASES = {
    "opportunity_framing": "opportunity_framing",
    "机会界定": "opportunity_framing",
    "机会界定与评估": "opportunity_framing",
    "evidence_falsifiability": "evidence_falsifiability",
    "证据权衡": "evidence_falsifiability",
    "证据权衡与可证伪性": "evidence_falsifiability",
    "resource_finance": "resource_finance",
    "资源财务": "resource_finance",
    "资源/财务配置": "resource_finance",
    "adaptive_action": "adaptive_action",
    "适应更新": "adaptive_action",
    "行动规划与适应更新": "adaptive_action",
    "stakeholder_ethics": "stakeholder_ethics",
    "利益相关者伦理": "stakeholder_ethics",
    "团队—利益相关者—伦理/合规权衡": "stakeholder_ethics",
}

# Each named group represents one distinct observable concept.  Several
# synonyms in one group count only once, preventing keyword repetition from
# inflating a score.  The complete rule is intentionally kept here rather than
# hidden in a fitted artifact so every prediction is auditable.
RULE_DIMENSION_GROUPS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "opportunity_framing": (
        ("actor", ("目标用户", "目标客户", "用户", "客户", "顾客", "受众")),
        ("problem", ("痛点", "需求", "问题", "困难", "待办任务", "使用场景")),
        ("segment", ("细分", "人群", "场景差异", "早期用户", "客户画像")),
        ("value", ("价值主张", "价值", "收益", "替代方案", "现有方案")),
        ("hypothesis", ("机会假设", "价值假设", "需求假设", "假设", "机会")),
    ),
    "evidence_falsifiability": (
        ("evidence", ("证据", "数据", "记录", "样本", "访谈", "观察")),
        ("test", ("验证", "测试", "实验", "对照", "试验", "预售")),
        ("metric", ("指标", "阈值", "转化率", "留存率", "成功标准", "判定标准")),
        ("counterevidence", ("反证", "证伪", "拒绝原因", "失败信号", "负面证据")),
        ("source_quality", ("证据质量", "偏差", "代表性", "可信度", "样本量", "来源")),
    ),
    "resource_finance": (
        ("money", ("预算", "成本", "现金", "资金", "现金流", "费用", "元")),
        ("people", ("人力", "人员", "工时", "团队产能", "负责人")),
        ("time", ("时间", "周期", "期限", "周", "天")),
        ("allocation", ("分配", "配置", "优先投入", "资源", "额度")),
        ("tradeoff", ("机会成本", "取舍", "权衡", "生存期", "盈亏", "回本")),
    ),
    "adaptive_action": (
        ("small_step", ("小步", "试点", "最小可行", "可逆", "低成本", "先做")),
        ("sequence", ("然后", "再根据", "分阶段", "下一步", "里程碑", "先后")),
        ("update", ("调整", "修订", "迭代", "更新", "改变方案", "复盘")),
        ("contingency", ("备选", "预案", "否则", "止损", "退出", "暂停")),
        ("learning_loop", ("反馈", "结果后", "根据结果", "学到", "重新评估")),
    ),
    "stakeholder_ethics": (
        ("team", ("团队", "合伙人", "成员", "分工", "职责", "员工")),
        ("stakeholder", ("利益相关者", "供应商", "合作方", "投资人", "社区")),
        ("ethics", ("伦理", "公平", "歧视", "透明", "知情同意", "伤害")),
        ("compliance", ("合规", "法律", "监管", "许可", "合同", "审查")),
        ("protection", ("隐私", "数据保护", "安全", "申诉", "退出权", "利益冲突")),
    ),
}

STRUCTURAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("conditional", re.compile(r"(?:如果|若|一旦|只有).{0,40}(?:则|就|才|会)|否则")),
    ("quantified", re.compile(r"(?:\d+(?:\.\d+)?\s*(?:%|％|元|人|次|天|周|月|小时)|百分之[一二三四五六七八九十百]+)")),
    ("sequence", re.compile(r"(?:首先|先).{0,80}(?:然后|再|之后|其次)|(?:第一|第二|第三)[，、:]")),
    ("tradeoff", re.compile(r"(?:权衡|取舍|相比|但|然而|同时考虑|机会成本|成本收益)")),
    ("revision", re.compile(r"(?:根据|依据).{0,40}(?:结果|证据|反馈).{0,30}(?:调整|修订|更新|迭代|停止|放弃)")),
)


@dataclass(frozen=True)
class ColumnMap:
    sample_id: str | None
    text: str
    target_dimension: str | None
    rationale: str | None
    information: str | None
    revision: str | None
    event: str | None
    allowed_actions: str | None

    def public_dict(self) -> dict[str, str | None]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "target_dimension": self.target_dimension,
            "rationale": self.rationale,
            "information": self.information,
            "revision": self.revision,
            "event": self.event,
            "allowed_actions": self.allowed_actions,
        }


@dataclass(frozen=True)
class AuditItem:
    sample_id: str
    source_row: int
    target_dimension: str | None
    context: dict[str, Any]
    item_hash: str


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def script_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def resolve_column(
    fieldnames: Sequence[str],
    explicit: str | None,
    aliases: Sequence[str],
    *,
    required: bool,
) -> str | None:
    lookup = {name.strip().casefold(): name for name in fieldnames if name is not None}
    if explicit:
        resolved = lookup.get(explicit.strip().casefold())
        if resolved is None:
            raise ValueError(f"CSV column {explicit!r} was requested but is absent")
        return resolved
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    if required:
        raise ValueError(
            "No compatible text column was found; use --text-column. "
            f"Accepted defaults: {', '.join(aliases)}"
        )
    return None


def resolve_columns(fieldnames: Sequence[str], args: argparse.Namespace) -> ColumnMap:
    return ColumnMap(
        sample_id=resolve_column(fieldnames, args.id_column, ID_ALIASES, required=False),
        text=resolve_column(fieldnames, args.text_column, TEXT_ALIASES, required=True) or "",
        target_dimension=resolve_column(
            fieldnames,
            args.target_dimension_column,
            TARGET_DIMENSION_ALIASES,
            required=False,
        ),
        rationale=resolve_column(
            fieldnames, args.rationale_column, RATIONALE_ALIASES, required=False
        ),
        information=resolve_column(
            fieldnames, args.information_column, INFORMATION_ALIASES, required=False
        ),
        revision=resolve_column(
            fieldnames, args.revision_column, REVISION_ALIASES, required=False
        ),
        event=resolve_column(fieldnames, args.event_column, EVENT_ALIASES, required=False),
        allowed_actions=resolve_column(
            fieldnames,
            args.allowed_actions_column,
            ALLOWED_ACTIONS_ALIASES,
            required=False,
        ),
    )


def normalize_dimension(raw: str, *, row_number: int) -> str | None:
    value = raw.strip()
    if not value:
        return None
    normalized = DIMENSION_ALIASES.get(value, DIMENSION_ALIASES.get(value.casefold()))
    if normalized is None:
        raise ValueError(f"row {row_number}: unknown target dimension {value!r}")
    return normalized


def cell(row: Mapping[str, str | None], column: str | None) -> str:
    if column is None:
        return ""
    value = row.get(column)
    return value.strip() if isinstance(value, str) else ""


def parse_information(value: str) -> str | list[Any]:
    if not value:
        return ""
    if value[:1] in {"[", "{"}:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, (list, dict)):
            return parsed
    return value


def default_allowed_actions() -> list[dict[str, str]]:
    return [
        {
            "action_code": "AUDIT_RESPONSE",
            "label": "编码当前回答",
            "description": "固定审计集的证据编码；不触发任何模拟状态变化。",
        }
    ]


ACTION_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def parse_allowed_actions(value: str, *, row_number: int) -> list[dict[str, str]]:
    if not value:
        return default_allowed_actions()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in re.split(r"[|,;]", value) if part.strip()]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"row {row_number}: allowed actions must be a non-empty JSON list")
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(parsed, 1):
        if isinstance(entry, str):
            code = entry.strip()
            label = code
            description = "CSV 提供的审计行动。"
        elif isinstance(entry, Mapping):
            code = str(entry.get("action_code", "")).strip()
            label = str(entry.get("label", code)).strip() or code
            description = str(entry.get("description", "CSV 提供的审计行动。")).strip()
        else:
            raise ValueError(f"row {row_number}: allowed action {index} has invalid type")
        if not ACTION_CODE_RE.fullmatch(code):
            raise ValueError(f"row {row_number}: invalid allowed action code {code!r}")
        if code in seen:
            continue
        seen.add(code)
        output.append(
            {
                "action_code": code,
                "label": label[:200] or code,
                "description": description[:800] or "CSV 提供的审计行动。",
            }
        )
    if not output:
        raise ValueError(f"row {row_number}: no unique allowed action codes")
    return output


def make_context(row: Mapping[str, str | None], columns: ColumnMap, row_number: int) -> dict[str, Any]:
    decision_text = cell(row, columns.text)
    if not decision_text:
        raise ValueError(f"row {row_number}: text is empty")
    response: dict[str, Any] = {
        "decision_text": decision_text,
        "rationale_text": cell(row, columns.rationale),
        "information_requests": parse_information(cell(row, columns.information)),
        "revision_text": cell(row, columns.revision),
    }
    context: dict[str, Any] = {
        "event_kind": "main",
        "event_text": cell(row, columns.event) or "固定合成 ECD 审计回答。",
        # Every item is scored on all dimensions.  The CSV target/reference
        # dimension is intentionally excluded from the provider payload.
        "target_dimensions": list(ECD_DIMENSIONS),
        "allowed_actions": parse_allowed_actions(
            cell(row, columns.allowed_actions), row_number=row_number
        ),
        "response": response,
    }
    return context


def load_items(path: Path, args: argparse.Namespace) -> tuple[list[AuditItem], ColumnMap]:
    if not path.is_file():
        raise FileNotFoundError(f"Audit CSV does not exist: {path}")
    items: list[AuditItem] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Audit CSV has no header")
        columns = resolve_columns(reader.fieldnames, args)
        for source_row, row in enumerate(reader, start=2):
            context = make_context(row, columns, source_row)
            target = normalize_dimension(
                cell(row, columns.target_dimension), row_number=source_row
            )
            raw_id = cell(row, columns.sample_id)
            if raw_id:
                sample_id = raw_id
            else:
                # Stable across reruns as long as the substantive row is unchanged.
                sample_id = "auto_" + hash_json(
                    {"target_dimension": target, "context": context}
                )[:16]
            if sample_id in seen_ids:
                raise ValueError(f"row {source_row}: duplicate sample id {sample_id!r}")
            seen_ids.add(sample_id)
            item_hash = hash_json(
                {
                    "sample_id": sample_id,
                    "target_dimension": target,
                    "context": context,
                }
            )
            items.append(
                AuditItem(
                    sample_id=sample_id,
                    source_row=source_row,
                    target_dimension=target,
                    context=context,
                    item_hash=item_hash,
                )
            )
            if args.limit is not None and len(items) >= args.limit:
                break
    if not items:
        raise ValueError("Audit CSV contains no data rows")
    return items, columns


def combined_response_text(context: Mapping[str, Any]) -> str:
    response = context.get("response", {})
    if not isinstance(response, Mapping):
        return ""
    parts: list[str] = []
    for key in ("decision_text", "rationale_text", "information_requests", "revision_text"):
        value = response.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            parts.extend(str(item) for item in value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts.extend(str(item) for item in value)
    return "\n".join(part for part in parts if part)


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    matched: list[str] = []
    folded = text.casefold()
    for term in terms:
        if term.casefold() in folded:
            matched.append(term)
    return matched


def quote_occurrence(text: str, term: str) -> str | None:
    match = re.search(re.escape(term), text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def rule_evidence(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    structural = {
        name: [match.group(0) for match in pattern.finditer(text)][:3]
        for name, pattern in STRUCTURAL_PATTERNS
    }
    active_structures = [name for name, matches in structural.items() if matches]
    evidence: list[dict[str, Any]] = []
    trace: dict[str, Any] = {
        "rule_version": RULE_VERSION,
        "text_length": len(text),
        "structural_features": structural,
        "dimensions": {},
    }
    for dimension in ECD_DIMENSIONS:
        matched_groups: dict[str, list[str]] = {}
        quotes: list[str] = []
        for group_name, terms in RULE_DIMENSION_GROUPS[dimension]:
            matches = find_terms(text, terms)
            if not matches:
                continue
            matched_groups[group_name] = matches
            for term in matches:
                quote = quote_occurrence(text, term)
                if quote and quote not in quotes:
                    quotes.append(quote)
        group_count = len(matched_groups)
        if group_count == 0:
            score = 0
        elif group_count == 1:
            score = 1
        elif group_count == 2:
            score = 2
        else:
            score = 3
            if group_count >= 4 and len(active_structures) >= 2 and len(text) >= 60:
                score = 4
        status = "insufficient" if score == 0 else ("partial" if score <= 2 else "supported")
        confidence = (
            0.78
            if score == 0
            else min(0.94, 0.54 + 0.07 * group_count + 0.03 * len(active_structures))
        )
        rationale = (
            "未命中该维度的预注册关键词概念组。"
            if score == 0
            else (
                f"命中 {group_count} 个不同概念组"
                f"（{', '.join(matched_groups)}）及 {len(active_structures)} 个结构特征。"
            )
        )
        evidence.append(
            {
                "dimension": dimension,
                "score": score,
                "quotes": quotes[:5] if score else [],
                "evidence_status": status,
                "rationale": rationale,
                "confidence": round(confidence, 3),
            }
        )
        trace["dimensions"][dimension] = {
            "matched_groups": matched_groups,
            "group_count": group_count,
            "active_structures": active_structures,
            "score": score,
        }
    return evidence, trace


def rule_output(item: AuditItem) -> tuple[dict[str, Any], dict[str, Any]]:
    text = combined_response_text(item.context)
    evidence, trace = rule_evidence(text)
    actions = item.context["allowed_actions"]
    codes = [entry["action_code"] for entry in actions]
    ambiguous = len(codes) > 1
    output = {
        "action_code": codes[0],
        "classification_confidence": 0.5 if ambiguous else 1.0,
        "mapping_uncertainty": {
            "level": "high" if ambiguous else "low",
            "explanation": (
                "关键词规则不执行语义行动映射，确定性选择首个白名单行动。"
                if ambiguous
                else "审计上下文只提供一个白名单行动。"
            ),
            "alternative_action_codes": codes[1:5],
        },
        "evidence": evidence,
        "feedback": {
            "strengths": ["规则仅编码文本中可审计的关键词和结构证据。"],
            "reflection_questions": ["哪些具体证据或判定阈值能使当前理由更可检验？"],
            "next_step": "补充未被当前文本明确支持的能力维度证据。",
            "limitations": "关键词规则不能理解隐含语义，结果仅作为低复杂度基线。",
            "state_change_notice": STATE_CHANGE_NOTICE,
        },
        "safety_flags": ["none"],
    }
    validate_structured_output("decision_evidence", output)
    return output, trace


def base_record(item: AuditItem, method: str, config_hash: str) -> dict[str, Any]:
    return {
        "record_version": 1,
        "sample_id": item.sample_id,
        "source_row": item.source_row,
        "target_dimension": item.target_dimension,
        "method": method,
        "item_hash": item.item_hash,
        "config_hash": config_hash,
        "generated_at_utc": utc_now(),
    }


def encode_rule(item: AuditItem, config_hash: str, current_script_hash: str) -> dict[str, Any]:
    record = base_record(item, "rule", config_hash)
    try:
        output, trace = rule_output(item)
        record.update(
            {
                "status": "success",
                "raw_json": output,
                "metadata": {
                    "encoder_version": RULE_VERSION,
                    "prompt_version": None,
                    "prompt_hash": None,
                    "schema_name": "decision_evidence",
                    "schema_hash": hash_json(DECISION_EVIDENCE_SCHEMA),
                    "input_hash": hash_json(item.context),
                    "script_hash": current_script_hash,
                    "attempts": 1,
                },
                "audit_trace": trace,
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "raw_json": None,
                "metadata": {
                    "encoder_version": RULE_VERSION,
                    "script_hash": current_script_hash,
                    "attempts": 1,
                },
                "error": {"type": type(exc).__name__, "code": "rule_failure"},
            }
        )
    return record


def encode_llm(item: AuditItem, config_hash: str, service: LLMService) -> dict[str, Any]:
    record = base_record(item, "llm", config_hash)
    try:
        result = service.score_decision(item.context)
        record.update(
            {
                "status": "success",
                "raw_json": result.data,
                "metadata": result.metadata.to_dict(),
            }
        )
    except LLMRequestFailed as exc:
        record.update(
            {
                "status": "failed",
                "raw_json": None,
                "metadata": exc.metadata.to_dict(),
                "error": {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "attempts": exc.attempts,
                },
            }
        )
    except Exception as exc:
        # Exception text can contain a provider response body, URL, or source
        # text.  Store only stable non-sensitive categories here.
        record.update(
            {
                "status": "failed",
                "raw_json": None,
                "metadata": None,
                "error": {"type": type(exc).__name__, "code": "preflight_or_call_failure"},
            }
        )
    return record


class PredictionStore:
    """Exclusively locked append-only JSONL store with resume indexing."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a+", encoding="utf-8")
        os.chmod(path, 0o600)
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.success_keys: set[tuple[str, str, str]] = set()
        self._load()

    def _load(self) -> None:
        self.handle.seek(0)
        for line_number, line in enumerate(self.handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume: {self.path} line {line_number} is invalid JSON"
                ) from exc
            if record.get("status") == "success":
                values = (
                    record.get("sample_id"),
                    record.get("item_hash"),
                    record.get("config_hash"),
                )
                if all(isinstance(value, str) and value for value in values):
                    self.success_keys.add(values)  # type: ignore[arg-type]
        self.handle.seek(0, os.SEEK_END)

    def is_complete(self, item: AuditItem, config_hash: str) -> bool:
        return (item.sample_id, item.item_hash, config_hash) in self.success_keys

    def append(self, record: Mapping[str, Any], secret: str | None = None) -> None:
        payload = canonical_json(record)
        if secret and secret in payload:
            raise RuntimeError("Refusing to persist a record containing the API credential")
        self.handle.write(payload + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        if record.get("status") == "success":
            key = (
                str(record["sample_id"]),
                str(record["item_hash"]),
                str(record["config_hash"]),
            )
            self.success_keys.add(key)

    def close(self) -> None:
        if not self.handle.closed:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def __enter__(self) -> "PredictionStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def rule_config(columns: ColumnMap, current_script_hash: str) -> dict[str, Any]:
    return {
        "method": "rule",
        "encoder_version": RULE_VERSION,
        "dimension_groups": RULE_DIMENSION_GROUPS,
        "structural_patterns": [(name, pattern.pattern) for name, pattern in STRUCTURAL_PATTERNS],
        "score_rule": "0/1/2 groups -> 0/1/2; >=3 -> 3; >=4 + >=2 structures + >=60 chars -> 4",
        "schema_hash": hash_json(DECISION_EVIDENCE_SCHEMA),
        "columns": columns.public_dict(),
        "script_hash": current_script_hash,
        "label_leakage_guard": "target_dimension_and_reference_columns_not_sent_to_encoder",
    }


def llm_config(
    columns: ColumnMap,
    runtime: LLMRuntimeConfig,
    provider: OpenAICompatibleProvider,
    current_script_hash: str,
) -> dict[str, Any]:
    return {
        "method": "llm",
        "prompt_version": DECISION_PROMPT_VERSION,
        "repair_policy_version": DECISION_REPAIR_POLICY_VERSION,
        "quote_candidate_policy_version": DECISION_QUOTE_CANDIDATE_POLICY_VERSION,
        "quote_candidate_policy_hash": DECISION_QUOTE_CANDIDATE_POLICY_HASH,
        "schema_hash": hash_json(DECISION_EVIDENCE_SCHEMA),
        "provider": provider.name,
        "model": provider.model,
        "api_version": provider.api_version,
        "base_url_hash": hashlib.sha256(provider.base_url.encode("utf-8")).hexdigest(),
        "runtime": {
            "timeout_seconds": runtime.timeout_seconds,
            "max_retries": runtime.max_retries,
            "retry_base_seconds": runtime.retry_base_seconds,
            "temperature": runtime.temperature,
            "max_tokens": runtime.max_tokens,
        },
        "columns": columns.public_dict(),
        "script_hash": current_script_hash,
        "label_leakage_guard": "target_dimension_and_reference_columns_not_sent_to_provider",
    }


def make_service(args: argparse.Namespace) -> tuple[LLMService, str, OpenAICompatibleProvider]:
    if DECISION_PROMPT_VERSION != EXPECTED_PROMPT_VERSION:
        raise RuntimeError(
            "VentureBridge decision prompt version changed; audit configuration must be reviewed"
        )
    api_key = os.environ.get("V_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LLM method requires V_API_KEY in the runtime environment")
    configured_model = os.environ.get("V_MODEL", QWEN_MODEL).strip() or QWEN_MODEL
    if configured_model != QWEN_MODEL:
        raise RuntimeError(
            f"This frozen audit requires {QWEN_MODEL}; runtime V_MODEL selects a different model"
        )
    base_url = os.environ.get("V_API_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    api_version = os.environ.get("V_API_VERSION", "").strip() or None
    provider = OpenAICompatibleProvider(
        base_url=base_url,
        _api_key=api_key,
        model=QWEN_MODEL,
        api_version=api_version,
    )
    runtime = LLMRuntimeConfig(
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        retry_base_seconds=args.retry_base_seconds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return LLMService(provider=provider, runtime=runtime), api_key, provider


def run_rule(
    items: Sequence[AuditItem],
    output_dir: Path,
    columns: ColumnMap,
    current_script_hash: str,
) -> dict[str, int]:
    config_hash = hash_json(rule_config(columns, current_script_hash))
    counts = {"success": 0, "failed": 0, "skipped": 0}
    output_path = output_dir / "ecd_audit_rule.jsonl"
    with PredictionStore(output_path) as store:
        for item in items:
            if store.is_complete(item, config_hash):
                counts["skipped"] += 1
                continue
            record = encode_rule(item, config_hash, current_script_hash)
            store.append(record)
            counts[str(record["status"])] += 1
    return counts


def run_llm(
    items: Sequence[AuditItem],
    output_dir: Path,
    columns: ColumnMap,
    current_script_hash: str,
    args: argparse.Namespace,
) -> dict[str, int]:
    service, api_key, provider = make_service(args)
    runtime = service.runtime
    config_hash = hash_json(llm_config(columns, runtime, provider, current_script_hash))
    counts = {"success": 0, "failed": 0, "skipped": 0}
    output_path = output_dir / "ecd_audit_llm.jsonl"
    with PredictionStore(output_path) as store:
        pending = [item for item in items if not store.is_complete(item, config_hash)]
        counts["skipped"] = len(items) - len(pending)
        if not pending:
            return counts
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="ecd-audit") as pool:
            futures: dict[Future[dict[str, Any]], AuditItem] = {
                pool.submit(encode_llm, item, config_hash, service): item for item in pending
            }
            completed = 0
            for future in as_completed(futures):
                record = future.result()
                store.append(record, secret=api_key)
                counts[str(record["status"])] += 1
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"llm progress: {completed}/{len(pending)} "
                        f"(success={counts['success']}, failed={counts['failed']})",
                        file=sys.stderr,
                    )
    return counts


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def workers_type(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 40:
        raise argparse.ArgumentTypeError("must be between 1 and 40")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rule and/or frozen VentureBridge ECD encoders on an audit CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", choices=("rule", "llm", "both"), default="both")
    parser.add_argument("--limit", type=positive_int, help="Process only the first N rows")
    parser.add_argument("--workers", type=workers_type, default=20, help="LLM concurrency (max 40)")

    parser.add_argument("--id-column")
    parser.add_argument("--text-column")
    parser.add_argument("--target-dimension-column")
    parser.add_argument("--rationale-column")
    parser.add_argument("--information-column")
    parser.add_argument("--revision-column")
    parser.add_argument("--event-column")
    parser.add_argument("--allowed-actions-column")

    parser.add_argument("--timeout", type=positive_int, default=180)
    parser.add_argument("--retries", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=positive_int, default=2400)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be between 0 and 2")
    if not 0.0 <= args.retry_base_seconds <= 30.0:
        raise ValueError("--retry-base-seconds must be between 0 and 30")
    if not 128 <= args.max_tokens <= 32000:
        raise ValueError("--max-tokens must be between 128 and 32000")
    if not 1 <= args.timeout <= 600:
        raise ValueError("--timeout must be between 1 and 600")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        items, columns = load_items(args.input.resolve(), args)
        current_script_hash = script_hash()
        methods = ("rule", "llm") if args.method == "both" else (args.method,)
        summaries: dict[str, dict[str, int]] = {}
        for method in methods:
            if method == "rule":
                summaries[method] = run_rule(
                    items, args.output_dir.resolve(), columns, current_script_hash
                )
            else:
                summaries[method] = run_llm(
                    items,
                    args.output_dir.resolve(),
                    columns,
                    current_script_hash,
                    args,
                )
        # Counts and paths are safe operational feedback; no source text,
        # provider response, prompt, or environment value is printed.
        print(canonical_json({"input_rows": len(items), "methods": summaries}))
        return 0 if all(summary["failed"] == 0 for summary in summaries.values()) else 2
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
