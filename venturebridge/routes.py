"""Flask routes for the learner-facing VentureBridge workflow."""

from __future__ import annotations

import copy
import json
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from .allocation import AnchorAllocationPlan, AllocationPlanIntegrityError
from .auth import login_required
from .domain import (
    BlueprintLibrary,
    EntrepreneurshipState,
    EventScheduler,
    ScheduledEvent,
    SchedulingError,
    StateTransitionError,
    apply_classified_action,
)
from .llm import (
    LLMRecorderError,
    LLMConfigurationError,
    LLMRequestFailed,
    LLMResult,
    LLMService,
    check_llm_readiness,
)
from .privacy import detect_sensitive_text, find_sensitive_data, sanitize_for_llm
from .repository import (
    acquire_operation_lock,
    active_profile,
    active_simulation,
    complete_delivery,
    create_simulation,
    create_user,
    delete_user_data,
    export_user_data,
    get_delivery,
    get_simulation,
    latest_recommendations,
    pending_delivery,
    record_llm_call,
    release_operation_lock,
    save_delivery,
    save_profile,
    save_recommendations,
    simulation_history,
    subject_ref,
    users_by_alias,
)
from .schemas import validate_structured_output


def _field(name: str, *, limit: int = 4_000) -> str:
    value = request.form.get(name, "").strip()
    if len(value) > limit:
        raise ValueError(f"{name} 超过允许长度（{limit} 字符）。")
    return value


def _current_user_id() -> str:
    user = g.get("user")
    if not user:
        raise LookupError("No authenticated workspace.")
    return str(user["id"])


_DIRECTION_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "technology": (
        "技术", "科技", "软件", "平台", "人工智能", "ai", "数据", "数字", "硬件",
        "technology", "software", "platform", "automation",
    ),
    "services": (
        "服务", "咨询", "培训", "教育", "维修", "物流", "健康", "service",
        "consulting", "training", "education", "health",
    ),
    "consumer": (
        "消费", "零售", "电商", "餐饮", "用户", "社区", "家庭", "生活", "consumer",
        "retail", "commerce", "food",
    ),
    "social": (
        "社会", "公益", "可持续", "环保", "乡村", "无障碍", "影响力", "social",
        "sustainable", "nonprofit", "inclusive",
    ),
    "creative": (
        "文化", "创意", "设计", "内容", "媒体", "艺术", "文旅", "creative",
        "design", "media", "art", "culture",
    ),
}


def _direction_scheduler_tags(direction: Mapping[str, Any]) -> list[str]:
    """Map a chosen exploration hypothesis to auditable blueprint tags.

    The map only adapts training content; it is not a claim that a learner is
    suited to an industry. Unknown directions deliberately fall back to the
    broad common path.
    """

    fields = (
        direction.get("title", ""),
        direction.get("exploration_hypothesis", ""),
        direction.get("why_it_might_fit", ""),
    )
    text = " ".join(str(value) for value in fields).casefold()
    tags = [
        tag
        for tag, keywords in _DIRECTION_TAG_KEYWORDS.items()
        if any(keyword.casefold() in text for keyword in keywords)
    ]
    return sorted(tags) or ["general"]


def _prior_process_trace(history: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    """Return a bounded, data-minimized trace for cross-event ECD context."""

    trace: list[dict[str, Any]] = []
    for item in history:
        if not item.get("response_id"):
            continue
        evidence = item.get("evidence_json") or {}
        dimension_rows = evidence.get("evidence", []) if isinstance(evidence, Mapping) else []
        trace.append(
            {
                "turn_no": int(item["turn_no"]),
                "event_kind": str(item["event_kind"]),
                "blueprint_id": str(item["blueprint_id"]),
                "action_code": str(item.get("action_code") or ""),
                "decision_text": str(item.get("decision_text") or "")[:1200],
                "rationale_text": str(item.get("rationale_text") or "")[:1200],
                "revision_text": str(item.get("revision_text") or "")[:1200],
                "ecd_labels": [
                    {
                        "dimension": row.get("dimension"),
                        "score": row.get("score"),
                        "evidence_status": row.get("evidence_status"),
                    }
                    for row in dimension_rows
                    if isinstance(row, Mapping)
                ],
                "main_state_after": item.get("main_state_after_json") or {},
            }
        )
    return trace[-limit:]


def _event_llm_context(event: ScheduledEvent) -> dict[str, Any]:
    """Project a scheduled event to the fields required by an LLM task.

    ``ScheduledEvent.to_dict`` also contains allocation-plan hashes, draw
    fingerprints and (in a registered plan) a stable allocation-unit
    pseudonym.  Those fields are essential in the local audit snapshot but
    irrelevant to narrative generation and evidence coding, so they must not
    cross the external API boundary.
    """

    return {
        "event_kind": event.event_kind,
        "blueprint_id": event.blueprint_id,
        "family_id": event.family_id,
        "form_id": event.form_id,
        "title": event.title,
        "target_dimensions": list(event.target_dimensions),
        "difficulty_level": event.difficulty_level,
        "visible_state": event.visible_state.to_dict(),
        "event_text": event.event_text,
        "allowed_actions": [dict(action) for action in event.allowed_actions],
        "evidence_requirements": dict(event.evidence_requirements),
    }


def _direction_llm_context(direction: Mapping[str, Any]) -> dict[str, Any]:
    """Return only pedagogically useful direction fields for narration."""

    allowed_fields = (
        "title",
        "exploration_hypothesis",
        "why_it_might_fit",
        "uncertainties",
        "counterarguments",
        "first_experiments",
    )
    return {name: direction[name] for name in allowed_fields if name in direction}


def _reject_detectable_identifiers(value: Any) -> None:
    """Reject, rather than silently store, detectable direct identifiers."""

    findings = find_sensitive_data(value)
    if findings:
        kinds = "、".join(sorted({item.kind for item in findings}))
        raise ValueError(
            f"修改内容含可检测的直接标识（{kinds}）；请删除后再保存。"
        )


def _llm_recorder(record: dict[str, Any]) -> None:
    metadata = record["metadata"]
    user = g.get("user")
    user_id = str(user["id"]) if user else None
    call_id = record_llm_call(
        user_id=user_id,
        purpose=metadata["purpose"],
        provider=metadata["provider"],
        model=metadata["model"],
        api_version=metadata.get("api_version"),
        prompt_version=metadata["prompt_version"],
        prompt_hash=metadata["prompt_hash"],
        parameters=metadata["parameters"],
        input_hash=metadata["input_hash"],
        output=record.get("output"),
        status=record["status"],
        error_code=record.get("error_code"),
        latency_ms=metadata["latency_ms"],
        attempts=metadata["attempts"],
    )
    if not hasattr(g, "llm_call_ids"):
        g.llm_call_ids = {}
    g.llm_call_ids[metadata["invocation_id"]] = call_id


def _llm_service() -> LLMService:
    service = g.get("llm_service")
    if service is None:
        provider = current_app.extensions.get("venturebridge_llm_provider")
        runtime = current_app.extensions.get("venturebridge_llm_runtime")
        service = LLMService(provider=provider, recorder=_llm_recorder, runtime=runtime)
        g.llm_service = service
    return service


def _call_id(result: LLMResult) -> str:
    call_id = getattr(g, "llm_call_ids", {}).get(result.metadata.invocation_id)
    if not call_id:
        raise RuntimeError("LLM result was not written to the audit store.")
    return str(call_id)


def _safe_llm_error(error: BaseException) -> tuple[str, int]:
    if isinstance(error, LLMRecorderError):
        return "模型调用完成，但审计记录未能安全保存；本次结果未被采用。", 503
    if isinstance(error, LLMConfigurationError):
        return (
            "LLM API 尚未配置。请由管理员设置 V_API_KEY、V_API_BASE_URL 和 V_MODEL。",
            503,
        )
    if isinstance(error, LLMRequestFailed):
        messages = {
            "rate_limited": "LLM 服务暂时限流，请稍后重试。",
            "timeout": "LLM 服务响应超时，请稍后重试。",
            "authentication_failed": "LLM API 凭据被拒绝，请联系管理员。",
            "provider_unavailable": "LLM 服务暂时不可用，请稍后重试。",
            "network_error": "暂时无法连接 LLM 服务，请稍后重试。",
        }
        return messages.get(error.code, "LLM 返回内容未通过结构校验，请重试。"), 502
    return "系统暂时无法完成该请求。", 500


def _profile_source_from_form() -> dict[str, Any]:
    text_limit = int(current_app.config["MAX_PROFILE_TEXT_CHARS"])
    source: dict[str, Any] = {
        "education_stage": _field("education_stage", limit=200),
        "discipline": _field("discipline", limit=300),
        "learning_goals": _field("learning_goals", limit=text_limit),
        "experience": _field("experience_text", limit=text_limit),
        "skills": _field("skills_text", limit=text_limit),
        "interests": _field("interests_text", limit=text_limit),
        "resources": _field("resources_text", limit=text_limit),
        "entrepreneurship_experience": _field(
            "entrepreneurship_experience", limit=text_limit
        ),
    }
    if request.form.get("include_mbti") == "yes":
        mbti_type = _field("mbti_type", limit=16).upper()
        valid_types = {
            "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ",
        }
        if mbti_type not in valid_types:
            raise ValueError("MBTI 类型格式无效；也可以选择不提供。")
        source["mbti"] = {
            "type": mbti_type,
            "self_reflection_note": _field("mbti_note", limit=1_000),
            "measurement_status": "optional_self_report_low_weight",
        }
    if not any(
        source[key]
        for key in ("learning_goals", "experience", "skills", "interests", "resources")
    ):
        raise ValueError("请至少填写学习目标、经历、技能、兴趣或可用资源中的一项。")
    total = sum(len(value) for value in source.values() if isinstance(value, str))
    if total > text_limit:
        raise ValueError(f"画像文字总长度不能超过 {text_limit} 字符。")
    safe = sanitize_for_llm(source)
    if not isinstance(safe, dict):
        raise ValueError("画像输入格式无效。")
    return safe


def _generate_recommendations(user_id: str, profile: Mapping[str, Any]) -> str:
    result = _llm_service().recommend(profile["profile_json"])
    return save_recommendations(
        user_id,
        str(profile["id"]),
        result.data,
        result.metadata.prompt_version,
        _call_id(result),
    )


def _scheduled_from_delivery(delivery: Mapping[str, Any]) -> ScheduledEvent:
    domain_event = delivery["event_json"].get("domain_event")
    if not isinstance(domain_event, Mapping):
        raise ValueError("Stored event is missing its domain snapshot.")
    return ScheduledEvent(
        event_kind=str(domain_event["event_kind"]),
        blueprint_id=str(domain_event["blueprint_id"]),
        family_id=str(domain_event["family_id"]),
        form_id=str(domain_event["form_id"]),
        title=str(domain_event["title"]),
        target_dimensions=tuple(domain_event["target_dimensions"]),
        difficulty_level=int(domain_event["difficulty_level"]),
        state_variant_id=str(
            domain_event.get("state_variant_id", "legacy_single_state")
        ),
        visible_state=EntrepreneurshipState.from_dict(domain_event["visible_state"]),
        event_text=str(domain_event["event_text"]),
        form_fixed=bool(domain_event["form_fixed"]),
        allowed_actions=tuple(domain_event["allowed_actions"]),
        evidence_requirements=dict(domain_event["evidence_requirements"]),
        selection_probability=float(domain_event["selection_probability"]),
        scheduler_audit=dict(domain_event["scheduler_audit"]),
    )


def _frozen_anchor_payload(event: ScheduledEvent) -> dict[str, Any]:
    """Render a frozen anchor without an LLM call or hidden variation."""

    return {
        "title": event.title,
        "narrative": event.event_text,
        "decision_prompt": (
            "请说明你的决策、理由、还需要哪些信息，以及什么新证据会让你调整方案。"
        ),
        "information_cues": list(event.evidence_requirements.values()),
        "allowed_action_codes": list(event.allowed_action_codes),
        "learning_focus": list(event.target_dimensions),
        "uncertainties": ["共同压力测试只模拟决策环境，不预测现实市场后果。"],
        "guardrail_notice": "这是教学模拟，不是市场预测、投资或法律建议。",
    }


def _event_view(delivery: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(delivery)
    event_json = result.get("event_json", {})
    result["content"] = event_json.get("content", {})
    result["domain_event"] = event_json.get("domain_event", {})
    return result


def _assert_simulation_anchor_binding(
    simulation: Mapping[str, Any],
    *,
    user_id: str,
    anchor_plan: AnchorAllocationPlan | None,
    anchor_every: int,
    plan_required: bool = False,
    next_turn: int | None = None,
) -> None:
    """Keep one simulation on its originally registered allocation path."""

    direction = simulation.get("direction_json")
    if not isinstance(direction, Mapping):
        raise SchedulingError("simulation is missing its anchor allocation registration")
    mode = direction.get("anchor_allocation_mode")
    registered_interval = direction.get("anchor_every_turns")
    if (
        isinstance(registered_interval, bool)
        or not isinstance(registered_interval, int)
        or registered_interval != anchor_every
    ):
        raise SchedulingError(
            "simulation anchor checkpoint interval does not match the active scheduler"
        )
    expected_unit = subject_ref(user_id)
    if anchor_plan is None:
        if plan_required:
            raise SchedulingError(
                "common-anchor delivery is blocked until a registered queue plan is active"
            )
        if (
            mode != "development_hash_fallback"
            or direction.get("anchor_plan_hash") is not None
            or direction.get("allocation_unit_id") is not None
        ):
            raise SchedulingError(
                "simulation is bound to a frozen anchor plan that is not active"
            )
        return
    if (
        mode != "precomputed_queue_balanced"
        or direction.get("anchor_plan_hash") != anchor_plan.plan_hash
        or direction.get("allocation_unit_id") != expected_unit
    ):
        raise SchedulingError(
            "simulation registration does not match the active frozen anchor plan"
        )
    try:
        anchor_plan.assignment_for(expected_unit, 1)
        if next_turn is not None and next_turn % anchor_every == 0:
            anchor_plan.assignment_for(expected_unit, next_turn // anchor_every)
    except KeyError as exc:
        raise SchedulingError(
            "the frozen queue plan has no assignment for this unit and checkpoint"
        ) from exc


def _assert_delivery_anchor_binding(
    delivery: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    anchor_plan: AnchorAllocationPlan | None,
    anchor_every: int,
) -> None:
    """Verify a pending delivery against the simulation's sticky allocation binding."""

    direction = simulation.get("direction_json")
    scheduler_audit = delivery.get("scheduler_json")
    if not isinstance(direction, Mapping) or not isinstance(scheduler_audit, Mapping):
        raise SchedulingError("pending delivery is missing anchor allocation metadata")
    mode = direction.get("anchor_allocation_mode")
    if scheduler_audit.get("allocation_mode") != mode:
        raise SchedulingError("pending delivery allocation mode does not match simulation")
    if scheduler_audit.get("anchor_every_turns") != anchor_every:
        raise SchedulingError("pending delivery checkpoint interval does not match simulation")
    if mode == "development_hash_fallback":
        if any(
            scheduler_audit.get(field) is not None
            for field in (
                "allocation_plan_hash",
                "allocation_unit_id",
                "assignment_hash",
                "checkpoint_index",
            )
        ):
            raise SchedulingError(
                "development delivery contains unexpected frozen-plan metadata"
            )
        return
    if anchor_plan is None or mode != "precomputed_queue_balanced":
        raise SchedulingError("pending delivery requires an unavailable frozen anchor plan")
    unit_id = direction.get("allocation_unit_id")
    if (
        scheduler_audit.get("allocation_plan_hash") != direction.get("anchor_plan_hash")
        or scheduler_audit.get("allocation_plan_hash") != anchor_plan.plan_hash
        or scheduler_audit.get("allocation_unit_id") != unit_id
    ):
        raise SchedulingError("pending delivery does not match the registered anchor plan")
    if delivery.get("event_kind") not in {"exact_anchor", "surface_anchor"}:
        return
    turn_no = delivery.get("turn_no")
    if isinstance(turn_no, bool) or not isinstance(turn_no, int) or turn_no % anchor_every:
        raise SchedulingError("planned anchor delivery has an invalid checkpoint turn")
    checkpoint_index = turn_no // anchor_every
    try:
        assignment = anchor_plan.assignment_for(str(unit_id), checkpoint_index)
    except KeyError as exc:
        raise SchedulingError(
            "the frozen queue plan has no assignment for this delivery checkpoint"
        ) from exc
    domain_event = delivery.get("event_json", {}).get("domain_event", {})
    expected_event_kind = (
        "exact_anchor" if assignment.get("form_kind") == "exact" else "surface_anchor"
    )
    if not isinstance(domain_event, Mapping) or (
        scheduler_audit.get("checkpoint_index") != checkpoint_index
        or scheduler_audit.get("assignment_hash") != assignment.get("assignment_hash")
        or scheduler_audit.get("allocation_rank") != assignment.get("allocation_rank")
        or scheduler_audit.get("allocation_algorithm_version")
        != assignment.get("algorithm_version")
        or scheduler_audit.get("library_version") != assignment.get("library_version")
        or scheduler_audit.get("state_variant_id")
        != assignment.get("state_variant_id")
        or scheduler_audit.get("conditional_design_probabilities")
        != assignment.get("conditional_probabilities")
        or delivery.get("event_kind") != expected_event_kind
        or delivery.get("blueprint_id") != assignment.get("blueprint_id")
        or delivery.get("form_id") != assignment.get("form_id")
        or domain_event.get("family_id") != assignment.get("family_id")
        or domain_event.get("state_variant_id") != assignment.get("state_variant_id")
        or domain_event.get("target_dimensions") != assignment.get("target_dimensions")
        or domain_event.get("difficulty_level") != assignment.get("difficulty_level")
    ):
        raise SchedulingError("planned anchor delivery hash or assignment does not match")


def register_routes(app: Flask) -> None:
    library = BlueprintLibrary.load_default()
    anchor_plan: AnchorAllocationPlan | None = None
    configured_plan_path = str(app.config.get("ANCHOR_PLAN_PATH") or "").strip()
    expected_hash = str(
        app.config.get("ANCHOR_PLAN_EXPECTED_HASH") or ""
    ).strip()
    plan_required = bool(app.config.get("ANCHOR_PLAN_REQUIRED")) or (
        str(app.config.get("ENVIRONMENT") or "").strip().lower() == "production"
    )
    if configured_plan_path:
        if not expected_hash:
            raise RuntimeError(
                "Configured common-anchor allocation plan requires a registered hash."
            )
        try:
            anchor_plan = AnchorAllocationPlan.load(
                Path(configured_plan_path).expanduser(),
                library=library,
                expected_plan_hash=expected_hash,
            )
        except AllocationPlanIntegrityError as exc:
            raise RuntimeError(
                "Configured common-anchor allocation plan failed integrity validation."
            ) from exc
    elif expected_hash:
        raise RuntimeError(
            "A registered common-anchor plan hash requires ANCHOR_PLAN_PATH."
        )
    scheduler = EventScheduler(
        library,
        anchor_every=int(app.config["ANCHOR_EVERY_TURNS"]),
        anchor_plan=anchor_plan,
        allow_development_fallback=not plan_required,
    )
    app.extensions["venturebridge_blueprint_library"] = library
    app.extensions["venturebridge_event_scheduler"] = scheduler
    app.extensions["venturebridge_anchor_plan"] = anchor_plan

    @app.route("/", methods=["GET"])
    def home() -> str:
        new_code = session.pop("new_access_code", None)
        profile = active_profile(_current_user_id()) if g.get("user") else None
        simulation = active_simulation(_current_user_id()) if g.get("user") else None
        return render_template(
            "home.html",
            user=g.get("user"),
            new_access_code=new_code,
            profile=profile,
            simulation=simulation,
        )

    @app.route("/workspace", methods=["POST"])
    def create_workspace() -> Response:
        alias = _field("alias", limit=40)
        if len(alias) < 2:
            flash("请使用至少 2 个字符的非实名代号。", "error")
            return redirect(url_for("home"))
        if detect_sensitive_text(alias):
            flash("学习代号不能使用电话、邮箱、网址或其他可检测身份信息。", "error")
            return redirect(url_for("home"))
        if request.form.get("consent_accepted") not in {"on", "yes", "1"}:
            flash("请先确认数据处理说明。", "error")
            return redirect(url_for("home"))
        access_code = secrets.token_urlsafe(12)
        user_id = create_user(alias, generate_password_hash(access_code), True)
        session.clear()
        session["user_id"] = user_id
        session["new_access_code"] = access_code
        flash("学习工作区已创建。访问码只显示这一次，请妥善保存。", "success")
        return redirect(url_for("home"))

    @app.route("/login", methods=["POST"])
    def login() -> Response:
        alias = _field("alias", limit=40)
        access_code = _field("access_code", limit=200)
        matched = next(
            (
                item
                for item in users_by_alias(alias)
                if check_password_hash(item["access_code_hash"], access_code)
            ),
            None,
        )
        if matched is None:
            flash("代号或访问码不正确。", "error")
            return redirect(url_for("home"))
        session.clear()
        session["user_id"] = matched["id"]
        flash("已恢复学习工作区。", "success")
        return redirect(url_for("home"))

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout() -> Response:
        session.clear()
        return redirect(url_for("home"))

    @app.route("/onboarding", methods=["GET", "POST"])
    @login_required
    def onboarding() -> str | Response:
        user_id = _current_user_id()
        existing = active_profile(user_id)
        if request.method == "GET":
            prefill: dict[str, Any] = {}
            if existing is not None:
                source = existing["source_json"]
                prefill = {
                    "education_stage": source.get("education_stage", ""),
                    "discipline": source.get("discipline", ""),
                    "learning_goals": source.get("learning_goals", ""),
                    "experience_text": source.get("experience", ""),
                    "skills_text": source.get("skills", ""),
                    "interests_text": source.get("interests", ""),
                    "resources_text": source.get("resources", ""),
                    "entrepreneurship_experience": source.get(
                        "entrepreneurship_experience", ""
                    ),
                    "include_mbti": bool(source.get("mbti")),
                    "mbti_type": source.get("mbti", {}).get("type", ""),
                    "mbti_note": source.get("mbti", {}).get(
                        "self_reflection_note", ""
                    ),
                }
            return render_template("onboarding.html", profile=existing, form=prefill)
        try:
            source = _profile_source_from_form()
            result = _llm_service().build_profile(source)
            save_profile(
                user_id,
                source,
                result.data,
                result.metadata.prompt_version,
                _call_id(result),
            )
        except ValueError as error:
            flash(str(error), "error")
            return render_template("onboarding.html", profile=existing), 400
        except (LLMConfigurationError, LLMRequestFailed, LLMRecorderError) as error:
            message, status = _safe_llm_error(error)
            flash(message, "error")
            return render_template("onboarding.html", profile=existing), status
        flash("画像草稿已生成。请逐项核对、修改或删除后再继续。", "success")
        return redirect(url_for("profile_review"))

    @app.route("/profile", methods=["GET"])
    @login_required
    def profile_review() -> str | Response:
        profile = active_profile(_current_user_id())
        if profile is None:
            return redirect(url_for("onboarding"))
        return render_template("profile_review.html", profile=profile)

    @app.route("/profile", methods=["POST"])
    @login_required
    def update_profile() -> Response:
        user_id = _current_user_id()
        current = active_profile(user_id)
        if current is None:
            return redirect(url_for("onboarding"))
        if _field("profile_version", limit=20) != str(current["version"]):
            flash("画像已在其他页面更新，请刷新后再修改。", "error")
            return redirect(url_for("profile_review"))

        revised = copy.deepcopy(current["profile_json"])
        revised["profile_summary"] = _field("profile_summary", limit=4_000)
        removed = set(request.form.getlist("remove_trait"))
        kept_traits: list[dict[str, Any]] = []
        for trait in revised["traits"]:
            if trait["trait_id"] in removed:
                continue
            edited = _field(f"trait__{trait['trait_id']}", limit=800)
            if edited and edited != trait["claim"]:
                trait.setdefault("original_claim", trait["claim"])
                trait["claim"] = edited
                trait["learner_edited"] = True
            kept_traits.append(trait)
        revised["traits"] = kept_traits
        if not kept_traits:
            flash("至少保留一条有原文证据的画像条目。", "error")
            return redirect(url_for("profile_review"))

        mbti_type = _field("mbti_type", limit=16).upper()
        valid_mbti_types = {
            "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ",
        }
        if mbti_type and mbti_type not in valid_mbti_types:
            flash("MBTI 类型格式无效；也可以选择不使用。", "error")
            return redirect(url_for("profile_review"))
        mbti_note = _field("mbti_note", limit=800)
        if mbti_type:
            revised["mbti_usage"] = {
                "used": True,
                "weight": "low",
                "note": mbti_note
                or f"学习者将 {mbti_type} 作为可修改的低权重自我反思线索。",
            }
        else:
            revised["mbti_usage"] = {
                "used": False,
                "weight": "none",
                "note": "学习者选择不使用 MBTI 线索。",
            }
        correction_note = _field("correction_note", limit=1_000)
        source = copy.deepcopy(current["source_json"])
        if mbti_type:
            source["mbti"] = {
                "type": mbti_type,
                "self_reflection_note": mbti_note,
                "measurement_status": "optional_self_report_low_weight",
            }
        else:
            source.pop("mbti", None)
        if correction_note:
            source["learner_correction_note"] = correction_note
        try:
            _reject_detectable_identifiers(revised)
            _reject_detectable_identifiers(source)
            revised = sanitize_for_llm(revised)
            source = sanitize_for_llm(source)
            if not isinstance(revised, dict) or not isinstance(source, dict):
                raise ValueError("画像修改格式无效。")
            validate_structured_output("profile", revised)
            save_profile(
                user_id,
                source,
                revised,
                "learner-correction-2026-10-v1",
                None,
            )
            updated = active_profile(user_id)
            if request.form.get("next") == "recommendations" and updated is not None:
                _generate_recommendations(user_id, updated)
        except ValueError as error:
            flash(f"画像修改未通过结构检查：{error}", "error")
            return redirect(url_for("profile_review"))
        except (LLMConfigurationError, LLMRequestFailed, LLMRecorderError) as error:
            message, _status = _safe_llm_error(error)
            flash(message, "error")
            return redirect(url_for("profile_review"))
        flash("画像修改已另存为新版本。", "success")
        if request.form.get("next") == "recommendations":
            return redirect(url_for("recommendations"))
        return redirect(url_for("profile_review"))

    @app.route("/recommendations", methods=["GET"])
    @login_required
    def recommendations() -> str | Response:
        user_id = _current_user_id()
        profile = active_profile(user_id)
        if profile is None:
            return redirect(url_for("onboarding"))
        recommendations_row = latest_recommendations(user_id)
        if recommendations_row is None or recommendations_row["profile_id"] != profile["id"]:
            flash("请先在画像页确认当前版本并生成探索方向。", "info")
            return redirect(url_for("profile_review"))
        return render_template(
            "recommendations.html",
            profile=profile,
            recommendation_record=recommendations_row,
            recommendations=recommendations_row["recommendations_json"],
        )

    @app.route("/simulation/start", methods=["POST"])
    @login_required
    def start_simulation() -> Response:
        user_id = _current_user_id()
        if plan_required and anchor_plan is None:
            return render_template(
                "error.html",
                message=(
                    "系统处于共同锚计划登记阶段；管理员登记并核验冻结计划后，"
                    "才能创建或投放模拟。"
                ),
                status=503,
            ), 503
        existing = active_simulation(user_id)
        if existing is not None:
            flash("当前已有进行中的模拟。", "info")
            return redirect(url_for("simulation_dashboard"))
        rec = latest_recommendations(user_id)
        if rec is None:
            return redirect(url_for("profile_review"))
        profile = active_profile(user_id)
        if profile is None or rec["profile_id"] != profile["id"]:
            flash("画像已更新，请先为当前画像重新生成探索方向。", "error")
            return redirect(url_for("profile_review"))
        direction_id = _field("direction_id", limit=80)
        options = rec["recommendations_json"].get("recommendations", [])
        custom_direction = _field("custom_direction", limit=600)
        if direction_id == "__custom__" and custom_direction:
            safe_custom = sanitize_for_llm(custom_direction)
            direction = {
                "direction_id": "learner_selected_alternative",
                "title": str(safe_custom),
                "exploration_hypothesis": str(safe_custom),
                "why_it_might_fit": "由学习者主动选择；系统不把推荐视为限制。",
                "uncertainties": ["尚未由当前画像证据支持，需要在模拟中检验。"],
                "learner_can_reject": True,
            }
        else:
            direction = next(
                (item for item in options if item.get("direction_id") == direction_id), None
            )
        if direction is None:
            flash("请选择一个推荐方向，或填写自己的替代探索方向。", "error")
            return redirect(url_for("recommendations"))
        scheduler_tags = _direction_scheduler_tags(direction)
        direction = {
            **direction,
            "scheduler_tags": scheduler_tags,
            "scheduler_tag_basis": "deterministic-keyword-v1",
            "anchor_every_turns": scheduler.anchor_every,
            "scheduler_tag_notice": (
                "标签只调整训练蓝图候选池，不表示学习者适合或不适合某行业。"
            ),
        }
        if anchor_plan is not None:
            allocation_unit_id = subject_ref(user_id)
            try:
                anchor_plan.assignment_for(allocation_unit_id, 1)
            except KeyError:
                flash(
                    "当前工作区尚未登记到冻结的共同锚分配计划，请联系管理员。",
                    "error",
                )
                return redirect(url_for("recommendations"))
            direction = {
                **direction,
                "anchor_allocation_mode": "precomputed_queue_balanced",
                "anchor_plan_hash": anchor_plan.plan_hash,
                "allocation_unit_id": allocation_unit_id,
            }
        else:
            direction = {
                **direction,
                "anchor_allocation_mode": "development_hash_fallback",
            }
        create_simulation(
            user_id,
            str(rec["id"]),
            direction,
            EntrepreneurshipState().to_dict(),
        )
        flash("连续创业模拟已创建。", "success")
        return redirect(url_for("simulation_event"))

    @app.route("/simulation", methods=["GET"])
    @login_required
    def simulation_dashboard() -> str | Response:
        user_id = _current_user_id()
        simulation = active_simulation(user_id)
        if simulation is None:
            return redirect(url_for("recommendations"))
        try:
            _assert_simulation_anchor_binding(
                simulation,
                user_id=user_id,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
                plan_required=plan_required,
                next_turn=int(simulation["turn_index"]) + 1,
            )
            pending = pending_delivery(simulation["id"])
            if pending is not None:
                _assert_delivery_anchor_binding(
                    pending,
                    simulation,
                    anchor_plan=anchor_plan,
                    anchor_every=scheduler.anchor_every,
                )
        except (SchedulingError, ValueError) as error:
            return render_template("error.html", message=str(error), status=409), 409
        history = simulation_history(simulation["id"], user_id)
        return render_template(
            "simulation_dashboard.html",
            simulation=simulation,
            pending=_event_view(pending) if pending else None,
            history=[item for item in history if item.get("response_id")],
        )

    @app.route("/simulation/event", methods=["GET"])
    @login_required
    def simulation_event() -> str | Response:
        user_id = _current_user_id()
        simulation = active_simulation(user_id)
        if simulation is None:
            return redirect(url_for("recommendations"))
        try:
            _assert_simulation_anchor_binding(
                simulation,
                user_id=user_id,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
                plan_required=plan_required,
                next_turn=int(simulation["turn_index"]) + 1,
            )
            existing = pending_delivery(simulation["id"])
            if existing is not None:
                _assert_delivery_anchor_binding(
                    existing,
                    simulation,
                    anchor_plan=anchor_plan,
                    anchor_every=scheduler.anchor_every,
                )
        except (SchedulingError, ValueError) as error:
            return render_template("error.html", message=str(error), status=409), 409
        if existing is not None:
            return render_template(
                "simulation_event.html",
                simulation=simulation,
                delivery=_event_view(existing),
            )

        history = simulation_history(simulation["id"], user_id)
        interval_hours = int(current_app.config.get("TRAINING_INTERVAL_HOURS", 0))
        completed = [item for item in history if item.get("response_created_at")]
        if interval_hours > 0 and completed:
            last_completed = datetime.fromisoformat(completed[-1]["response_created_at"])
            if last_completed.tzinfo is None:
                last_completed = last_completed.replace(tzinfo=UTC)
            available_at = last_completed + timedelta(hours=interval_hours)
            if datetime.now(UTC) < available_at:
                flash(
                    f"下一事件将在 {available_at.astimezone().strftime('%Y-%m-%d %H:%M')} 开放。",
                    "info",
                )
                return redirect(url_for("simulation_dashboard"))
        used = [item["blueprint_id"] for item in history]
        state = EntrepreneurshipState.from_dict(simulation["main_state_json"])
        lock_key = f"event-generation:{simulation['id']}"
        lock_token = acquire_operation_lock(lock_key, ttl_seconds=600)
        if lock_token is None:
            existing = pending_delivery(simulation["id"])
            if existing is not None:
                try:
                    _assert_delivery_anchor_binding(
                        existing,
                        simulation,
                        anchor_plan=anchor_plan,
                        anchor_every=scheduler.anchor_every,
                    )
                except SchedulingError as error:
                    return render_template(
                        "error.html", message=str(error), status=409
                    ), 409
                return render_template(
                    "simulation_event.html",
                    simulation=simulation,
                    delivery=_event_view(existing),
                )
            flash("事件正在生成，请稍后刷新；系统不会并发重复调用模型。", "info")
            return redirect(url_for("simulation_dashboard"))
        try:
            existing = pending_delivery(simulation["id"])
            if existing is not None:
                _assert_delivery_anchor_binding(
                    existing,
                    simulation,
                    anchor_plan=anchor_plan,
                    anchor_every=scheduler.anchor_every,
                )
                return render_template(
                    "simulation_event.html",
                    simulation=simulation,
                    delivery=_event_view(existing),
                )
            _assert_simulation_anchor_binding(
                simulation,
                user_id=user_id,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
                plan_required=plan_required,
                next_turn=int(simulation["turn_index"]) + 1,
            )
            scheduled = scheduler.schedule(
                simulation_id=simulation["id"],
                turn_no=int(simulation["turn_index"]) + 1,
                main_state=state,
                used_blueprint_ids=used,
                direction_tags=simulation["direction_json"].get("scheduler_tags", ["general"]),
                allocation_unit_id=subject_ref(user_id),
            )
            if scheduled.event_kind == "main":
                context = {
                    **_event_llm_context(scheduled),
                    "direction": _direction_llm_context(
                        simulation["direction_json"]
                    ),
                    "method_boundary": (
                        "LLM only narrates; numeric state transition is external and deterministic."
                    ),
                }
                generated = _llm_service().generate_event(context)
                content = generated.data
                generation_call_id = _call_id(generated)
            else:
                content = _frozen_anchor_payload(scheduled)
                generation_call_id = None
            event_json = {"domain_event": scheduled.to_dict(), "content": content}
            delivery_id = save_delivery(
                user_id=user_id,
                simulation_id=simulation["id"],
                turn_no=int(simulation["turn_index"]) + 1,
                event_kind=scheduled.event_kind,
                blueprint_id=scheduled.blueprint_id,
                form_id=scheduled.form_id,
                selection_probability=scheduled.selection_probability,
                scheduler=scheduled.scheduler_audit,
                visible_state=scheduled.visible_state.to_dict(),
                event=event_json,
                generation_call_id=generation_call_id,
            )
        except (LLMConfigurationError, LLMRequestFailed, LLMRecorderError) as error:
            message, status = _safe_llm_error(error)
            return render_template("error.html", message=message, status=status), status
        except (SchedulingError, ValueError) as error:
            return render_template("error.html", message=str(error), status=409), 409
        finally:
            release_operation_lock(lock_key, lock_token)
        delivery = get_delivery(delivery_id, user_id)
        if delivery is None:
            abort(500)
        return render_template(
            "simulation_event.html",
            simulation=simulation,
            delivery=_event_view(delivery),
        )

    @app.route("/simulation/respond", methods=["POST"])
    @login_required
    def submit_response() -> Response:
        user_id = _current_user_id()
        delivery_id = _field("delivery_id", limit=100)
        delivery = get_delivery(delivery_id, user_id)
        if delivery is None:
            abort(404)
        simulation = get_simulation(delivery["simulation_id"], user_id)
        if simulation is None:
            abort(404)
        try:
            _assert_simulation_anchor_binding(
                simulation,
                user_id=user_id,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
                plan_required=plan_required,
                next_turn=int(simulation["turn_index"]) + 1,
            )
            _assert_delivery_anchor_binding(
                delivery,
                simulation,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
            )
        except SchedulingError as error:
            return render_template("error.html", message=str(error), status=409), 409
        if delivery["status"] != "pending":
            flash("该事件已经提交。", "info")
            return redirect(url_for("simulation_history"))
        lock_key = f"response-scoring:{delivery_id}"
        lock_token = acquire_operation_lock(lock_key, ttl_seconds=600)
        if lock_token is None:
            flash("该回答正在处理，请勿重复提交。", "info")
            return redirect(url_for("simulation_event"))
        max_chars = int(current_app.config["MAX_DECISION_TEXT_CHARS"])
        try:
            delivery = get_delivery(delivery_id, user_id)
            if delivery is None:
                raise LookupError("Event delivery was not found.")
            if delivery["status"] != "pending":
                flash("该事件已经提交。", "info")
                return redirect(url_for("simulation_history"))
            simulation = get_simulation(delivery["simulation_id"], user_id)
            if simulation is None:
                raise LookupError("Simulation not found.")
            _assert_simulation_anchor_binding(
                simulation,
                user_id=user_id,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
                plan_required=plan_required,
                next_turn=int(simulation["turn_index"]) + 1,
            )
            _assert_delivery_anchor_binding(
                delivery,
                simulation,
                anchor_plan=anchor_plan,
                anchor_every=scheduler.anchor_every,
            )
            response = {
                "decision_text": _field("decision_text", limit=max_chars),
                "rationale_text": _field("rationale_text", limit=max_chars),
                "information_requests": _field("information_requests", limit=max_chars),
                "revision_text": _field("revision_text", limit=max_chars),
            }
            if not response["decision_text"]:
                raise ValueError("请填写你的决策。")
            if not response["rationale_text"]:
                raise ValueError("请填写理由与证据权衡。")
            _reject_detectable_identifiers(response)
            response = sanitize_for_llm(response)
            event = _scheduled_from_delivery(delivery)
            history = simulation_history(str(simulation["id"]), user_id)
            context = {
                **_event_llm_context(event),
                "presented_event": delivery["event_json"]["content"],
                "response": response,
                "prior_process_trace": _prior_process_trace(history),
                "trace_boundary": (
                    "历史仅用于识别本轮是否延续、修正或重复先前策略；"
                    "每个分数仍须由当前回答中的可观察证据支持。"
                ),
                "rubric_notice": "Scores are uncalibrated process-evidence labels, not ability claims.",
            }
            scored = _llm_service().score_decision(context)
            mapping = scored.data["mapping_uncertainty"]
            confidence = float(scored.data["classification_confidence"])
            flags = set(scored.data["safety_flags"])
            if flags - {"none"}:
                flash(
                    "系统检测到回答可能包含隐私、提示注入或高风险建议内容；"
                    "本轮尚未推进。请删去相关内容并以教学决策语言重新提交。",
                    "error",
                )
                return redirect(url_for("simulation_event"))
            if mapping["level"] != "low" or confidence < 0.70:
                flash(
                    "系统无法足够确定地把开放回答映射到规则行动；本轮尚未推进。"
                    "请补充具体行动、先后顺序或停止条件后重新提交。",
                    "error",
                )
                return redirect(url_for("simulation_event"))
            main_state = EntrepreneurshipState.from_dict(simulation["main_state_json"])
            transition = apply_classified_action(
                library=library,
                event=event,
                main_state=main_state,
                action_code=scored.data["action_code"],
            )
            feedback = {
                **scored.data["feedback"],
                "mapping_uncertainty": scored.data["mapping_uncertainty"],
                "safety_flags": scored.data["safety_flags"],
                "transition": transition.to_dict(),
                "interpretation_boundary": (
                    "ECD 等级是本轮文本的待验证过程编码，不是已校准能力分或现实创业预测。"
                ),
            }
            complete_delivery(
                user_id=user_id,
                delivery_id=delivery_id,
                decision_text=str(response["decision_text"]),
                rationale_text=str(response["rationale_text"]),
                information_requests=str(response["information_requests"]),
                revision_text=str(response["revision_text"]),
                action_code=scored.data["action_code"],
                classification_confidence=float(scored.data["classification_confidence"]),
                evidence={"evidence": scored.data["evidence"]},
                feedback=feedback,
                main_state_before=transition.main_state_before.to_dict(),
                main_state_after=transition.main_state_after.to_dict(),
                sandbox_state_after=(
                    transition.sandbox_state_after.to_dict()
                    if transition.sandbox_state_after
                    else None
                ),
                scoring_call_id=_call_id(scored),
            )
        except (StateTransitionError, SchedulingError) as error:
            flash(f"状态转换被安全阻断：{error}", "error")
            return redirect(url_for("simulation_dashboard"))
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("simulation_event"))
        except (LLMConfigurationError, LLMRequestFailed, LLMRecorderError) as error:
            message, _status = _safe_llm_error(error)
            flash(message, "error")
            return redirect(url_for("simulation_event"))
        except LookupError as error:
            flash(f"状态转换被安全阻断：{error}", "error")
            return redirect(url_for("simulation_dashboard"))
        finally:
            release_operation_lock(lock_key, lock_token)
        flash("本轮决策已记录；数值状态仅由规则引擎更新。", "success")
        return redirect(url_for("simulation_history"))

    @app.route("/simulation/history", methods=["GET"], endpoint="simulation_history")
    @login_required
    def simulation_history_view() -> str | Response:
        simulation = active_simulation(_current_user_id())
        if simulation is None:
            return redirect(url_for("recommendations"))
        history = simulation_history(simulation["id"], _current_user_id())
        return render_template(
            "simulation_history.html", simulation=simulation, history=history
        )

    @app.route("/data", methods=["GET"])
    @login_required
    def data_controls() -> str:
        user_id = _current_user_id()
        payload = export_user_data(user_id)
        histories = payload.get("histories", {})
        history_count = sum(
            len(items) for items in histories.values() if isinstance(items, list)
        )
        inventory = [
            {
                "label": "画像版本",
                "count": len(payload.get("profiles", [])),
                "description": "去标识化资料、画像条目与学习者修订。",
                "url": url_for("profile_review"),
            },
            {
                "label": "方向推荐",
                "count": len(payload.get("recommendations", [])),
                "description": "推荐依据、不确定性、反对理由与验证步骤。",
                "url": url_for("recommendations"),
            },
            {
                "label": "模拟事件",
                "count": history_count,
                "description": "已投放及已完成的主线或共同锚事件。",
                "url": url_for("simulation_history"),
            },
            {
                "label": "LLM 调用记录",
                "count": len(payload.get("llm_call_metadata", [])),
                "description": "模型、提示版本、调用状态及结构化输出记录。",
            },
        ]
        return render_template(
            "data_controls.html",
            user=g.user,
            workspace_id=subject_ref(user_id),
            created_at=g.user.get("created_at"),
            storage_updated_at=g.user.get("updated_at"),
            data_inventory=inventory,
        )

    @app.route("/data/export", methods=["GET"])
    @login_required
    def export_data() -> Response:
        payload = export_user_data(_current_user_id())
        response = Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            mimetype="application/json",
        )
        response.headers["Content-Disposition"] = "attachment; filename=venturebridge-export.json"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/data/delete", methods=["POST"])
    @login_required
    def delete_data() -> Response:
        if (
            _field("confirmation", limit=20) != "删除全部数据"
            or request.form.get("understand_irreversible") not in {"on", "yes", "1"}
        ):
            flash("请完整输入“删除全部数据”并确认不可恢复。", "error")
            return redirect(url_for("data_controls"))
        receipt = delete_user_data(_current_user_id())
        session.clear()
        return render_template("deletion_complete.html", receipt=receipt)

    @app.route("/transparency", methods=["GET"])
    def transparency() -> str:
        return render_template(
            "transparency.html",
            library_version=library.library_version,
            scheduler_version=scheduler.VERSION,
        )

    @app.route("/health", methods=["GET"])
    def health() -> Response:
        provider = current_app.extensions.get("venturebridge_llm_provider")
        if provider is not None:
            readiness = {
                "ready": True,
                "errors": [],
                "warnings": ["injected_provider"],
                "model": getattr(provider, "model", None),
                "api_version": getattr(provider, "api_version", None),
                "build_id": os.getenv("APP_BUILD_VERSION") or os.getenv("APP_BUILD_ID") or "unversioned",
            }
        else:
            readiness = check_llm_readiness().to_public_dict()
        return jsonify(
            {
                "status": "ok",
                "llm_configured": readiness["ready"],
                "llm_readiness": readiness,
                "blueprint_library_version": library.library_version,
                "anchor_allocation_mode": (
                    "precomputed_queue_balanced"
                    if anchor_plan is not None
                    else (
                        "commissioning_blocked"
                        if plan_required
                        else "development_hash_fallback"
                    )
                ),
                "anchor_allocation_ready": (
                    anchor_plan is not None
                    or (
                        str(current_app.config.get("ENVIRONMENT")) == "development"
                        and not plan_required
                    )
                ),
                "anchor_plan_required": plan_required,
                "anchor_plan_configured": anchor_plan is not None,
                "anchor_plan_hash": anchor_plan.plan_hash if anchor_plan is not None else None,
                "anchor_plan_checkpoint_count": (
                    anchor_plan.checkpoint_count if anchor_plan is not None else None
                ),
                "anchor_every_turns": scheduler.anchor_every,
                "scientific_validation_status": "not_run",
            }
        )
