"""VentureBridge application factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from flask import Flask, render_template

from . import auth, db, security
from .allocation import AllocationPlanError, generate_anchor_plan
from .config import load_config
from .llm import (
    FakeProvider,
    LLMConfigurationError,
    LLMRuntimeConfig,
    validate_llm_provider,
)
from .repository import subject_ref
from .routes import register_routes


DIMENSION_LABELS = {
    "opportunity_framing": "机会界定与评估",
    "evidence_falsifiability": "证据权衡与可证伪性",
    "resource_finance": "资源与财务配置",
    "adaptive_action": "不确定性下的行动与适应",
    "stakeholder_ethics": "团队、利益相关者与伦理合规",
}

STATE_LABELS = {
    "cash": "现金",
    "team": "团队",
    "product": "产品",
    "market_evidence": "市场证据",
    "reputation": "信誉",
    "compliance": "合规",
}


def create_app(
    test_config: dict[str, Any] | None = None,
    *,
    llm_provider: Any | None = None,
    llm_runtime: Any | None = None,
) -> Flask:
    """Create the site.

    ``llm_provider`` and ``llm_runtime`` are explicit dependency-injection
    seams for engineering tests.  The normal runtime never chooses a fake
    provider and resolves the real provider from environment variables only
    when an LLM feature is used, so public/transparency pages can still start
    before administrators add a secret.
    """

    load_dotenv(override=False)
    app = Flask(__name__, instance_relative_config=True)
    instance_path = security.enforce_private_directory(Path(app.instance_path))
    app.config.from_mapping(load_config(Path(app.instance_path)))
    if test_config:
        app.config.from_mapping(test_config)

    if llm_provider is not None:
        validate_llm_provider(llm_provider)
        is_test_double = isinstance(llm_provider, FakeProvider) or bool(
            getattr(llm_provider, "is_test_double", False)
        )
        if is_test_double and not app.testing:
            raise LLMConfigurationError(
                "A test-only LLM provider cannot be installed outside TESTING mode."
            )
    if llm_runtime is not None and not isinstance(llm_runtime, LLMRuntimeConfig):
        raise TypeError("llm_runtime must be an LLMRuntimeConfig instance")

    database_path = security.prepare_private_database_path(
        app.config["DATABASE"], private_root=instance_path
    )
    app.extensions["venturebridge_llm_provider"] = llm_provider
    app.extensions["venturebridge_llm_runtime"] = llm_runtime

    db.init_app(app)
    security.init_app(app)
    app.before_request(auth.load_logged_in_user)
    register_routes(app)

    app.jinja_env.globals.update(
        dimension_labels=DIMENSION_LABELS,
        state_labels=STATE_LABELS,
    )

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def render_http_error(error: Any) -> tuple[str, int]:
        status = int(getattr(error, "code", 500) or 500)
        if status >= 500:
            message = "服务器无法完成请求；错误详情已隐藏，避免暴露学习者数据。"
        else:
            message = str(getattr(error, "description", "请求无法处理。"))
        return render_template("error.html", message=message, status=status), status


    @app.cli.command("generate-anchor-plan")
    @click.option(
        "--units-file",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        required=True,
        help=(
            "JSON array of pre-pseudonymized allocation-unit IDs, or an object "
            "with an allocation_unit_ids array."
        ),
    )
    @click.option(
        "--anchor-checkpoints",
        type=click.IntRange(min=1),
        required=True,
        help="Number of non-repeating common-anchor checkpoints per unit.",
    )
    @click.option(
        "--seed",
        envvar="V_ANCHOR_PLAN_SEED",
        required=True,
        show_envvar=True,
        help=(
            "Reproducible allocation seed. Prefer V_ANCHOR_PLAN_SEED to avoid "
            "shell-history exposure; the raw seed is not written to the plan."
        ),
    )
    @click.option(
        "--output",
        type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
        required=True,
        help="New JSON plan path; an existing file is never overwritten.",
    )
    def generate_anchor_plan_command(
        units_file: Path,
        anchor_checkpoints: int,
        seed: str,
        output: Path,
    ) -> None:
        """Generate a frozen queue plan without delivering events or running an experiment."""

        try:
            raw = json.loads(units_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"Cannot read units JSON: {exc}") from exc
        if isinstance(raw, list):
            unit_ids = raw
        elif isinstance(raw, dict) and isinstance(raw.get("allocation_unit_ids"), list):
            unit_ids = raw["allocation_unit_ids"]
        else:
            raise click.ClickException(
                "Units JSON must be an array or contain an allocation_unit_ids array."
            )
        try:
            plan = generate_anchor_plan(unit_ids, anchor_checkpoints, seed)
        except AllocationPlanError as exc:
            raise click.ClickException(str(exc)) from exc

        output = output.expanduser()
        if not output.parent.exists() or not output.parent.is_dir():
            raise click.ClickException(
                f"Output parent must already be a directory: {output.parent}"
            )
        try:
            with output.open("x", encoding="utf-8") as handle:
                json.dump(plan, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        except FileExistsError as exc:
            raise click.ClickException(
                f"Refusing to overwrite existing allocation plan: {output}"
            ) from exc
        except OSError as exc:
            raise click.ClickException(f"Cannot write allocation plan: {exc}") from exc
        security.enforce_private_file(output)
        click.echo(
            f"Generated {len(plan['assignments'])} unit allocations at {output}. "
            "No event was delivered and no experiment or model fit was run."
        )
        click.echo(
            f"Register this plan hash separately before delivery: {plan['plan_hash']}"
        )

    @app.cli.command("export-anchor-units")
    @click.option(
        "--output",
        type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
        required=True,
        help="New private JSON file containing pseudonymous online allocation IDs.",
    )
    def export_anchor_units_command(output: Path) -> None:
        """Export the exact pseudonyms used by the online plan lookup."""

        unit_ids = sorted(
            subject_ref(str(row["id"]))
            for row in db.get_db().execute(
                "SELECT id FROM users WHERE consent_accepted = 1 ORDER BY id"
            ).fetchall()
        )
        if len(unit_ids) < 2:
            raise click.ClickException(
                "At least two consented workspaces are required for a linked anchor plan."
            )
        output = output.expanduser()
        if not output.parent.exists() or not output.parent.is_dir():
            raise click.ClickException(
                f"Output parent must already be a directory: {output.parent}"
            )
        payload = {
            "allocation_unit_ids": unit_ids,
            "identifier_contract": "venturebridge.repository.subject_ref(user_id)",
            "contains_direct_identifiers": False,
        }
        try:
            with output.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        except FileExistsError as exc:
            raise click.ClickException(
                f"Refusing to overwrite existing allocation-unit file: {output}"
            ) from exc
        except OSError as exc:
            raise click.ClickException(f"Cannot write allocation units: {exc}") from exc
        security.enforce_private_file(output)
        click.echo(
            f"Exported {len(unit_ids)} pseudonymous allocation units. "
            "No profile, response, outcome or direct identifier was exported."
        )

    with app.app_context():
        db.init_db()
    if database_path is not None:
        security.enforce_private_file(database_path)

    return app


__all__ = ["create_app"]
