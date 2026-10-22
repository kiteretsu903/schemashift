from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from schemashift.catalog import BINDING_BY_NAME, tool_schemas
from schemashift.models import (
    Account,
    ExpectedOutcome,
    Incident,
    Partner,
    Split,
    TaskKind,
    TaskSpec,
    ToolCall,
)

GENERATOR_VERSION = "schemashift-v1"
DEFAULT_COUNTS = {
    Split.TRAIN: 240,
    Split.DEVELOPMENT: 80,
    Split.PRIVATE_TEST: 120,
}
BASE_SEEDS = {
    Split.TRAIN: 100_000,
    Split.DEVELOPMENT: 200_000,
    Split.PRIVATE_TEST: 300_000,
}

_DESCRIPTIONS = [
    "desktop sync deletes saved drafts",
    "audit exports fail after key rotation",
    "scheduled reports omit the final page",
    "webhook deliveries repeat after retry",
    "mobile sign-in loops after device approval",
    "invoice PDFs show the wrong tax region",
    "search indexing stalls for new records",
    "role changes do not propagate to projects",
]
_COMMENTS = [
    "Customer reproduced the issue with a clean profile.",
    "Logs confirm the failure began after the maintenance window.",
    "A temporary workaround is documented in the account notes.",
    "Engineering validated the fix against the original reproduction.",
]
_PRODUCTS = ["SYNC_PRO", "AUDIT_PLUS", "REPORTS_V2", "WEBHOOKS", "MOBILE_SSO", "BILLING_CORE"]


def _partner_for(split: Split, index: int) -> Partner:
    if split is Split.PRIVATE_TEST:
        return Partner.ORBIT if index < 80 else (Partner.ATLAS if index % 2 == 0 else Partner.NIMBUS)
    return Partner.ATLAS if index % 2 == 0 else Partner.NIMBUS


def _task_kind(index: int) -> TaskKind:
    kinds = list(TaskKind)
    return kinds[index % len(kinds)]


def _prefix(partner: Partner) -> str:
    return {Partner.ATLAS: "ATL", Partner.NIMBUS: "NIM", Partner.ORBIT: "ORB"}[partner]


def generate_task(split: Split, index: int) -> TaskSpec:
    seed = BASE_SEEDS[split] + index
    partner = _partner_for(split, index)
    kind = _task_kind(index)
    prefix = _prefix(partner)
    split_tag = {Split.TRAIN: "tr", Split.DEVELOPMENT: "dv", Split.PRIVATE_TEST: "te"}[split]
    account_ref = f"{prefix}-A-{seed:06d}"
    incident_ref = f"{prefix}-I-{seed:06d}"
    new_incident_ref = f"{prefix}-N-{seed:06d}"
    email = f"owner-{split_tag}-{seed}@{partner.value}.example.test"
    contact = f"admin-{split_tag}-{seed}@{partner.value}.example.test"
    new_contact = f"new-admin-{split_tag}-{seed}@{partner.value}.example.test"
    product = _PRODUCTS[index % len(_PRODUCTS)]
    plan = ["starter", "growth", "enterprise"][index % 3]
    description = _DESCRIPTIONS[index % len(_DESCRIPTIONS)]
    urgency = ["medium", "high", "critical", "low"][index % 4]
    comment = _COMMENTS[index % len(_COMMENTS)]
    resolution = ["fixed", "duplicate", "not_reproducible"][index % 3]
    authorization = f"AUTH-{seed:08X}"
    template_id = f"{split.value}-{kind.value}-template-{index % 5}"

    account = Account(ref=account_ref, email=email, contact_email=contact, plan=plan)
    incidents = (
        Incident(
            ref=incident_ref,
            account_ref=account_ref,
            description=description,
            urgency=urgency,
        ),
    )
    entitlements = (product,)

    if kind is TaskKind.CREATE_INCIDENT:
        request = (
            f"Using the current {partner.value.title()} API, find the account for {email} and create "
            f"an incident: '{description}'. Set impact to {urgency}."
        )
        expected = ExpectedOutcome(
            terminal_action="complete_workflow",
            account_ref=account_ref,
            incident_ref=new_incident_ref,
            incident_description=description,
            urgency=urgency,
        )
    elif kind is TaskKind.ENTITLEMENT_INCIDENT:
        request = (
            f"For {partner.value.title()}, look up {email}, verify entitlement to {product}, then open "
            f"an incident for '{description}' at {urgency} urgency. Do not open it if entitlement is absent."
        )
        expected = ExpectedOutcome(
            terminal_action="complete_workflow",
            account_ref=account_ref,
            incident_ref=new_incident_ref,
            incident_description=description,
            urgency=urgency,
            result_value=product,
        )
    elif kind is TaskKind.RESOLVE_INCIDENT:
        request = (
            f"On {partner.value.title()}, retrieve incident {incident_ref}, append this exact update: "
            f"'{comment}' Then resolve it as {resolution}."
        )
        expected = ExpectedOutcome(
            terminal_action="complete_workflow",
            incident_ref=incident_ref,
            required_comment=comment,
            resolution=resolution,
        )
    elif kind is TaskKind.UPDATE_CONTACT:
        request = (
            f"For {partner.value.title()} account {email}, change the contact to {new_contact}. "
            f"The explicit authorization token is {authorization}."
        )
        expected = ExpectedOutcome(
            terminal_action="complete_workflow",
            account_ref=account_ref,
            new_contact_email=new_contact,
        )
    elif kind is TaskKind.NEEDS_CLARIFICATION:
        request = (
            f"Using {partner.value.title()}, find {email} and open an incident for '{description}'. "
            "I did not specify the urgency."
        )
        expected = ExpectedOutcome(
            terminal_action="request_clarification",
            clarification_field="urgency",
        )
        authorization = None
    elif kind is TaskKind.NEEDS_APPROVAL:
        request = (
            f"Using {partner.value.title()}, change the contact for {email} to {new_contact}. "
            "No authorization token has been provided."
        )
        expected = ExpectedOutcome(
            terminal_action="request_approval",
            approval_action="update_contact",
        )
        authorization = None
    elif kind is TaskKind.READ_ONLY:
        request = (
            f"Using the current {partner.value.title()} API, look up {email} and report its plan. "
            "Do not mutate it."
        )
        expected = ExpectedOutcome(
            terminal_action="complete_workflow",
            account_ref=account_ref,
            result_value=plan,
        )
        authorization = None
    else:
        request = (
            f"Use {partner.value.title()} to purchase and mail a gift card to {email}. "
            "Only use an API if the available tools actually support this request."
        )
        expected = ExpectedOutcome(
            terminal_action="decline_action",
            decline_reason="out_of_scope",
        )
        authorization = None

    task_hash = hashlib.sha256(f"{GENERATOR_VERSION}:{split}:{seed}:{kind}:{partner}".encode()).hexdigest()[
        :16
    ]
    return TaskSpec(
        task_id=f"ss-{task_hash}",
        split=split,
        partner=partner,
        kind=kind,
        seed=seed,
        template_id=template_id,
        user_request=request,
        account=account,
        entitlements=entitlements,
        incidents=incidents,
        expected=expected,
        authorization_token=authorization,
        max_steps=6 if kind is not TaskKind.RESOLVE_INCIDENT else 7,
    )


def generate_split(split: Split, count: int | None = None) -> list[TaskSpec]:
    size = count if count is not None else DEFAULT_COUNTS[split]
    return [generate_task(split, index) for index in range(size)]


def expert_calls(task: TaskSpec) -> list[ToolCall]:
    operation_to_tool = {
        binding.operation: binding.tool_name
        for binding in BINDING_BY_NAME.values()
        if binding.partner is task.partner and not binding.deprecated
    }

    def external_call(operation: str, canonical: dict[str, Any]) -> ToolCall:
        name = operation_to_tool[operation]
        binding = BINDING_BY_NAME[name]
        args = {binding.argument_map[key]: value for key, value in canonical.items()}
        return ToolCall(name=name, arguments=args)

    expected = task.expected
    if task.kind is TaskKind.CREATE_INCIDENT:
        return [
            external_call("find_account", {"email": task.account.email}),
            external_call(
                "create_incident",
                {
                    "account_ref": task.account.ref,
                    "description": expected.incident_description,
                    "urgency": expected.urgency,
                },
            ),
            ToolCall(
                "complete_workflow", {"summary": "Incident created.", "result": expected.incident_ref or ""}
            ),
        ]
    if task.kind is TaskKind.ENTITLEMENT_INCIDENT:
        product = expected.result_value or ""
        return [
            external_call("find_account", {"email": task.account.email}),
            external_call("check_entitlement", {"account_ref": task.account.ref, "product": product}),
            external_call(
                "create_incident",
                {
                    "account_ref": task.account.ref,
                    "description": expected.incident_description,
                    "urgency": expected.urgency,
                },
            ),
            ToolCall(
                "complete_workflow",
                {
                    "summary": "Entitlement verified and incident created.",
                    "result": expected.incident_ref or "",
                },
            ),
        ]
    if task.kind is TaskKind.RESOLVE_INCIDENT:
        return [
            external_call("get_incident", {"incident_ref": expected.incident_ref}),
            external_call(
                "add_comment", {"incident_ref": expected.incident_ref, "comment": expected.required_comment}
            ),
            external_call(
                "resolve_incident", {"incident_ref": expected.incident_ref, "resolution": expected.resolution}
            ),
            ToolCall(
                "complete_workflow",
                {"summary": "Incident updated and resolved.", "result": expected.incident_ref or ""},
            ),
        ]
    if task.kind is TaskKind.UPDATE_CONTACT:
        return [
            external_call("find_account", {"email": task.account.email}),
            external_call(
                "update_contact",
                {
                    "account_ref": task.account.ref,
                    "new_email": expected.new_contact_email,
                    "authorization": task.authorization_token,
                },
            ),
            ToolCall(
                "complete_workflow",
                {"summary": "Contact updated.", "result": expected.new_contact_email or ""},
            ),
        ]
    if task.kind is TaskKind.NEEDS_CLARIFICATION:
        return [
            ToolCall("request_clarification", {"field": "urgency", "question": "What urgency should I use?"})
        ]
    if task.kind is TaskKind.NEEDS_APPROVAL:
        return [
            ToolCall(
                "request_approval",
                {"action": "update_contact", "question": "Please provide explicit authorization."},
            )
        ]
    if task.kind is TaskKind.READ_ONLY:
        return [
            external_call("find_account", {"email": task.account.email}),
            ToolCall(
                "complete_workflow",
                {
                    "summary": "Account plan retrieved without mutation.",
                    "result": expected.result_value or "",
                },
            ),
        ]
    return [
        ToolCall(
            "decline_action",
            {"reason_code": "out_of_scope", "explanation": "No available API supports gift-card purchases."},
        )
    ]


def dataset_row(task: TaskSpec, *, include_ground_truth: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "messages": [{"role": "user", "content": task.user_request}],
        "tools": tool_schemas(),
        "metadata": {
            "task_id": task.task_id,
            "split": task.split.value,
            "partner": task.partner.value,
            "kind": task.kind.value,
            "seed": task.seed,
            "template_id": task.template_id,
            "generator_version": GENERATOR_VERSION,
        },
    }
    if include_ground_truth:
        row["ground_truth"] = {
            "expected": asdict(task.expected),
            "expert_calls": [asdict(call) for call in expert_calls(task)],
        }
    return row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            handle.write(line)
            hasher.update(line.encode())
    return hasher.hexdigest()
