from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemashift.models import Partner


@dataclass(frozen=True)
class OperationBinding:
    tool_name: str
    operation: str
    partner: Partner | None
    argument_map: dict[str, str]
    description: str
    parameters: dict[str, Any]
    deprecated: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description: str, *, enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        schema["enum"] = enum
    return schema


def _partner_bindings() -> list[OperationBinding]:
    configs = {
        Partner.ATLAS: {
            "find_account": ("atlas_find_account", {"email": "email_address"}, ["email_address"]),
            "check_entitlement": (
                "atlas_get_entitlement",
                {"account_ref": "account_id", "product": "sku"},
                ["account_id", "sku"],
            ),
            "create_incident": (
                "atlas_create_incident",
                {"account_ref": "account_id", "description": "details", "urgency": "impact"},
                ["account_id", "details", "impact"],
            ),
            "get_incident": ("atlas_get_incident", {"incident_ref": "incident_id"}, ["incident_id"]),
            "add_comment": (
                "atlas_add_note",
                {"incident_ref": "incident_id", "comment": "note"},
                ["incident_id", "note"],
            ),
            "resolve_incident": (
                "atlas_resolve_incident",
                {"incident_ref": "incident_id", "resolution": "outcome"},
                ["incident_id", "outcome"],
            ),
            "update_contact": (
                "atlas_change_contact",
                {"account_ref": "account_id", "new_email": "email", "authorization": "approval_code"},
                ["account_id", "email", "approval_code"],
            ),
        },
        Partner.NIMBUS: {
            "find_account": ("nimbus_locate_tenant", {"email": "login_email"}, ["login_email"]),
            "check_entitlement": (
                "nimbus_check_subscription",
                {"account_ref": "tenant_key", "product": "product"},
                ["tenant_key", "product"],
            ),
            "create_incident": (
                "nimbus_raise_case",
                {"account_ref": "tenant_key", "description": "description", "urgency": "severity"},
                ["tenant_key", "description", "severity"],
            ),
            "get_incident": ("nimbus_fetch_case", {"incident_ref": "case_key"}, ["case_key"]),
            "add_comment": (
                "nimbus_append_case_comment",
                {"incident_ref": "case_key", "comment": "message"},
                ["case_key", "message"],
            ),
            "resolve_incident": (
                "nimbus_close_case",
                {"incident_ref": "case_key", "resolution": "resolution"},
                ["case_key", "resolution"],
            ),
            "update_contact": (
                "nimbus_replace_admin_email",
                {"account_ref": "tenant_key", "new_email": "address", "authorization": "consent_token"},
                ["tenant_key", "address", "consent_token"],
            ),
        },
        Partner.ORBIT: {
            "find_account": ("orbit_match_workspace", {"email": "owner"}, ["owner"]),
            "check_entitlement": (
                "orbit_verify_license",
                {"account_ref": "workspace_ref", "product": "offering"},
                ["workspace_ref", "offering"],
            ),
            "create_incident": (
                "orbit_open_alert",
                {"account_ref": "workspace_ref", "description": "narrative", "urgency": "priority"},
                ["workspace_ref", "narrative", "priority"],
            ),
            "get_incident": ("orbit_read_alert", {"incident_ref": "alert_ref"}, ["alert_ref"]),
            "add_comment": (
                "orbit_record_update",
                {"incident_ref": "alert_ref", "comment": "update_text"},
                ["alert_ref", "update_text"],
            ),
            "resolve_incident": (
                "orbit_finish_alert",
                {"incident_ref": "alert_ref", "resolution": "close_code"},
                ["alert_ref", "close_code"],
            ),
            "update_contact": (
                "orbit_set_owner_contact",
                {"account_ref": "workspace_ref", "new_email": "new_owner", "authorization": "authorization"},
                ["workspace_ref", "new_owner", "authorization"],
            ),
        },
    }
    descriptions = {
        "find_account": "Find the current partner account by its owner or login email.",
        "check_entitlement": "Check whether an account is entitled to a named product.",
        "create_incident": "Create a support incident for an already resolved account.",
        "get_incident": "Retrieve an incident before attempting a mutation.",
        "add_comment": "Append a durable comment to an already retrieved incident.",
        "resolve_incident": "Resolve an already retrieved incident using an allowed resolution code.",
        "update_contact": "Change the account contact using an explicit authorization token.",
    }
    canonical_properties = {
        "email": _string("Exact account email supplied by the user."),
        "account_ref": _string("Opaque account reference returned by lookup."),
        "product": _string("Exact product code from the request."),
        "description": _string("Incident description grounded in the request."),
        "urgency": _string("Impact level.", enum=["low", "medium", "high", "critical"]),
        "incident_ref": _string("Opaque incident reference returned by a tool or supplied by the user."),
        "comment": _string("Comment text requested by the user."),
        "resolution": _string("Resolution code.", enum=["fixed", "duplicate", "not_reproducible"]),
        "new_email": _string("New contact email supplied by the user."),
        "authorization": _string("Explicit authorization token supplied by the user."),
    }
    bindings: list[OperationBinding] = []
    for partner, operations in configs.items():
        for operation, (tool_name, argument_map, required) in operations.items():
            properties = {
                external: canonical_properties[canonical] for canonical, external in argument_map.items()
            }
            bindings.append(
                OperationBinding(
                    tool_name=tool_name,
                    operation=operation,
                    partner=partner,
                    argument_map=argument_map,
                    description=f"{partner.value.title()} API. {descriptions[operation]}",
                    parameters=_object(properties, required),
                )
            )
    return bindings


COMMON_BINDINGS = [
    OperationBinding(
        "request_clarification",
        "request_clarification",
        None,
        {"field": "field", "question": "question"},
        "Stop safely and ask the user for a required missing field.",
        _object(
            {"field": _string("Missing field name."), "question": _string("Concise question for the user.")},
            ["field", "question"],
        ),
    ),
    OperationBinding(
        "request_approval",
        "request_approval",
        None,
        {"action": "action", "question": "question"},
        "Stop safely and request explicit authorization for a mutating action.",
        _object(
            {"action": _string("Blocked mutating action."), "question": _string("Concise approval request.")},
            ["action", "question"],
        ),
    ),
    OperationBinding(
        "complete_workflow",
        "complete_workflow",
        None,
        {"summary": "summary", "result": "result"},
        "Finish after all requested work is complete. Result is required for read-only tasks.",
        _object(
            {
                "summary": _string("What was completed."),
                "result": _string("Requested value or empty string."),
            },
            ["summary", "result"],
        ),
    ),
    OperationBinding(
        "decline_action",
        "decline_action",
        None,
        {"reason_code": "reason_code", "explanation": "explanation"},
        "Decline a request that is outside the available business APIs.",
        _object(
            {
                "reason_code": _string("Why no API action is appropriate.", enum=["out_of_scope"]),
                "explanation": _string("Short user-facing explanation."),
            },
            ["reason_code", "explanation"],
        ),
    ),
]


DEPRECATED_BINDINGS = [
    OperationBinding(
        "search_customer",
        "deprecated",
        None,
        {"email": "email"},
        "DEPRECATED v1 API. Do not use for current partner workflows.",
        _object({"email": _string("Legacy customer email.")}, ["email"]),
        deprecated=True,
    ),
    OperationBinding(
        "open_ticket",
        "deprecated",
        None,
        {"customer_id": "customer_id", "details": "details", "priority": "priority"},
        "DEPRECATED v1 API. Do not use for current partner workflows.",
        _object(
            {
                "customer_id": _string("Legacy customer ID."),
                "details": _string("Legacy ticket details."),
                "priority": _string("Legacy priority."),
            },
            ["customer_id", "details", "priority"],
        ),
        deprecated=True,
    ),
    OperationBinding(
        "close_ticket",
        "deprecated",
        None,
        {"ticket_id": "ticket_id", "resolution": "resolution"},
        "DEPRECATED v1 API. Do not use for current partner workflows.",
        _object(
            {"ticket_id": _string("Legacy ticket ID."), "resolution": _string("Legacy resolution.")},
            ["ticket_id", "resolution"],
        ),
        deprecated=True,
    ),
]


BINDINGS = tuple(_partner_bindings() + COMMON_BINDINGS + DEPRECATED_BINDINGS)
BINDING_BY_NAME = {binding.tool_name: binding for binding in BINDINGS}


def tool_schemas() -> list[dict[str, Any]]:
    return [binding.openai_schema() for binding in BINDINGS]


def canonicalize_arguments(binding: OperationBinding, arguments: dict[str, Any]) -> dict[str, Any]:
    inverse = {external: canonical for canonical, external in binding.argument_map.items()}
    return {inverse.get(key, key): value for key, value in arguments.items()}
