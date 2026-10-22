"""Explicit v5 control contract and a held-out synthetic API-renaming profile."""

from typing import Any

from schemashift.models import Partner, ToolCall
from schemashift.safety_v4 import tool_schemas_v4

# Cedar is a new tool/argument naming interface over Atlas semantics. It is not
# a new business workflow, vendor API, or a real-world transfer benchmark.
CEDAR = {
    "atlas_find_account": ("cedar_search_organization", {"email_address": "lookup_email"}),
    "atlas_get_entitlement": (
        "cedar_verify_license",
        {"account_id": "organization_key", "sku": "license_name"},
    ),
    "atlas_create_incident": (
        "cedar_open_ticket",
        {"account_id": "organization_key", "details": "issue_text", "impact": "impact_level"},
    ),
    "atlas_get_incident": ("cedar_read_ticket", {"incident_id": "ticket_key"}),
    "atlas_add_note": ("cedar_write_ticket_note", {"incident_id": "ticket_key", "note": "note_text"}),
    "atlas_resolve_incident": (
        "cedar_close_ticket",
        {"incident_id": "ticket_key", "outcome": "closure_text"},
    ),
    "atlas_change_contact": (
        "cedar_set_owner_email",
        {"account_id": "organization_key", "email": "replacement_email", "approval_code": "approval_token"},
    ),
}


def tool_schemas_v5(partner: Partner, profile: str | None = None) -> list[dict[str, Any]]:
    profile = profile or partner.value
    if profile not in (partner.value, "cedar") or (profile == "cedar" and partner is not Partner.ATLAS):
        raise ValueError("unsupported backend/profile pairing")
    schemas = tool_schemas_v4(partner)
    for item in schemas:
        function = item["function"]
        if function["name"] == "decline_action":
            function["parameters"]["properties"]["reason_code"].update(
                enum=["out_of_scope", "not_entitled"],
                description=(
                    "out_of_scope for unsupported requests; "
                    "not_entitled only after verifying absent entitlement."
                ),
            )
        if profile == "cedar" and function["name"] in CEDAR:
            name, fields = CEDAR[function["name"]]
            function["name"] = name
            function["description"] = function["description"].replace("Atlas", "Cedar")
            parameters = function["parameters"]
            parameters["properties"] = {fields.get(k, k): v for k, v in parameters["properties"].items()}
            parameters["required"] = [fields.get(k, k) for k in parameters["required"]]
    return schemas


def backend_call(call: ToolCall, profile: str) -> ToolCall:
    if profile != "cedar":
        return call
    for original, (name, fields) in CEDAR.items():
        if name == call.name:
            inverse = {value: key for key, value in fields.items()}
            return ToolCall(original, {inverse.get(k, k): v for k, v in call.arguments.items()}, call.call_id)
    return call


def profile_call(call: ToolCall, profile: str) -> ToolCall:
    if profile == "cedar" and call.name in CEDAR:
        name, fields = CEDAR[call.name]
        return ToolCall(name, {fields.get(k, k): v for k, v in call.arguments.items()}, call.call_id)
    return call
