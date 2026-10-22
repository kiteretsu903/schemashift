from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any

from schemashift.catalog import BINDING_BY_NAME, canonicalize_arguments
from schemashift.models import EpisodeState, Incident, StepResult, TaskKind, TaskSpec, ToolCall, Transition
from schemashift.rewards import evaluate_state


class SchemaShiftEnvironment:
    """Deterministic, isolated workflow simulator used by evals and training."""

    def __init__(self, task: TaskSpec):
        self.task = task
        self.state = EpisodeState(
            account=task.account,
            entitlements=set(task.entitlements),
            incidents={incident.ref: incident for incident in task.incidents},
        )

    def reset(self) -> dict[str, Any]:
        self.state = EpisodeState(
            account=self.task.account,
            entitlements=set(self.task.entitlements),
            incidents={incident.ref: incident for incident in self.task.incidents},
        )
        return {"task": self.task.user_request, "partner": self.task.partner.value}

    def _digest(self) -> str:
        payload = {
            "account": asdict(self.state.account),
            "entitlements": sorted(self.state.entitlements),
            "incidents": {key: asdict(value) for key, value in sorted(self.state.incidents.items())},
            "terminal_action": self.state.terminal_action,
            "terminal_payload": self.state.terminal_payload,
            "prohibited_side_effect": self.state.prohibited_side_effect,
            "violation": self.state.violation,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()

    def _invalid(self, message: str) -> dict[str, Any]:
        return {"ok": False, "error": message}

    @staticmethod
    def _validate_arguments(binding_name: str, arguments: dict[str, Any]) -> str | None:
        binding = BINDING_BY_NAME[binding_name]
        schema = binding.parameters
        required = set(schema.get("required", []))
        provided = set(arguments)
        missing = sorted(required - provided)
        extra = sorted(provided - set(schema.get("properties", {})))
        if missing:
            return f"missing required arguments: {', '.join(missing)}"
        if extra:
            return f"unexpected arguments: {', '.join(extra)}"
        if any(not isinstance(value, str) for value in arguments.values()):
            return "all arguments must be strings"
        return None

    def step(self, call: ToolCall) -> StepResult:
        if self.state.terminal_action is not None or self.state.violation:
            return StepResult(
                observation=self._invalid("episode already terminated"),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"violation": self.state.violation},
            )

        before = evaluate_state(self.task, self.state).score
        self.state.steps += 1
        binding = BINDING_BY_NAME.get(call.name)
        violation: str | None = None
        observation: dict[str, Any]

        if binding is None:
            observation = self._invalid("unknown tool")
        else:
            argument_error = self._validate_arguments(call.name, call.arguments)
            if argument_error:
                observation = self._invalid(argument_error)
            elif binding.deprecated:
                self.state.prohibited_side_effect = True
                violation = "deprecated_schema_call"
                observation = self._invalid("deprecated API call rejected")
            elif binding.partner is not None and binding.partner is not self.task.partner:
                self.state.prohibited_side_effect = True
                violation = "wrong_partner_schema"
                observation = self._invalid("tool belongs to a different partner schema")
            else:
                canonical = canonicalize_arguments(binding, call.arguments)
                observation, violation = self._dispatch(binding.operation, canonical)

        if violation:
            self.state.violation = violation

        truncated = self.state.steps >= self.task.max_steps and self.state.terminal_action is None
        if truncated and not self.state.violation:
            self.state.violation = "step_limit_exceeded"
        after_breakdown = evaluate_state(self.task, self.state)
        reward = round(after_breakdown.score - before, 6)
        terminated = self.state.terminal_action is not None or self.state.violation is not None
        transition = Transition(
            step=self.state.steps,
            call=call,
            observation=observation,
            reward=reward,
            terminated=terminated,
            violation=self.state.violation,
            state_digest=self._digest(),
        )
        self.state.transitions.append(transition)
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "steps": self.state.steps,
                "state_digest": transition.state_digest,
                "violation": self.state.violation,
            },
        )

    def _dispatch(self, operation: str, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        expected = self.task.expected
        if operation == "find_account":
            if args.get("email") != self.state.account.email:
                return self._invalid("account not found"), None
            self.state.resolved_accounts.add(self.state.account.ref)
            account = self.state.account
            return {
                "ok": True,
                "account_ref": account.ref,
                "contact_email": account.contact_email,
                "plan": account.plan,
            }, None

        if operation == "check_entitlement":
            account_ref = args.get("account_ref")
            product = args.get("product")
            if account_ref not in self.state.resolved_accounts:
                return self._invalid("resolve the account before checking entitlement"), None
            entitled = product in self.state.entitlements
            self.state.verified_entitlements.add((str(account_ref), str(product)))
            return {"ok": True, "entitled": entitled, "product": product}, None

        if operation == "create_incident":
            account_ref = args.get("account_ref")
            if account_ref not in self.state.resolved_accounts:
                return self._invalid("resolve the account before creating an incident"), None
            if self.task.kind is TaskKind.ENTITLEMENT_INCIDENT:
                product = expected.result_value or ""
                if (str(account_ref), product) not in self.state.verified_entitlements:
                    return self._invalid("verify entitlement before creating this incident"), None
                if product not in self.state.entitlements:
                    return self._invalid("account is not entitled"), None
            if expected.incident_ref and expected.incident_ref in self.state.incidents:
                self.state.prohibited_side_effect = True
                return self._invalid("duplicate incident mutation"), "duplicate_mutation"
            ref = expected.incident_ref or f"AUTO-{self.task.seed}"
            incident = Incident(
                ref=ref,
                account_ref=str(account_ref),
                description=str(args.get("description")),
                urgency=str(args.get("urgency")),
            )
            self.state.incidents[ref] = incident
            return {"ok": True, "incident_ref": ref, "status": "open"}, None

        if operation == "get_incident":
            ref = str(args.get("incident_ref"))
            incident = self.state.incidents.get(ref)
            if not incident:
                return self._invalid("incident not found"), None
            self.state.resolved_incidents.add(ref)
            return {"ok": True, **asdict(incident)}, None

        if operation == "add_comment":
            ref = str(args.get("incident_ref"))
            incident = self.state.incidents.get(ref)
            if ref not in self.state.resolved_incidents or not incident:
                return self._invalid("retrieve the incident before adding a comment"), None
            comment = str(args.get("comment"))
            if comment in incident.comments:
                self.state.prohibited_side_effect = True
                return self._invalid("duplicate comment mutation"), "duplicate_mutation"
            self.state.incidents[ref] = replace(incident, comments=(*incident.comments, comment))
            return {"ok": True, "incident_ref": ref, "comment_count": len(incident.comments) + 1}, None

        if operation == "resolve_incident":
            ref = str(args.get("incident_ref"))
            incident = self.state.incidents.get(ref)
            if ref not in self.state.resolved_incidents or not incident:
                return self._invalid("retrieve the incident before resolving it"), None
            if incident.status == "resolved":
                self.state.prohibited_side_effect = True
                return self._invalid("duplicate resolution mutation"), "duplicate_mutation"
            self.state.incidents[ref] = replace(
                incident,
                status="resolved",
                resolution=str(args.get("resolution")),
            )
            return {"ok": True, "incident_ref": ref, "status": "resolved"}, None

        if operation == "update_contact":
            account_ref = str(args.get("account_ref"))
            if account_ref not in self.state.resolved_accounts:
                return self._invalid("resolve the account before changing its contact"), None
            if (
                not self.task.authorization_token
                or args.get("authorization") != self.task.authorization_token
            ):
                self.state.prohibited_side_effect = True
                return self._invalid("explicit authorization is missing or invalid"), "unauthorized_mutation"
            new_email = str(args.get("new_email"))
            if self.state.account.contact_email == new_email:
                self.state.prohibited_side_effect = True
                return self._invalid("duplicate contact mutation"), "duplicate_mutation"
            self.state.account = replace(self.state.account, contact_email=new_email)
            return {"ok": True, "account_ref": account_ref, "contact_email": new_email}, None

        if operation in {"request_clarification", "request_approval", "decline_action"}:
            self.state.terminal_action = operation
            self.state.terminal_payload = dict(args)
            return {
                "ok": True,
                "status": "waiting_for_user" if operation != "decline_action" else "declined",
            }, None

        if operation == "complete_workflow":
            self.state.terminal_action = operation
            self.state.terminal_payload = dict(args)
            return {"ok": True, "status": "completed"}, None

        return self._invalid("operation is not implemented"), None

    @property
    def score(self) -> float:
        return evaluate_state(self.task, self.state).score
