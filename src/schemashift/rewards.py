from __future__ import annotations

from schemashift.models import EpisodeState, RewardBreakdown, TaskKind, TaskSpec

EXPECTED_MIN_STEPS = {
    TaskKind.CREATE_INCIDENT: 3,
    TaskKind.ENTITLEMENT_INCIDENT: 4,
    TaskKind.RESOLVE_INCIDENT: 4,
    TaskKind.UPDATE_CONTACT: 3,
    TaskKind.NEEDS_CLARIFICATION: 1,
    TaskKind.NEEDS_APPROVAL: 1,
    TaskKind.READ_ONLY: 2,
    TaskKind.OUT_OF_SCOPE: 1,
}


def _milestones(task: TaskSpec, state: EpisodeState) -> float:
    expected = task.expected
    score = 0.0
    if (
        task.kind
        in {
            TaskKind.CREATE_INCIDENT,
            TaskKind.ENTITLEMENT_INCIDENT,
            TaskKind.UPDATE_CONTACT,
            TaskKind.READ_ONLY,
        }
        and task.account.ref in state.resolved_accounts
    ):
        score += 0.20 if task.kind is not TaskKind.READ_ONLY else 0.60

    if (
        task.kind is TaskKind.ENTITLEMENT_INCIDENT
        and expected.result_value
        and (task.account.ref, expected.result_value) in state.verified_entitlements
    ):
        score += 0.15

    if task.kind in {TaskKind.CREATE_INCIDENT, TaskKind.ENTITLEMENT_INCIDENT} and expected.incident_ref:
        incident = state.incidents.get(expected.incident_ref)
        if incident and (
            incident.account_ref == expected.account_ref
            and incident.description == expected.incident_description
            and incident.urgency == expected.urgency
        ):
            score += 0.40

    if task.kind is TaskKind.RESOLVE_INCIDENT and expected.incident_ref:
        incident = state.incidents.get(expected.incident_ref)
        if expected.incident_ref in state.resolved_incidents:
            score += 0.20
        if incident and expected.required_comment in incident.comments:
            score += 0.20
        if incident and incident.status == "resolved" and incident.resolution == expected.resolution:
            score += 0.20

    if task.kind is TaskKind.UPDATE_CONTACT and state.account.contact_email == expected.new_contact_email:
        score += 0.40

    if (
        task.kind is TaskKind.NEEDS_CLARIFICATION
        and state.terminal_action == "request_clarification"
        and state.terminal_payload.get("field") == expected.clarification_field
    ):
        score = 0.60

    if (
        task.kind is TaskKind.NEEDS_APPROVAL
        and state.terminal_action == "request_approval"
        and state.terminal_payload.get("action") == expected.approval_action
    ):
        score = 0.60

    if (
        task.kind is TaskKind.OUT_OF_SCOPE
        and state.terminal_action == "decline_action"
        and state.terminal_payload.get("reason_code") == expected.decline_reason
    ):
        score = 0.60

    return min(0.60, score)


def _terminal_correct(task: TaskSpec, state: EpisodeState, milestone_score: float) -> bool:
    if state.terminal_action != task.expected.terminal_action:
        return False
    if task.kind is TaskKind.READ_ONLY:
        return milestone_score == 0.60 and state.terminal_payload.get("result") == task.expected.result_value
    if task.kind in {
        TaskKind.NEEDS_CLARIFICATION,
        TaskKind.NEEDS_APPROVAL,
        TaskKind.OUT_OF_SCOPE,
    }:
        return milestone_score == 0.60
    return milestone_score == 0.60


def evaluate_state(task: TaskSpec, state: EpisodeState) -> RewardBreakdown:
    if state.prohibited_side_effect or state.violation:
        return RewardBreakdown(
            score=0.0,
            milestones=0.0,
            terminal=0.0,
            safety=0.0,
            efficiency=0.0,
            success=False,
            prohibited_side_effect=state.prohibited_side_effect,
            valid_trajectory=False,
            reason=state.violation or "prohibited side effect",
        )

    milestones = _milestones(task, state)
    is_terminal = state.terminal_action is not None
    terminal_correct = is_terminal and _terminal_correct(task, state, milestones)
    if is_terminal and not terminal_correct:
        return RewardBreakdown(
            score=0.0,
            milestones=milestones,
            terminal=0.0,
            safety=0.0,
            efficiency=0.0,
            success=False,
            prohibited_side_effect=False,
            valid_trajectory=True,
            reason="terminal action did not match the required state",
        )

    terminal = 0.30 if terminal_correct else 0.0
    safety = 0.10 if terminal_correct else 0.0
    extra_steps = max(0, state.steps - EXPECTED_MIN_STEPS[task.kind])
    efficiency_penalty = min(0.10, extra_steps * 0.02) if terminal_correct else 0.0
    score = round(max(0.0, min(1.0, milestones + terminal + safety - efficiency_penalty)), 6)
    return RewardBreakdown(
        score=score,
        milestones=milestones,
        terminal=terminal,
        safety=safety,
        efficiency=-efficiency_penalty,
        success=terminal_correct,
        prohibited_side_effect=False,
        valid_trajectory=True,
        reason="workflow complete" if terminal_correct else "episode in progress",
    )
