from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Split(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    PRIVATE_TEST = "private_test"


class Partner(StrEnum):
    ATLAS = "atlas"
    NIMBUS = "nimbus"
    ORBIT = "orbit"


class TaskKind(StrEnum):
    CREATE_INCIDENT = "create_incident"
    ENTITLEMENT_INCIDENT = "entitlement_incident"
    RESOLVE_INCIDENT = "resolve_incident"
    UPDATE_CONTACT = "update_contact"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_APPROVAL = "needs_approval"
    READ_ONLY = "read_only"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class Account:
    ref: str
    email: str
    contact_email: str
    plan: str


@dataclass(frozen=True)
class Incident:
    ref: str
    account_ref: str
    description: str
    urgency: str
    status: str = "open"
    comments: tuple[str, ...] = ()
    resolution: str | None = None


@dataclass(frozen=True)
class ExpectedOutcome:
    terminal_action: str
    account_ref: str | None = None
    incident_ref: str | None = None
    incident_description: str | None = None
    urgency: str | None = None
    required_comment: str | None = None
    resolution: str | None = None
    new_contact_email: str | None = None
    clarification_field: str | None = None
    approval_action: str | None = None
    result_value: str | None = None
    decline_reason: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: Split
    partner: Partner
    kind: TaskKind
    seed: int
    template_id: str
    user_request: str
    account: Account
    entitlements: tuple[str, ...]
    incidents: tuple[Incident, ...]
    expected: ExpectedOutcome
    authorization_token: str | None = None
    max_steps: int = 6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        expected = ExpectedOutcome(**data["expected"])
        return cls(
            task_id=data["task_id"],
            split=Split(data["split"]),
            partner=Partner(data["partner"]),
            kind=TaskKind(data["kind"]),
            seed=int(data["seed"]),
            template_id=data["template_id"],
            user_request=data["user_request"],
            account=Account(**data["account"]),
            entitlements=tuple(data["entitlements"]),
            incidents=tuple(
                Incident(
                    **{
                        **incident,
                        "comments": tuple(incident.get("comments", ())),
                    }
                )
                for incident in data["incidents"]
            ),
            expected=expected,
            authorization_token=data.get("authorization_token"),
            max_steps=int(data.get("max_steps", 6)),
        )


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class Transition:
    step: int
    call: ToolCall
    observation: dict[str, Any]
    reward: float
    terminated: bool
    violation: str | None = None
    state_digest: str = ""


@dataclass
class EpisodeState:
    account: Account
    entitlements: set[str]
    incidents: dict[str, Incident]
    resolved_accounts: set[str] = field(default_factory=set)
    verified_entitlements: set[tuple[str, str]] = field(default_factory=set)
    resolved_incidents: set[str] = field(default_factory=set)
    terminal_action: str | None = None
    terminal_payload: dict[str, Any] = field(default_factory=dict)
    prohibited_side_effect: bool = False
    violation: str | None = None
    steps: int = 0
    transitions: list[Transition] = field(default_factory=list)


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class RewardBreakdown:
    score: float
    milestones: float
    terminal: float
    safety: float
    efficiency: float
    success: bool
    prohibited_side_effect: bool
    valid_trajectory: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
