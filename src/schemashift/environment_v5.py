"""Guarded v5 environment with explicit negative-entitlement completion."""

from dataclasses import replace
from typing import Any

from schemashift.catalog import BINDING_BY_NAME
from schemashift.models import StepResult, TaskSpec, ToolCall
from schemashift.rewards_v5 import evaluate_state_v5
from schemashift.safety_v4 import MUTATING_OPERATIONS, GuardedSchemaShiftEnvironment
from schemashift.schema_v5 import backend_call, tool_schemas_v5
from schemashift.simulator import SchemaShiftEnvironment

VERSION = "schemashift-guarded-v5-1"


class SchemaShiftEnvironmentV5(GuardedSchemaShiftEnvironment):
    def __init__(self, task: TaskSpec, profile: str | None = None):
        super().__init__(task)
        self.profile = profile or task.partner.value
        self.exposed_tools = {s["function"]["name"] for s in tool_schemas_v5(task.partner, profile)}
        self.exposed_parameters = {
            s["function"]["name"]: s["function"]["parameters"] for s in tool_schemas_v5(task.partner, profile)
        }

    @staticmethod
    def _validate_arguments(binding_name: str, arguments: dict[str, Any]) -> str | None:
        if binding_name == "decline_action" and arguments.get("reason_code") == "not_entitled":
            return SchemaShiftEnvironment._validate_arguments(binding_name, arguments)
        return GuardedSchemaShiftEnvironment._validate_arguments(binding_name, arguments)

    def step(self, call: ToolCall) -> StepResult:
        before = self.score
        transition_count = len(self.state.transitions)
        translated = backend_call(call, self.profile)
        parameters = self.exposed_parameters.get(call.name)
        invalid_profile_fields = bool(parameters) and (
            not set(parameters["required"]).issubset(call.arguments)
            or bool(set(call.arguments) - set(parameters["properties"]))
        )
        if (
            (call.name not in self.exposed_tools or invalid_profile_fields)
            and not self.state.violation
            and self.state.terminal_action is None
        ):
            result = self._reject_call(
                call, "invalid_api_contract", "Tool name and argument keys must match the active API profile."
            )
            binding = BINDING_BY_NAME.get(translated.name)
            result = replace(
                result,
                info={
                    **result.info,
                    "business_state_changed": False,
                    "unsafe_action_attempt": True,
                    "blocked_mutation_attempt": bool(binding and binding.operation in MUTATING_OPERATIONS),
                },
            )
        else:
            result = super().step(translated)
        reward = round(self.score - before, 6)
        if len(self.state.transitions) > transition_count:
            self.state.transitions[-1] = replace(self.state.transitions[-1], reward=reward)
        return replace(result, reward=reward, info={**result.info, "safety_version": VERSION})

    @property
    def score(self) -> float:
        return evaluate_state_v5(self.task, self.state).score
