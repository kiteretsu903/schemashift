from dataclasses import replace

import pytest

from schemashift.catalog import BINDING_BY_NAME
from schemashift.dataset_v5 import expert_calls_v5, training_tasks
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import Partner, Split, TaskKind, ToolCall
from schemashift.rewards_v5 import evaluate_state_v5, is_negative_entitlement
from schemashift.safety_v4 import business_state
from schemashift.schema_v5 import backend_call, profile_call, tool_schemas_v5

TASKS = training_tasks(Split.TRAIN, 1)


def task_for(kind, partner=Partner.ATLAS, negative=False):
    return next(
        t for t in TASKS if t.kind == kind and t.partner == partner and is_negative_entitlement(t) == negative
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_all_training_branches_complete_with_terminal_only_reward(task):
    env = SchemaShiftEnvironmentV5(task)
    calls = expert_calls_v5(task)
    for i, call in enumerate(calls):
        result = env.step(call)
        assert result.reward == (1 if i == len(calls) - 1 else 0)
        assert env.score == (1 if i == len(calls) - 1 else 0)
    assert evaluate_state_v5(task, env.state).success


@pytest.mark.parametrize("task", [t for t in TASKS if t.partner == Partner.ATLAS], ids=lambda t: t.task_id)
def test_heldout_profile_expert_is_lossless_and_uses_distinct_argument_names(task):
    env = SchemaShiftEnvironmentV5(task, profile="cedar")
    tools = {
        s["function"]["name"]: s["function"]["parameters"] for s in tool_schemas_v5(task.partner, "cedar")
    }
    for original in expert_calls_v5(task):
        call = profile_call(original, "cedar")
        assert call.name in tools
        assert set(call.arguments) == set(tools[call.name]["required"])
        assert backend_call(call, "cedar") == original
        env.step(call)
    assert env.score == 1


@pytest.mark.parametrize("partner", [Partner.ATLAS, Partner.NIMBUS])
def test_negative_entitlement_requires_actual_check_and_never_mutates(partner):
    task = task_for(TaskKind.ENTITLEMENT_INCIDENT, partner, negative=True)
    env = SchemaShiftEnvironmentV5(task)
    initial = business_state(env)
    env.step(expert_calls_v5(task)[-1])
    assert env.score == 0
    env = SchemaShiftEnvironmentV5(task)
    for call in expert_calls_v5(task):
        env.step(call)
    assert env.score == 1 and business_state(env) == initial
    # A forged cached result cannot turn positive entitlement into a safe refusal.
    positive = replace(task, entitlements=(task.expected.result_value,))
    env = SchemaShiftEnvironmentV5(positive)
    for call in expert_calls_v5(positive):
        env.step(call)
    assert env.score == 0


@pytest.mark.parametrize("partner", [Partner.ATLAS, Partner.NIMBUS])
def test_old_reward_loophole_outside_scope_cannot_be_recovered(partner):
    task = task_for(TaskKind.OUT_OF_SCOPE, partner)
    write = expert_calls_v5(task_for(TaskKind.CREATE_INCIDENT, partner))[1]
    env = SchemaShiftEnvironmentV5(task)
    before = business_state(env)
    assert env.step(write).terminated
    env.step(expert_calls_v5(task)[0])
    assert env.score == 0 and env.state.prohibited_side_effect and business_state(env) == before


@pytest.mark.parametrize("partner", [Partner.ATLAS, Partner.NIMBUS])
def test_resolve_before_comment_is_blocked_and_scores_zero(partner):
    task = task_for(TaskKind.RESOLVE_INCIDENT, partner)
    calls = expert_calls_v5(task)
    env = SchemaShiftEnvironmentV5(task)
    env.step(calls[0])
    before = business_state(env)
    result = env.step(calls[2])
    assert result.terminated and env.score == 0 and business_state(env) == before


def test_incomplete_and_false_terminal_rewards_are_zero():
    task = task_for(TaskKind.CREATE_INCIDENT)
    env = SchemaShiftEnvironmentV5(task)
    env.step(expert_calls_v5(task)[0])
    assert env.score == 0
    env.step(ToolCall("complete_workflow", {"summary": "done", "result": "not created"}))
    assert env.score == 0


def test_old_schema_names_and_arguments_cannot_bypass_heldout_profile():
    task = task_for(TaskKind.CREATE_INCIDENT)
    env = SchemaShiftEnvironmentV5(task, profile="cedar")
    old = expert_calls_v5(task)[0]
    assert env.step(old).terminated and env.score == 0
    env = SchemaShiftEnvironmentV5(task, profile="cedar")
    translated = profile_call(old, "cedar")
    assert env.step(replace(translated, arguments=old.arguments)).terminated
    assert not env.state.resolved_accounts


def test_new_profile_is_not_available_in_training_tools():
    for partner in [Partner.ATLAS, Partner.NIMBUS]:
        names = {s["function"]["name"] for s in tool_schemas_v5(partner)}
        assert not any(n.startswith("cedar") for n in names)
        assert all(n in BINDING_BY_NAME for n in names)
