"""Replay serialized curriculum examples without trusting generator labels."""
from __future__ import annotations
import json
from schemashift.agent_v5 import SYSTEM_PROMPT_V5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import TaskSpec, ToolCall
from schemashift.schema_v5 import tool_schemas_v5


def audit_prefix(row, record):
    task, profile = TaskSpec.from_dict(record['task']), record['profile']
    messages = row['messages']
    if messages[:2] != [{'role': 'system', 'content': SYSTEM_PROMPT_V5}, {'role': 'user', 'content': task.user_request}]:
        raise ValueError('Prompt drift')
    if row['tools'] != tool_schemas_v5(task.partner, profile):
        raise ValueError('Tool schema drift')
    env = SchemaShiftEnvironmentV5(task, profile)
    if len(messages) % 2 != 1:
        raise ValueError('Expected assistant/tool pairs followed by one target')
    for index in range(2, len(messages), 2):
        message = messages[index]
        if message['role'] != 'assistant' or len(message.get('tool_calls', [])) != 1:
            raise ValueError('Expected one native call')
        tool = message['tool_calls'][0]
        call = ToolCall(tool['function']['name'], json.loads(tool['function']['arguments']), tool['id'])
        if env.state.terminal_action:
            raise ValueError('Post-terminal target')
        result = env.step(call)
        if not result.observation.get('ok') or env.state.prohibited_side_effect:
            raise ValueError('Rejected/unsafe target or context')
        if index + 1 < len(messages):
            observed = messages[index + 1]
            if observed != {'role': 'tool', 'tool_call_id': tool['id'], 'name': call.name,
                'content': json.dumps(result.observation, sort_keys=True)}:
                raise ValueError('Fabricated lookup or mismatched call ID')
    return {'score': env.score, 'steps': env.state.steps, 'terminal': env.state.terminal_action,
        'prohibited': env.state.prohibited_side_effect}
