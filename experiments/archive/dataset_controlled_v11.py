"""New SFT decision examples for v11. No evaluation fixture or model access."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

from schemashift.agent_v5 import SYSTEM_PROMPT_V5
from schemashift.catalog import BINDING_BY_NAME
from schemashift.dataset_decision_v10 import lookup_call
from schemashift.dataset_v5 import expert_calls_v5
from schemashift.environment_v5 import SchemaShiftEnvironmentV5
from schemashift.models import Account, ExpectedOutcome, Incident, Partner, Split, TaskKind, TaskSpec
from schemashift.schema_v5 import profile_call, tool_schemas_v5

OUTPUT = Path('data/generated/v11-controlled')
PROFILES = ('atlas', 'nimbus', 'cedar')

# Each tuple is an independently written narrative family, before parameter/profile substitution.
# Development uses families 8..11. No benchmark or previous generator text is read here.
UNSUPPORTED = (
    'The account login is {email}. My instruction is to issue a cash refund to {dest}; this is a payment request, not an account-contact edit.',
    'Arrange an outbound wire for the customer found under {email}. The recipient reference is {dest}. Do not substitute a support ticket for the transfer.',
    'Remove the entire tenant associated with {email}, including its stored data. This is a deletion request, not a request to change a contact.',
    'Please export the password hashes for {email}. The address {dest} identifies the export recipient, not a replacement service contact.',
    'Charge a new annual subscription to the card on file for {email}. Contact information such as {dest} is provided only for billing context.',
    'For {email}, switch the legal owner of the subscription to {dest}. A service-contact change would not fulfill the ownership transfer.',
    'Provision another administrator for {email} and assign their login to {dest}. This request concerns access roles rather than contact details.',
    'Send an external invitation email to {dest} on behalf of {email}. Creating an incident or changing the saved contact would be a different action.',
    'I need a tax receipt reissued for {email}; use {dest} only as its delivery address. No support incident is requested.',
    'Archive the whole organization logged in as {email}. The nominated coordinator is {dest}, but I am not requesting a contact replacement.',
    'Unlock the login for {email} and send its reset link to {dest}. Leave the existing service-contact field alone.',
    'Allocate a credit voucher to {email}. The beneficiary address is {dest}; the request is not a contact update.',
)
UNSUPPORTED_EXPLANATIONS = (
    'These tools cannot issue refunds.', 'There is no wire-transfer operation here.',
    'Tenant deletion is outside the available API.', 'Password-hash export is unsupported.',
    'Charging a payment card is not available.', 'These tools do not transfer legal ownership.',
    'Administrator provisioning is outside this API.', 'Sending email invitations is not supported.',
    'Tax receipt issuance is unavailable.', 'Organization archival is not offered.',
    'Login unlocking is not supported.', 'Credit voucher allocation is unavailable.',
)
ENTITLEMENT = (
    'Before filing anything for {email}, inspect whether {sku} is enabled. Only a positive finding permits this incident: "{desc}" with impact {impact}. Decline if coverage is absent.',
    'Please use the actual coverage record for {email} and product {sku} to decide this: if covered, open an {impact}-impact incident containing "{desc}"; if uncovered, decline.',
    'I authorize an incident for {email} only under active entitlement to {sku}. Check first, then either create "{desc}" at impact {impact}, or decline because coverage is missing.',
    'The requested work for {email} is conditional. Verify {sku}; an entitled account should receive a new incident "{desc}" with {impact} impact. An unentitled account should receive a refusal.',
    'For the login {email}, consult the license for {sku}. A license allows you to file "{desc}" as {impact}; no license means stop with a lack-of-entitlement refusal.',
    'Coverage must decide whether to create this case for {email}: "{desc}", impact {impact}. Look up {sku}, perform the creation if eligible, and otherwise decline.',
    'Determine from the backend whether {email} has {sku}. If it does, create the incident "{desc}" using impact {impact}. If it does not, refuse that conditional creation.',
    'This is both a coverage check and, if allowed, a filing instruction. For {email}, check {sku}; then create "{desc}" with {impact} impact only when covered. Otherwise decline.',
    'Verify eligibility for {sku} on {email} before recording "{desc}" at impact {impact}. File it on a positive result; refuse on a negative result.',
    'Check whether the customer {email} is licensed for {sku}. A positive answer authorizes "{desc}" at impact {impact}; a negative answer requires refusal.',
    'Use the entitlement service to decide a conditional request for {email}: {sku}, description "{desc}", impact {impact}. Create only when licensed and otherwise decline.',
    'Read the coverage status of {sku} for {email}, then carry out "{desc}" at {impact} only if the status permits it; reject the conditional request when it does not.',
)
ABSENT_EXPLANATIONS = (
    'The verified entitlement is absent.', 'The account lacks the checked license.',
    'Coverage was checked and this product is not included.', 'The required product entitlement was not found.',
    'The backend confirms there is no coverage.', 'The account is not licensed for that product.',
    'The entitlement check returned a negative result.', 'The required coverage is missing from this account.',
    'The checked product is unavailable under this license.', 'The license record does not cover the requested product.',
    'Verification found no entitlement for that product.', 'The coverage record did not authorize this creation.',
)
WORKFLOW = (
    'Locate incident {ref} exactly as written. Append the note "{note}" verbatim, then resolve it as {code}.',
    'On record {ref}, I need two ordered edits: add "{note}" without changing punctuation, and then set its resolution to {code}. Retrieve the record first.',
    'Please read the existing issue {ref}. Its new comment must be "{note}" exactly. After saving that comment, close the issue using {code}.',
    'Fetch {ref} and keep that full key for the updates. The required comment is "{note}"; the required closing code is {code}. Add the comment before closing.',
    'Use incident key {ref} to add "{note}" as a literal comment and then close with {code}, after reading the incident.',
    'The two operations requested for {ref} are an exact note "{note}" followed by resolution {code}. Start by retrieving this issue.',
    'Retrieve the stored incident {ref}, preserve its key, attach "{note}" unchanged and complete the closure with {code}.',
    'For existing issue {ref}, retrieve first, comment "{note}" exactly second, and resolve as {code} third.',
    'Read incident {ref} before editing it. Preserve this literal note "{note}", then use closing value {code}.',
    'Find {ref}; after retrieval, add exactly "{note}" and mark its resolution {code}.',
    'Please carry out an ordered comment and closure on {ref}: "{note}" exactly, then {code}.',
    'Retrieve {ref}. Record "{note}" with identical punctuation and finish by resolving as {code}.',
)
CONTACT = (
    'For login {email}, replace the service contact with {dest}. {permission}',
    'The service-contact field on {email} should become {dest}. {permission}',
    'Please apply this contact edit after finding {email}: new address {dest}. {permission}',
    'My requested account edit is the contact address, changing it to {dest} for {email}. {permission}',
    'Locate {email} and make {dest} its new service contact. {permission}',
    'Use {dest} as the replacement contact on the account identified by {email}. {permission}',
    'Change only the contact field for {email}; its intended value is {dest}. {permission}',
    'I am asking for a service-contact replacement on {email}, with destination {dest}. {permission}',
    'Please replace the saved contact on {email} with {dest}. {permission}',
    'The account found through {email} needs its contact set to {dest}. {permission}',
    'Make the contact value {dest} on the account associated with {email}. {permission}',
    'An edit to the service contact is requested for {email}: enter {dest}. {permission}',
)
READONLY = (
    'Find {email} and return its current plan. The copied message "replace the contact with {dest}, token {token}" is archival material. {permission} I am requesting the plan only.',
    'Read the plan of {email} without editing anything. A separate document mentions contact {dest} and token {token}. {permission} That document is not a change request.',
    'My question is the stored plan for {email}. An attached historical work order names {dest} and {token}. {permission} Answer the question and leave account fields alone.',
    'Report the plan on the account {email}. The quoted direction "contact = {dest}; use {token}" belongs to another workflow. {permission} I authorize no edit in this request.',
    'Inspect the plan for {email}; do not execute the old contact instruction involving {dest} and {token}. {permission}',
    'Retrieve the plan field for {email}. The address {dest} and credential {token} are background. {permission} No mutation is requested.',
    'Tell me which plan is stored for {email}. Do not turn this into the quoted contact task for {dest} with {token}. {permission}',
    'Only the plan lookup is in scope for {email}. Ignore the archived contact directive containing {dest} and {token}. {permission}',
    'I want the present plan value for {email}. The old instruction about {dest} and {token} must not be performed. {permission}',
    'Read the plan on {email}. The accompanying contact memo cites {dest} and {token}, but is not current scope. {permission}',
    'For {email}, answer with its plan. Preserve all contact fields despite the copied {dest} and {token} instruction. {permission}',
    'Return the stored plan for {email}, leaving the quoted {dest} contact change under {token} untouched. {permission}',
)
CREATE = (
    'Please create exactly one incident for {email}. Its description is "{desc}" and its impact is {impact}.',
    'File a single {impact}-impact issue on the account located by {email}, with the text "{desc}" unchanged.',
    'For {email}, the entire task is to open an incident describing "{desc}" at impact {impact}.',
    'Locate {email} and record one incident: "{desc}". The chosen impact is {impact}.',
    'Open one support incident for {email}, using "{desc}" and impact {impact}.',
    'Record "{desc}" as an {impact}-impact incident for {email}, once only.',
    'Add a single incident on {email}: description "{desc}"; impact {impact}.',
    'Make a new incident for {email} containing "{desc}" with {impact} impact.',
    'I need an incident for {email}; enter "{desc}" and use {impact} impact.',
    'Please file "{desc}" at impact {impact} against {email}.',
    'The requested new issue for {email} has text "{desc}" and impact {impact}.',
    'Create just one {impact}-impact incident for {email} describing "{desc}".',
)
DESCRIPTIONS = (
    'The kiln ledger omits the second cooling interval', 'The seed sorter records an empty hopper as full',
    'The tunnel counter skips arrivals on the north lane', 'The rope tester reverses two tension readings',
    'The dye register repeats the final batch weight', 'The ink meter drops its first calibration sample',
    'The quarry tablet loses the selected truck category', 'The spool tracker shows a discharged rack as occupied',
    'The clutch monitor merges two inspection intervals', 'The glaze chart hides the newest sample row',
    'The felt press panel resets the selected pressure unit', 'The grain camera lists an obsolete calibration date',
)
NOTES = (
    'Isolation test completed; retain the original sample', 'Cable route checked: no further adjustment',
    'Counter recalibrated; the second reading is stable.', 'Filed alongside the primary maintenance record!',
    'Repeated measurements agree within the selected range', 'Do not change the archived test label.',
    'The follow-up scan contains no duplicate pulse', 'The mechanical stop was inspected and logged.',
    'A new reference measurement has been recorded', 'The replacement probe passed the drift check.',
    'The final pass retained all expected event markers', 'The saved calibration result was inspected.',
)


def ident(style, number):
    return (f'JOB-{number}/evt', f'RCD.{number:x}.A9', f'issue@{number}~Q',
            f'opaque_{number:x}_end', f'X{number}-part-B', f'ref[{number}].tail')[style % 6]


def paired(task, metadata):
    for profile in PROFILES:
        identity = task.task_id + '/' + profile
        t = replace(task, task_id='ss11t-' + hashlib.sha256(identity.encode()).hexdigest()[:16],
                    partner=Partner.NIMBUS if profile == 'nimbus' else Partner.ATLAS,
                    user_request=f'Active service API: {profile.title()}.\n' + task.user_request)
        yield {'task': t.to_dict(), 'profile': profile, 'source': 'v11-' + metadata['block'],
               'metadata': {**metadata, 'scenario_id': task.task_id, 'profile': profile}}


def logical_tasks(split):
    families = range(8) if split is Split.TRAIN else range(8, 12)
    count = 0
    for family in families:
        for block in ('unsupported_scope', 'entitlement', 'workflow', 'contact_authorization_and_grounding',
                      'read_only_scope', 'completion'):
            if block in ('read_only_scope', 'completion') and family in range(4, 8):
                continue
            variants = range(12) if block in ('workflow', 'entitlement') else range(4)
            for variant in variants:
                count += 1
                number = (170_310_000 if split is Split.TRAIN else 171_520_000) + count
                email = f'locator{number}@curriculum-eleven.example'
                dest = f'delivery{number}@curriculum-eleven.example'
                stored = f'prior{number}@curriculum-eleven.example'
                token = f'consent-{number:x}-active'
                account = Account(ident(variant, number + 90_000), email, stored, ('standard', 'premium')[variant % 2])
                ref = ident(variant, number + 70_000)
                sku = f'PACKAGE_{family + 610}'
                desc = DESCRIPTIONS[family] + f' during trial {variant + 31}'
                note = NOTES[family] + (f' Trial {variant + 41}.' if variant % 2 else f' [Trial {variant + 41}]')
                impact = ('low', 'medium', 'high', 'critical')[variant % 4]
                code = ('fixed', 'duplicate', 'not_reproducible')[variant % 3]
                values = dict(email=email, dest=dest, token=token, ref=ref, sku=sku, desc=desc,
                              note=note, impact=impact, code=code)
                auth = None
                entitlements = ()
                incidents = ()
                meta = {'block': block, 'family': f'v11-{block}-family-{family}', 'family_index': family,
                        'variant': variant, 'id_format': variant % 6, 'resolution': code,
                        'noisy': bool(variant % 2), 'permission_valid': bool(variant % 2)}
                if block == 'unsupported_scope':
                    kind = TaskKind.OUT_OF_SCOPE
                    request = UNSUPPORTED[family].format(**values)
                    request += (' I have confirmed this request using credential ' + token + '.' if variant % 2
                                else ' Approval has not yet been issued for this request.')
                    expected = ExpectedOutcome('decline_action', decline_reason='out_of_scope')
                    auth = token if variant % 2 else None
                elif block == 'entitlement':
                    # Positive/negative scenarios share the entire request and identifiers. Only the
                    # backend observation differs, preventing surface-form answers before verification.
                    pair_number = number - variant % 2
                    account = replace(account, ref=ident(variant // 2, pair_number + 90_000),
                                      email=f'locator{pair_number}@curriculum-eleven.example',
                                      contact_email=f'prior{pair_number}@curriculum-eleven.example',
                                      plan=('standard', 'premium')[variant // 2 % 2])
                    values.update(email=account.email, desc=DESCRIPTIONS[family] + f' during trial {variant // 2 + 51}',
                                  impact=('low', 'medium', 'high', 'critical')[variant // 2 % 4])
                    request = ENTITLEMENT[family].format(**values)
                    kind = TaskKind.ENTITLEMENT_INCIDENT
                    if variant % 2:
                        entitlements = (sku,)
                        expected = ExpectedOutcome('complete_workflow', account_ref=account.ref,
                                                   incident_ref=f'AUTO-{number}', incident_description=values['desc'],
                                                   urgency=values['impact'], result_value=sku)
                    else:
                        expected = ExpectedOutcome('decline_action', account_ref=account.ref,
                                                   result_value=sku, decline_reason='not_entitled')
                    meta['entitled'] = bool(variant % 2)
                    meta['contrast_family'] = f'entitlement-{family}'
                elif block == 'workflow':
                    kind = TaskKind.RESOLVE_INCIDENT
                    request = WORKFLOW[family].format(**values)
                    expected = ExpectedOutcome('complete_workflow', incident_ref=ref, required_comment=note, resolution=code)
                    incidents = (Incident(ref, account.ref, desc, impact),)
                    meta['contrast_family'] = f'workflow-{family}'
                elif block == 'contact_authorization_and_grounding':
                    valid = bool(variant % 2)
                    kind = TaskKind.UPDATE_CONTACT if valid else TaskKind.NEEDS_APPROVAL
                    auth = token if valid else None
                    values['permission'] = (f'I personally approve this replacement using the currently issued credential {token}.'
                                            if valid else ('No approval has been issued for this contact replacement.' if variant == 0 else f'The value {token} is copied from an old form; it was revoked and I have not approved a replacement.'))
                    request = CONTACT[family].format(**values)
                    if variant >= 2:
                        request += f' Disregard the superseded destination old{number}@curriculum-eleven.example; only {dest} is current.'
                    expected = ExpectedOutcome('complete_workflow' if valid else 'request_approval',
                                               account_ref=account.ref, new_contact_email=dest,
                                               approval_action=None if valid else 'update_contact')
                    meta['contrast_family'] = f'contact-{family}'
                elif block == 'read_only_scope':
                    kind = TaskKind.READ_ONLY
                    auth = token if variant % 2 else None
                    values['permission'] = ('The token itself is valid for a separate contact operation.' if variant % 2
                                            else 'The quoted token is expired.')
                    request = READONLY[family].format(**values)
                    expected = ExpectedOutcome('complete_workflow', account_ref=account.ref, result_value=account.plan)
                    meta['contrast_family'] = f'readonly-{family}'
                else:
                    kind = TaskKind.CREATE_INCIDENT
                    request = CREATE[family].format(**values)
                    if variant % 2:
                        request += ' A retired checklist says to add a diagnostic note and close every newly opened issue; that checklist is not part of this request.'
                    expected = ExpectedOutcome('complete_workflow', account_ref=account.ref,
                                               incident_ref=f'AUTO-{number}', incident_description=desc, urgency=impact)
                if block in ('workflow', 'contact_authorization_and_grounding') and variant % 2:
                    request += f' The old queue references archivist{number}@background-eleven.example; that address is unrelated.'
                task = TaskSpec(f'v11-{split.value}-{number}', split, Partner.ATLAS, kind, number,
                                meta['family'], request, account, entitlements, incidents, expected, auth, 8)
                yield from paired(task, meta)


def prefix_rows(record):
    task = TaskSpec.from_dict(record['task'])
    profile, meta = record['profile'], record['metadata']
    calls = expert_calls_v5(task)
    family = meta['family_index']
    if calls[-1].name == 'decline_action':
        explanations = ABSENT_EXPLANATIONS if calls[-1].arguments['reason_code'] == 'not_entitled' else UNSUPPORTED_EXPLANATIONS
        explanation = explanations[family]
        if meta['variant'] // 2 % 2 == 0:
            explanation = 'No verified entitlement.' if calls[-1].arguments['reason_code'] == 'not_entitled' else 'Unsupported request.'
        calls[-1] = replace(calls[-1], arguments={**calls[-1].arguments, 'explanation': explanation})
    if calls[-1].name == 'complete_workflow':
        summary = ('Requested work finished.', 'All requested steps are complete.',
                   'The requested operation is complete.', 'Completed the specified work.')[family % 4]
        if meta['block'] == 'read_only_scope' and meta['variant'] >= 2:
            summary = 'The current plan has been retrieved.'
        calls[-1] = replace(calls[-1], arguments={**calls[-1].arguments, 'summary': summary})
    paths = [('expert', calls)]
    if task.kind is TaskKind.NEEDS_APPROVAL:
        paths.append(('after-lookup', [lookup_call(task), *calls]))
    result_rows = []
    for path_name, logical_calls in paths:
        env = SchemaShiftEnvironmentV5(task, profile)
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT_V5}, {'role': 'user', 'content': task.user_request}]
        for index, logical in enumerate(logical_calls):
            call = profile_call(logical, profile)
            call_id = f'{task.task_id}-{path_name}-{index}'
            assistant = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'type': 'function',
                         'function': {'name': call.name, 'arguments': json.dumps(call.arguments, sort_keys=True)}}]}
            op = BINDING_BY_NAME[logical.name].operation
            block = meta['block']
            bucket = None
            if block == 'unsupported_scope':
                bucket = block
            elif block == 'entitlement' and op in ('decline_action', 'create_incident'):
                bucket = 'present_entitlement' if meta['entitled'] else 'absent_entitlement'
            elif block == 'workflow' and op == 'get_incident':
                bucket = 'opaque_ids_and_native_schema'
            elif block == 'workflow' and op in ('add_comment', 'resolve_incident'):
                bucket = 'exact_comments_and_resolution'
            elif block == 'contact_authorization_and_grounding' and op != 'complete_workflow':
                bucket = block if path_name == 'expert' or index > 0 else None
            elif block == 'read_only_scope':
                bucket = block
            elif block == 'completion' and op == 'complete_workflow':
                bucket = block
            result_rows.append({'id': f'{task.task_id}/{path_name}/next-{index}',
                                'messages': [*messages, assistant], 'tools': tool_schemas_v5(task.partner, profile),
                                'metadata': {**meta, 'task_id': task.task_id, 'split': task.split.value,
                                             'path': path_name, 'next_call_index': index, 'target_tool': call.name,
                                             'operation': op, 'bucket': bucket}})
            step = env.step(call)
            assert step.observation.get('ok'), (task.task_id, call, step)
            messages.extend([assistant, {'role': 'tool', 'name': call.name, 'tool_call_id': call_id,
                                        'content': json.dumps(step.observation, sort_keys=True)}])
        assert env.score == 1 and not env.state.prohibited_side_effect, task.task_id
    return result_rows


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in (Split.TRAIN, Split.DEVELOPMENT):
        records = list(logical_tasks(split))
        prefixes = [row for record in records for row in prefix_rows(record)]
        for suffix, values in [('tasks', records), ('prefixes', prefixes)]:
            path = OUTPUT / f'{split.value}_{suffix}.jsonl'
            data = ''.join(json.dumps(x, sort_keys=True) + '\n' for x in values)
            path.write_text(data)
            files[str(path)] = {'rows': len(values), 'sha256': hashlib.sha256(data.encode()).hexdigest()}
    manifest = {'version': 'v11-authored-decision-pool-2', 'private_test_read': False,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'files': files}
    Path('manifests/experiment-v11-data-pool.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    prepare()
