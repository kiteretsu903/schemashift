"""Evidence integrity, replay tamper detection, and severity separation."""

import json
from pathlib import Path

import pytest

from schemashift.classification import audit_suite, classify_episode
from schemashift.independent_v5 import read_cases
from schemashift.replay import replay_comparison, verify_files


def load_rows(model="v16", suite="fresh"):
    return [json.loads(line) for line in
            Path(f"evaluation/recorded_runs/{model}/{suite}.jsonl").read_text().splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_all_576_native_records_reproduce_published_comparison():
    assert verify_files()["passed"]
    result = replay_comparison()
    assert result["episodes_replayed"] == 576
    assert result["matches_published_results"]
    pair = result["suites"]["fresh"]["pairs"]["v16_minus_rft"]
    assert len(pair["right_only_successes"]) == 9
    assert pair["left_only_successes"] == []


@pytest.mark.parametrize("tamper", ("success", "observation", "final_state", "retry"))
def test_replay_rejects_changed_native_evidence(tmp_path, tamper):
    rows = load_rows()
    if tamper == "success":
        row = next(row for row in rows if not row["success"])
        row["success"] = True
    elif tamper == "observation":
        rows[0]["calls"][0]["observation"]["injected_change"] = True
    elif tamper == "final_state":
        rows[0]["final_state"]["steps"] += 1
    else:
        rows[0]["model_turns"][0]["attempts"] = 2
    path = tmp_path / "changed.jsonl"
    write_rows(path, rows)
    with pytest.raises((AssertionError, ValueError)):
        audit_suite("evaluation/protocols/fresh.json", path, "v16")


def test_real_punctuation_failures_remain_distinct_from_wrong_target_writes():
    tasks, profiles, _metadata = read_cases(Path("evaluation/fixtures/fresh.json"))
    indexed = {task.task_id: task for task in tasks}
    minor = []
    for row in load_rows():
        classified = classify_episode(indexed[row["task_id"]], profiles[row["task_id"]], row)
        if "punctuation_or_whitespace_only" in classified["categories"]:
            assert not row["success"]
            assert classified["categories"] == ["punctuation_or_whitespace_only"]
            minor.append(row)
    assert len(minor) == 2
    wrong = next(row for row in load_rows("rft") if row["failure_reason"] == "mutation_not_grounded")
    classified = classify_episode(indexed[wrong["task_id"]], profiles[wrong["task_id"]], wrong)
    assert "wrong_mutation_target" in classified["categories"]
    assert "punctuation_or_whitespace_only" not in classified["categories"]
