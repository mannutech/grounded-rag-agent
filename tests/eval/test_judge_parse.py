"""Judge: robust JSON parsing + N-vote majority / abstain behaviour."""

from __future__ import annotations

import json

import pytest

from grounded_rag.core.errors import JudgeParseError
from grounded_rag.core.providers import MockChatProvider
from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    GoldRecord,
    QueryType,
    TokenUsage,
)
from grounded_rag.eval.judge import judge, parse_judge_response

_BALLOT = {
    "correct": True,
    "grounded": True,
    "refusal_appropriate": None,
    "keypoints_hit": ["Y"],
    "score": 0.9,
    "rationale": "matches the reference",
}


def _ballot_json(correct: bool = True) -> str:
    return json.dumps({**_BALLOT, "correct": correct})


# -- parsing ----------------------------------------------------------------


def test_parse_clean_json() -> None:
    ballot = parse_judge_response(_ballot_json())
    assert ballot.correct is True
    assert ballot.score == 0.9


def test_parse_code_fenced() -> None:
    raw = f"```json\n{_ballot_json()}\n```"
    assert parse_judge_response(raw).grounded is True


def test_parse_with_surrounding_prose() -> None:
    raw = f"Sure! Here is my verdict:\n{_ballot_json()}\nHope that helps."
    assert parse_judge_response(raw).correct is True


def test_parse_nested_braces_in_rationale() -> None:
    raw = json.dumps({**_BALLOT, "rationale": "the answer used {x: 1} notation, which is fine"})
    assert parse_judge_response(raw).rationale.startswith("the answer used {x: 1}")


def test_parse_trailing_comma_is_repaired() -> None:
    raw = (
        '{"correct": true, "grounded": true, "refusal_appropriate": null, '
        '"keypoints_hit": ["Y"], "score": 0.5, "rationale": "ok",}'
    )
    assert parse_judge_response(raw).score == 0.5


def test_parse_irrecoverable_junk_raises() -> None:
    with pytest.raises(JudgeParseError):
        parse_judge_response("there is no json here at all")


def test_parse_missing_required_field_raises() -> None:
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"correct": true}')


# -- majority / abstain -----------------------------------------------------


def _gold() -> GoldRecord:
    return GoldRecord(
        id="n1", question="q", type=QueryType.NORMAL, expected_answer="Y", keypoints=["Y"]
    )


def _result() -> AgentResult:
    trace = AgentRunTrace(
        query="q",
        model="m",
        steps=1,
        timings=[],
        tool_calls=[],
        injection_flagged=False,
        seed=None,
        temperature=0.0,
        total_duration_ms=0.0,
    )
    return AgentResult(
        answer="Y is the answer",
        refused=False,
        citations=[],
        retrieved=[],
        tool_calls=[],
        usage=TokenUsage(),
        cost_usd=0.0,
        steps=1,
        trace=trace,
    )


def test_majority_two_of_three() -> None:
    provider = MockChatProvider(texts=[_ballot_json(True), _ballot_json(True), _ballot_json(False)])
    verdict = judge(_gold(), _result(), provider=provider, n_votes=3)
    assert verdict.correct is True  # 2 of 3
    assert verdict.n_votes == 3
    assert verdict.agreement == pytest.approx(2 / 3)


def test_all_votes_abstain_on_unparseable_output() -> None:
    # one vote, original + retry both junk -> abstain -> empty verdict
    provider = MockChatProvider(texts=["garbage", "more garbage"])
    verdict = judge(_gold(), _result(), provider=provider, n_votes=1)
    assert verdict.n_votes == 0
    assert verdict.correct is False
    assert verdict.ballots == []


def test_retry_recovers_after_one_bad_response() -> None:
    # first response unparseable, retry returns a valid ballot
    provider = MockChatProvider(texts=["oops not json", _ballot_json(True)])
    verdict = judge(_gold(), _result(), provider=provider, n_votes=1)
    assert verdict.n_votes == 1
    assert verdict.correct is True
