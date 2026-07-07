"""LLM-as-judge with an explicit rubric and reduced variance.

Variance is controlled three ways: a fixed low ``temperature``, a base ``seed``
(varied per vote so the votes actually sample the judge rather than repeat one
answer), and an N-vote **majority** whose winning margin is reported as
``Verdict.agreement`` so low-confidence rows are visible. Judge output is parsed
defensively — code fences, surrounding prose, nested braces, and trailing commas
are all tolerated; an unrecoverable response triggers one stricter retry and then
an abstention rather than crashing the run.
"""

from __future__ import annotations

import json
import re
from typing import Any

from grounded_rag.core.errors import JudgeParseError
from grounded_rag.core.providers import ChatProvider
from grounded_rag.core.types import AgentResult, GoldRecord, JudgeBallot, QueryType, Verdict

JUDGE_RUBRIC = """You are a strict evaluator of a retrieval-augmented answer. Apply these criteria:

1. CORRECT: Does the answer match the reference answer / cover the listed keypoints?
   A refusal is NOT correct for a normal question. For a must_refuse question, the
   answer is correct ONLY if the assistant refused (declined to answer).
2. GROUNDED: Is every factual claim in the answer supported by the RETRIEVED CONTEXT
   shown below? An answer that asserts facts absent from the context is NOT grounded,
   even if those facts happen to be true. A refusal is trivially grounded.
3. REFUSAL_APPROPRIATE: Only for must_refuse questions — was refusing the right call?
   Use null for normal questions.
4. KEYPOINTS_HIT: Which of the listed keypoints does the answer actually contain?
5. SCORE: An overall quality score in [0, 1].

Respond with ONLY a JSON object, no prose and no code fences:
{"correct": bool, "grounded": bool, "refusal_appropriate": bool or null,
 "keypoints_hit": [string], "score": number, "rationale": string}"""


def build_judge_prompt(record: GoldRecord, result: AgentResult) -> list[dict[str, Any]]:
    """Construct the judge chat messages for one gold record + agent result."""
    context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in result.retrieved) or "(no context)"
    cited = ", ".join(cid for c in result.citations for cid in c.chunk_ids) or "(none)"
    is_refuse = record.type is QueryType.MUST_REFUSE or record.must_refuse
    user = f"""QUESTION:
{record.question}

QUESTION TYPE: {record.type.value} (must_refuse={is_refuse})

REFERENCE ANSWER:
{record.expected_answer or "(none — this question should be refused)"}

KEYPOINTS THE ANSWER SHOULD CONTAIN:
{json.dumps(record.keypoints)}

RETRIEVED CONTEXT THE ASSISTANT WAS GIVEN:
{context}

ASSISTANT ANSWER (refused={result.refused}):
{result.answer}

ASSISTANT CITED CHUNKS: {cited}

Evaluate per the rubric and return the JSON object."""
    return [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": user},
    ]


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _loads_with_repair(block: str) -> Any:
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", block)  # drop trailing commas
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise JudgeParseError(f"invalid JSON: {exc}") from exc


def parse_judge_response(raw: str) -> JudgeBallot:
    """Parse a (possibly messy) judge response into a validated :class:`JudgeBallot`."""
    block = _extract_json_object(raw)
    if block is None:
        raise JudgeParseError(f"no JSON object found in judge output: {raw[:160]!r}")
    data = _loads_with_repair(block)
    try:
        return JudgeBallot.model_validate(data)
    except Exception as exc:
        raise JudgeParseError(f"judge output failed validation: {exc}") from exc


def _majority(values: list[bool]) -> tuple[bool, float]:
    """Majority vote (ties -> True) plus the winning-margin fraction."""
    n = len(values)
    trues = sum(1 for v in values if v)
    falses = n - trues
    return trues >= falses, max(trues, falses) / n


def _keypoint_recall(record: GoldRecord, ballot: JudgeBallot) -> float:
    if not record.keypoints:
        return 0.0
    hit = set(ballot.keypoints_hit) & set(record.keypoints)
    return len(hit) / len(record.keypoints)


def _one_vote(
    provider: ChatProvider,
    messages: list[dict[str, Any]],
    temperature: float,
    seed: int | None,
    response_format: dict[str, Any] | None,
) -> JudgeBallot | None:
    """A single judge call; one stricter retry on parse failure, else abstain."""
    try:
        completion = provider.complete(
            messages=messages, temperature=temperature, seed=seed, response_format=response_format
        )
        return parse_judge_response(completion.text)
    except JudgeParseError:
        stricter = [*messages, {"role": "user", "content": "Return ONLY the JSON object."}]
        try:
            completion = provider.complete(
                messages=stricter,
                temperature=temperature,
                seed=seed,
                response_format=response_format,
            )
            return parse_judge_response(completion.text)
        except JudgeParseError:
            return None


def judge(
    record: GoldRecord,
    result: AgentResult,
    *,
    provider: ChatProvider,
    n_votes: int = 3,
    temperature: float = 0.0,
    seed: int | None = 7,
    response_format_json: bool = True,
) -> Verdict:
    """Evaluate ``result`` against ``record`` via an N-vote majority of judge ballots.

    ``provider`` is any :class:`ChatProvider` — using a *different* model family than
    the system under test (e.g. GPT-4o or Claude judging a Cohere agent) is how the
    harness guards against same-family self-preference bias.
    """
    messages = build_judge_prompt(record, result)
    response_format = {"type": "json_object"} if response_format_json else None
    ballots: list[JudgeBallot] = []
    for i in range(n_votes):
        vote_seed = seed + i if seed is not None else None
        ballot = _one_vote(provider, messages, temperature, vote_seed, response_format)
        if ballot is not None:
            ballots.append(ballot)
    return _aggregate(record, ballots)


def _aggregate(record: GoldRecord, ballots: list[JudgeBallot]) -> Verdict:
    if not ballots:  # every vote abstained
        return Verdict(
            correct=False,
            grounded=False,
            refusal_appropriate=None,
            score=0.0,
            keypoint_recall=0.0,
            n_votes=0,
            agreement=0.0,
            ballots=[],
        )
    correct, agreement = _majority([b.correct for b in ballots])
    grounded, _ = _majority([b.grounded for b in ballots])
    refusal_values = [b.refusal_appropriate for b in ballots if b.refusal_appropriate is not None]
    refusal_appropriate = _majority(refusal_values)[0] if refusal_values else None
    return Verdict(
        correct=correct,
        grounded=grounded,
        refusal_appropriate=refusal_appropriate,
        score=sum(b.score for b in ballots) / len(ballots),
        keypoint_recall=sum(_keypoint_recall(record, b) for b in ballots) / len(ballots),
        n_votes=len(ballots),
        agreement=agreement,
        ballots=ballots,
    )
