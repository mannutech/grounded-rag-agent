"""Build a large, real evaluation dataset from SQuAD 2.0.

Turns SQuAD 2.0 into the harness's format:

* corpus  -> data/squad/docs.jsonl   (one {"doc_id","title","text"} per line;
             each SQuAD paragraph becomes a retrievable document)
* gold    -> data/squad/gold.jsonl   (answerable questions -> normal cases with a
             reference answer + the source paragraph as the relevant doc for
             recall@k; SQuAD's *unanswerable* questions -> genuine must_refuse cases)

Human-authored (fixes the "author wrote the questions" circularity), and large
enough that retrieval recall no longer saturates. Reproducible via a fixed seed.

Usage:  python scripts/build_squad.py [path/to/dev-v2.0.json]
(downloads the SQuAD dev set if no path is given and it isn't cached).
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from pathlib import Path

_SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
_OUT_DIR = Path("data/squad")
_N_DOCS = 150
_N_REFUSE = 30
_SEED = 13


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def _load_squad(arg_path: str | None) -> dict:
    if arg_path:
        return json.loads(Path(arg_path).read_text(encoding="utf-8"))
    cache = _OUT_DIR / "_dev-v2.0.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading SQuAD 2.0 dev set -> {cache}")
        urllib.request.urlretrieve(_SQUAD_URL, cache)  # noqa: S310 - trusted URL
    return json.loads(cache.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    data = _load_squad(argv[1] if len(argv) > 1 else None)

    # Flatten to (title, paragraph) and sample a corpus deterministically.
    paragraphs = [(a["title"], p) for a in data["data"] for p in a["paragraphs"]]
    rng = random.Random(_SEED)
    sample = rng.sample(paragraphs, min(_N_DOCS, len(paragraphs)))

    docs: list[dict] = []
    normal: list[dict] = []
    refuse_pool: list[tuple[str, dict]] = []
    title_counter: dict[str, int] = {}

    for title, para in sample:
        n = title_counter.get(title, 0)
        title_counter[title] = n + 1
        doc_id = f"{_slug(title)}-{n}"
        docs.append({"doc_id": doc_id, "title": title, "text": para["context"]})

        answerable = [q for q in para["qas"] if not q.get("is_impossible") and q.get("answers")]
        impossible = [q for q in para["qas"] if q.get("is_impossible")]
        if answerable:
            q = answerable[0]
            answer = q["answers"][0]["text"]
            normal.append(
                {
                    "id": f"sq-{q['id']}",
                    "question": q["question"],
                    "type": "normal",
                    "expected_answer": answer,
                    "keypoints": [answer],
                    "relevant_doc_ids": [doc_id],
                }
            )
        refuse_pool.extend((doc_id, q) for q in impossible)

    # Unanswerable questions whose topic IS in the corpus -> honest must_refuse cases.
    refuse_sample = rng.sample(refuse_pool, min(_N_REFUSE, len(refuse_pool)))
    refuse = [
        {"id": f"sq-{q['id']}", "question": q["question"], "type": "must_refuse", "must_refuse": True}
        for _, q in refuse_sample
    ]

    gold = normal + refuse
    rng.shuffle(gold)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "docs.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8"
    )
    (_OUT_DIR / "gold.jsonl").write_text(
        "".join(json.dumps(g) + "\n" for g in gold), encoding="utf-8"
    )
    print(
        f"wrote {len(docs)} docs -> {_OUT_DIR / 'docs.jsonl'}\n"
        f"wrote {len(gold)} gold cases ({len(normal)} normal, {len(refuse)} must_refuse) "
        f"-> {_OUT_DIR / 'gold.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
