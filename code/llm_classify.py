#!/usr/bin/env python3
"""Have Claude Fable 5 (via OpenRouter) classify every item's Bloom level and
difficulty under four conditions, so the labels can be compared with the two
human raters, the generating model's self-labels, and student performance.

Conditions
  bloom_instructed   : system prompt = the exact instructions the human raters saw
  bloom_bare         : only the six level names, no definitions
  diff_instructed    : easy/medium/hard defined by expected proportion correct
  diff_bare          : "rate the difficulty" with no definitions

Items are shown as the raters saw them: stem + four options, key marked.
Results are cached in data/llm_labels.json (one record per condition x item);
re-running only fills gaps.  OPENROUTER_API_KEY is read from .env.

    uv run python llm_classify.py [--limit N] [--workers 8] [--model anthropic/claude-fable-5]

--structured switches to OpenRouter structured outputs (response_format =
json_schema, strict) and leaves the temperature at the provider default;
results go to data/llm_labels_structured_<model>.json.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
from dotenv import load_dotenv
from tqdm import tqdm
import os

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
API_KEY = os.environ["OPENROUTER_API_KEY"]
CACHE_DEFAULT = HERE / "data" / "llm_labels.json"
LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
DIFFS = ["easy", "medium", "hard"]


# ------------------------------------------------------------------ prompts
def rater_instructions() -> str:
    """The instruction text shown to human raters, extracted from the app's HTML."""
    src = (HERE.parent / "annotator" / "static" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<aside id="guide".*?</aside>', src, flags=re.S)
    t = m.group(0)
    t = re.sub(r"<button.*?</button>", "", t, flags=re.S)
    t = re.sub(r"<p class=\"muted small\">You can annotate in any order.*?</p>", "", t, flags=re.S)  # keyboard help
    t = re.sub(r"<(h3|dt)[^>]*>", "\n\n", t)
    t = re.sub(r"<(li|dd|p)[^>]*>", "\n", t)
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t).strip()
    return t


INSTR = rater_instructions()

SYSTEM = {
    "bloom_instructed": (
        "You are one of the annotators in the study described below. Follow the instructions exactly as a human "
        "annotator would.\n\n" + INSTR +
        "\n\nRespond with a JSON object only: {\"primary_level\": <one of remember, understand, apply, analyze, "
        "evaluate, create>, \"rationale\": <one sentence>}."
    ),
    "bloom_bare": (
        "Assign the level of Bloom's taxonomy that this multiple-choice exam question assesses. The levels are: "
        "remember, understand, apply, analyze, evaluate, create.\n\n"
        "Respond with a JSON object only: {\"primary_level\": <one of the six levels>, \"rationale\": <one sentence>}."
    ),
    "diff_instructed": (
        "You are estimating how difficult a multiple-choice exam question will be for students who have completed a "
        "master's-level course on applied machine learning (deployment and operations), sitting a closed-book exam "
        "generated from the course textbook and lecture slides.\n\n"
        "Rate the item as one of:\n"
        "- easy: you expect more than 85% of these students to answer correctly\n"
        "- medium: you expect 60% to 85% to answer correctly\n"
        "- hard: you expect fewer than 60% to answer correctly\n\n"
        "Consider how specific the required knowledge is, whether the wrong options are plausible enough to tempt a "
        "student who knows the material, whether the key can be found by elimination or by wording alone, and how "
        "much reasoning the stem demands. Judge the item as a whole, not the topic.\n\n"
        "Respond with a JSON object only: {\"difficulty\": <easy|medium|hard>, \"rationale\": <one sentence>}."
    ),
    "diff_bare": (
        "Rate the difficulty of this multiple-choice exam question for students in a master's-level applied machine "
        "learning course as easy, medium or hard.\n\n"
        "Respond with a JSON object only: {\"difficulty\": <easy|medium|hard>, \"rationale\": <one sentence>}."
    ),
}


def item_text(q: dict) -> str:
    opts = [(q["correct_answer"], True)] + [(o["text"], False) for o in q["incorrect_answers"]]
    lines = [f"Question: {q['question_text']}", ""]
    for letter, (text, correct) in zip("ABCD", opts):
        lines.append(f"{letter}. {text}" + ("   (correct answer)" if correct else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------ API
def schema_for(cond: str) -> dict:
    field, values = ("primary_level", LEVELS) if cond.startswith("bloom") else ("difficulty", DIFFS)
    return {"type": "json_schema", "json_schema": {"name": f"{cond}_label", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": [field, "rationale"],
        "properties": {field: {"type": "string", "enum": values}, "rationale": {"type": "string"}}}}}


def call(model: str, system: str, user: str, retries: int = 6, structured: str | None = None) -> dict:
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if structured:  # condition name -> JSON schema; no temperature is set
        payload["max_tokens"] = 4000
        payload["response_format"] = schema_for(structured)
        payload["provider"] = {"require_parameters": True}
    else:
        payload["temperature"] = 0
        payload["max_tokens"] = 500
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/bloom-study", "X-Title": "bloom-level-analysis"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if content is None:  # e.g. finish_reason "content_filter" (provider refusal): not retried
                return {"raw": None, "parsed": {}, "error": f"no content: finish_reason={choice.get('finish_reason')} "
                        f"native={choice.get('native_finish_reason')}", "model": data.get("model"), "usage": data.get("usage")}
            parsed = {}
            m = re.search(r"\{.*\}", content, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    parsed = {}
            if not parsed:  # fall back to key: "value" patterns anywhere in the text
                for key in ("primary_level", "difficulty", "rationale"):
                    km = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, content)
                    if km:
                        parsed[key] = km.group(1)
            return {"raw": content, "parsed": parsed, "usage": data.get("usage"), "model": data.get("model")}
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            if attempt == retries - 1:
                return {"raw": None, "parsed": {}, "error": str(e)}
            time.sleep(wait)


def normalise(cond: str, parsed: dict) -> str | None:
    if cond.startswith("bloom"):
        v = str(parsed.get("primary_level", "")).strip().lower()
        return v if v in LEVELS else None
    v = str(parsed.get("difficulty", "")).strip().lower()
    return v if v in DIFFS else None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-fable-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    ap.add_argument("--conditions", default="bloom_instructed,bloom_bare,diff_instructed,diff_bare")
    ap.add_argument("--structured", action="store_true", help="use structured outputs (json_schema) and provider-default temperature")
    args = ap.parse_args()
    # one cache per model; the default model keeps the original file name
    short = args.model.split("/")[-1]
    if args.structured:
        CACHE = HERE / "data" / f"llm_labels_structured_{short}.json"
    else:
        CACHE = CACHE_DEFAULT if args.model == "anthropic/claude-fable-5" else HERE / "data" / f"llm_labels_{short}.json"

    data = json.loads((HERE / "data" / "export.json").read_text())
    items = data["questions"][: args.limit] if args.limit else data["questions"]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    lock = threading.Lock()

    todo = []
    for cond in args.conditions.split(","):
        for q in items:
            key = f"{cond}|{q['exam']}|{q['question_no']}"
            if key in cache and cache[key].get("label"):
                continue
            todo.append((key, cond, q))
    print(f"{len(cache)} cached, {len(todo)} calls to make with {args.model}")

    def work(job):
        key, cond, q = job
        r = call(args.model, SYSTEM[cond], item_text(q), structured=cond if args.structured else None)
        rec = {"condition": cond, "exam": q["exam"], "question_no": q["question_no"],
               "label": normalise(cond, r["parsed"]), "rationale": r["parsed"].get("rationale"),
               "raw": r["raw"], "error": r.get("error"), "model": r.get("model"), "usage": r.get("usage")}
        with lock:
            cache[key] = rec
        return key

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for i, _ in enumerate(tqdm(as_completed(futs), total=len(futs), desc="classifying", unit="item")):
            if i % 25 == 0:
                with lock:
                    CACHE.write_text(json.dumps(cache, indent=1))
    CACHE.write_text(json.dumps(cache, indent=1))

    missing = [k for k, v in cache.items() if not v.get("label")]
    print(f"done: {len(cache)} records, {len(missing)} without a valid label")
    if missing:
        print("  e.g.", missing[:5])
    tok = sum((v.get("usage") or {}).get("total_tokens", 0) for v in cache.values())
    print(f"total tokens used: {tok}")


if __name__ == "__main__":
    main()
