#!/usr/bin/env python3
"""Check of the rater-comment coding (Table 5) by a second author.

The review sheet (made by make_materials.py) shows each comment, its item and the first coding,
and the reviewer marks Agree or Disagree for every theme. This script reads the returned sheet
and reports, theme by theme: how many codes the reviewer confirmed, the reviewer's implied coding
(the first code where they agree, the opposite where they disagree), and Cohen's κ between the
first and the implied coding. It also re-runs the Section 4.5 contrast with the reviewer's
distractor flags for rater B's comments.

Note: a confirm-or-reject review is not blind to the first coding, so its agreement is an upper
bound on what an independent second coding would give.

  uv run --with openpyxl python coding_check.py ../feedback/coding_check/<returned sheet>.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from comment_codes import CODES

HERE = Path(__file__).resolve().parent
THEMES = ["distractors", "recall_dep", "common_sense", "flaw", "lower_bound", "giveaway", "unfamiliar"]

sheet = pd.read_excel(sys.argv[1], sheet_name="Comments")
cols = list(sheet.columns)
key = pd.read_csv(HERE / "data" / "coding_check_key.csv")
d = key.merge(sheet, on="id", validate="one_to_one")

# Decision columns are found by position (the column right after each "First coding" column), because a
# reviewer may overwrite a header cell. The first-coding headers are checked against the theme order.
first_cols = [c for c in cols if "First coding" in str(c)]
assert [str(c).split(".")[0] for c in first_cols] == [f"T{k}" for k in range(1, len(THEMES) + 1)], first_cols
rows, missing = [], 0
implied_distractors = {}
for k, t in enumerate(THEMES, 1):
    decide_col = cols[cols.index(first_cols[k - 1]) + 1]
    first = [int(t in CODES[(e, n, r)]) for e, n, r in zip(d.exam_orig, d["no"], d.rater)]
    verdict = d[decide_col].astype(str).str.strip().str.lower()
    blank = ~verdict.isin(["agree", "disagree"])
    missing += int(blank.sum())
    implied = [f if v != "disagree" else 1 - f for f, v in zip(first, verdict)]
    if t == "distractors":
        implied_distractors = {(e, int(n)): v for e, n, r, v in zip(d.exam_orig, d["no"], d.rater, implied) if r == "B"}
    rows.append({"theme": t, "first coding": sum(first), "reviewer": sum(implied),
                 "agree": int((verdict == "agree").sum()), "disagree": int((verdict == "disagree").sum()),
                 "blank on Yes": int((blank & (pd.Series(first, index=d.index) == 1)).sum()),
                 "blank on No": int((blank & (pd.Series(first, index=d.index) == 0)).sum()),
                 "kappa": cohen_kappa_score(first, implied) if len(set(first) | set(implied)) > 1 else float("nan")})
res = pd.DataFrame(rows).set_index("theme")
print(res.round(2).to_markdown())
tot_a, tot_d = res.agree.sum(), res.disagree.sum()
print(f"\nOverall: {tot_a} agree, {tot_d} disagree ({tot_a / max(tot_a + tot_d, 1):.1%} agreement)"
      + (f"; {missing} decisions missing (counted as agree)" if missing else ""))

# Section 4.5 with the reviewer's distractor flags
m = pd.read_csv(HERE / "out" / "items_merged.csv")
m["flag2"] = [implied_distractors.get((e, int(n)), 0) == 1 for e, n in zip(m.exam_orig, m["no"])]
f1, f0 = m[m.flag2], m[~m.flag2]
print(f"\nReviewer's distractor flags on rater B's comments: {len(f1)} items. Mean facility {f1.p.mean():.2f} "
      f"vs {f0.p.mean():.2f}; within-exam rank {f1.p_rank_in_exam.mean():.2f} vs {f0.p_rank_in_exam.mean():.2f} "
      f"(Mann-Whitney p = {stats.mannwhitneyu(f1.p_rank_in_exam, f0.p_rank_in_exam).pvalue:.3f})")
