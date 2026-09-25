#!/usr/bin/env python3
"""All quantitative results for the CELDA paper, regenerated from
data/export.json (annotation export) and ../data/result_export_*.csv (scores).

Outputs go to out/: tables as CSV + a single results.md, figures as PDF + PNG.
Run:  uv run python analyze.py
"""
from __future__ import annotations

import csv
import glob
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import krippendorff
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260828)
LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
LIDX = {l: i for i, l in enumerate(LEVELS)}
SHORT = ["Rem", "Und", "App", "Ana", "Eva", "Cre"]
DIFF = ["easy", "medium", "hard"]
RENAME = {"Exam 2": "Exam 1", "Midterm 3": "Exam 2", "Final Exam": "Exam 3"}  # paper names
EXAMS = ["Exam 1", "Exam 2", "Exam 3"]
# Annotator login -> anonymised rater letter (both cover all 188 items since the 2026-09-01 export).
# Kept in a private file so that the logins are not in the published code.
ANON = json.loads((HERE / "data" / "raters.json").read_text())

md: list[str] = []


def say(s: str = "") -> None:
    md.append(s)
    print(s)


# ----------------------------------------------------------------- load
data = json.loads((HERE / "data" / "export.json").read_text())
rows = []
for q in data["questions"]:
    r = {"exam": RENAME[q["exam"]], "no": q["question_no"], "exam_orig": q["exam"], "llm_bloom": q["llm_bloom_level"],
         "llm_diff": q["llm_difficulty"], "p": q["p_correct"], "n": q["n_answered"],
         "stem": q["question_text"], "correct": q["correct_answer"],
         "stem_words": len(q["question_text"].split()),
         "correct_words": len(q["correct_answer"].split()),
         "incorrect_words": np.mean([len(o["text"].split()) for o in q["incorrect_answers"]])}
    for u, a in ANON.items():
        ann = q["annotations"].get(u)
        r[f"{a}_bloom"] = ann["primary_level"] if ann else None
        r[f"{a}_sec"] = ann["secondary_level"] if ann else None
        r[f"{a}_comment"] = ann["comment"] if ann else None
    rows.append(r)
df = pd.DataFrame(rows)
for c in ["llm_bloom", "A_bloom", "B_bloom"]:
    df[c + "_i"] = df[c].map(LIDX)
df["llm_diff_i"] = df["llm_diff"].map({d: i for i, d in enumerate(DIFF)})
df["p_rank_in_exam"] = df.groupby("exam").p.rank(pct=True)

# ------------------------------------------------ item statistics from scores
exam_for_file = {e["result_file"]: e["name"] for e in data["exams"]}
item_stats = {}
exam_stats = {}
for f in sorted(glob.glob(str(HERE.parent / "data" / "result_export_*.csv"))):
    exam = RENAME[exam_for_file[Path(f).name]]
    with open(f, encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
    M = defaultdict(dict)
    for r in recs:
        M[r["UserId"]][int(r["QuestionNo"])] = float(r["AutoGradedScore"])
    S = list(M)
    Q = sorted({q for s in M.values() for q in s})
    X = np.array([[M[s][q] for q in Q] for s in S])  # students x items
    tot = X.sum(1)
    p = X.mean(0)
    var_t = tot.var()
    kr20 = len(Q) / (len(Q) - 1) * (1 - (p * (1 - p)).sum() / var_t)
    for j, q in enumerate(Q):
        rest = tot - X[:, j]
        r_pb = np.nan if X[:, j].std() == 0 else np.corrcoef(X[:, j], rest)[0, 1]
        item_stats[(exam, q)] = r_pb
    exam_stats[exam] = {"students": len(S), "items": len(Q), "total_mean": tot.mean(), "total_sd": tot.std(ddof=1),
                        "pct_mean": tot.mean() / len(Q) * 100, "kr20": kr20,
                        "p_mean": p.mean(), "p_sd": p.std(ddof=1), "p_min": p.min(), "p_max": p.max(),
                        "n_p1": int((p == 1).sum()), "n_p_ge_90": int((p >= 0.9).sum())}
df["disc"] = [item_stats[(e, n)] for e, n in zip(df["exam"], df["no"])]

# ================================================================ 1. descriptives
say("# Results (auto-generated)\n")
say("## Table 1. Exams and item statistics\n")
t1 = pd.DataFrame(exam_stats).T
t1["disc_mean"] = df.groupby("exam")["disc"].mean()
t1["disc_undef"] = df.groupby("exam")["disc"].apply(lambda s: s.isna().sum())
t1["disc_neg"] = df.groupby("exam")["disc"].apply(lambda s: (s < 0).sum())
t1 = t1.loc[EXAMS]
t1.to_csv(OUT / "table1_exams.csv")
say(t1.round(2).to_markdown())
say(f"\nAll: {len(df)} items, p mean {df.p.mean():.2f} (SD {df.p.std():.2f}), "
    f"{(df.p == 1).sum()} items with p = 1, {(df.p >= .9).sum()} with p ≥ .9; "
    f"discrimination mean {df.disc.mean():.2f} (n defined {df.disc.notna().sum()}).")

say("\n## Table 2. Label distributions (primary level)\n")
dist = pd.DataFrame({
    "LLM (n=188)": df.llm_bloom.value_counts().reindex(LEVELS, fill_value=0),
    "A (n=188)": df.A_bloom.value_counts().reindex(LEVELS, fill_value=0),
    f"B (n={df.B_bloom.notna().sum()})": df.B_bloom.value_counts().reindex(LEVELS, fill_value=0),
})
dist.to_csv(OUT / "table2_distributions.csv")
say(dist.to_markdown())
for c in ["llm", "A", "B"]:
    v = df[f"{c}_bloom_i"].dropna()
    say(f"  mean level {c}: {v.mean() + 1:.2f} (1=remember … 6=create), median {np.median(v) + 1:.0f}")
say("\nLLM difficulty: " + ", ".join(f"{k} {v}" for k, v in df.llm_diff.value_counts().reindex(DIFF).items()))
say("LLM difficulty by exam:\n" + pd.crosstab(df.exam, df.llm_diff).reindex(EXAMS)[DIFF].to_markdown())
say("LLM bloom by exam:\n" + pd.crosstab(df.exam, df.llm_bloom).reindex(EXAMS).reindex(columns=LEVELS, fill_value=0).to_markdown())
say("\nLLM difficulty × LLM Bloom:\n" + pd.crosstab(df.llm_diff, df.llm_bloom).reindex(DIFF).reindex(columns=LEVELS, fill_value=0).to_markdown())
say(f"Spearman(LLM difficulty, LLM Bloom) = {stats.spearmanr(df.llm_diff_i, df.llm_bloom_i).statistic:.2f}")


# ================================================================ 2. agreement
def kappa_ci(a, b, weights=None, B=2000):
    a, b = np.asarray(a), np.asarray(b)
    k = cohen_kappa_score(a, b, labels=list(range(6)), weights=weights)
    boots = []
    n = len(a)
    for _ in range(B):
        i = RNG.integers(0, n, n)
        try:
            boots.append(cohen_kappa_score(a[i], b[i], labels=list(range(6)), weights=weights))
        except Exception:
            pass
    return k, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)


def agreement_block(name, a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    exact = (a == b).mean()
    within1 = (np.abs(a - b) <= 1).mean()
    k0 = kappa_ci(a, b)
    kl = kappa_ci(a, b, "linear")
    kq = kappa_ci(a, b, "quadratic")
    # collapse to within-1 agreement kappa: treat |diff|<=1 as agreement is not a kappa; report
    # instead kappa on a 3-level collapse (lower / middle / higher) as a robustness check
    a3 = np.select([a <= 1, a <= 3], [0, 1], 2)
    b3 = np.select([b <= 1, b <= 3], [0, 1], 2)
    k3 = cohen_kappa_score(a3, b3, labels=[0, 1, 2])
    d = b - a
    higher = (d > 0).mean(); lower = (d < 0).mean()
    alpha_n = krippendorff.alpha(reliability_data=np.vstack([a, b]).astype(float), level_of_measurement="nominal")
    alpha_o = krippendorff.alpha(reliability_data=np.vstack([a, b]).astype(float), level_of_measurement="ordinal")
    w = stats.wilcoxon(d, zero_method="wilcox") if (d != 0).any() else None
    rec = {"pair": name, "n": n, "exact": exact, "within1": within1,
           "kappa": k0[0], "kappa_lo": k0[1], "kappa_hi": k0[2],
           "kappa_lin": kl[0], "kappa_lin_lo": kl[1], "kappa_lin_hi": kl[2],
           "kappa_quad": kq[0], "kappa_quad_lo": kq[1], "kappa_quad_hi": kq[2],
           "kappa_3bin": k3, "alpha_nominal": alpha_n, "alpha_ordinal": alpha_o,
           "second_higher": higher, "second_lower": lower, "mean_diff": d.mean(),
           "wilcoxon_p": w.pvalue if w else np.nan, "spearman": stats.spearmanr(a, b).statistic}
    return rec, pd.crosstab(pd.Categorical(a, range(6)), pd.Categorical(b, range(6)), dropna=False)


say("\n## Table 3. Agreement (primary labels)\n")
both = df.dropna(subset=["A_bloom", "B_bloom"])
blocks = []
confs = {}
r, c = agreement_block("A vs B", both.A_bloom_i, both.B_bloom_i); blocks.append(r); confs["A vs B"] = c
r, c = agreement_block("A vs LLM", df.A_bloom_i, df.llm_bloom_i); blocks.append(r); confs["A vs LLM"] = c
r, c = agreement_block("B vs LLM", both.B_bloom_i, both.llm_bloom_i); blocks.append(r); confs["B vs LLM"] = c
r, c = agreement_block("A vs LLM (B-subset)", both.A_bloom_i, both.llm_bloom_i); blocks.append(r)
agree = pd.DataFrame(blocks).set_index("pair")
agree.to_csv(OUT / "table3_agreement.csv")
say(agree.round(3).to_markdown())
say("\n(second_higher = share of items where the *second* labeller in the pair name assigned a higher level; "
    "wilcoxon_p tests whether level differences are centred on zero)")


def per_level(a, b):
    """Specific (positive) agreement per category: 2*agree_k / (n_k in a + n_k in b)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    rows = []
    for i, l in enumerate(LEVELS):
        na, nb, ag = (a == i).sum(), (b == i).sum(), ((a == i) & (b == i)).sum()
        rows.append({"level": l, "n1": na, "n2": nb, "agree": ag, "specific": 2 * ag / (na + nb) if na + nb else np.nan})
    return pd.DataFrame(rows).set_index("level")


say("\n### Per-level specific agreement (2*agree/(n1+n2))")
pl = {}
for name, s_, c1, c2 in [("A vs B", both, "A_bloom_i", "B_bloom_i"), ("A vs LLM", df, "A_bloom_i", "llm_bloom_i"), ("B vs LLM", both, "B_bloom_i", "llm_bloom_i")]:
    pl[name] = per_level(s_[c1], s_[c2])
    say(f"{name}:\n" + pl[name].round(2).to_markdown())
pd.concat(pl, axis=1).to_csv(OUT / "table3b_per_level_agreement.csv")

# lenient via secondary label
def lenient_secondary(x, ycols):
    s1 = {x["A_bloom"], x["A_sec"]} - {None}
    s2 = {x[ycols[0]], x[ycols[1]] if len(ycols) > 1 else None} - {None}
    return bool(s1 & s2)
sec_ab = both.apply(lambda x: bool(({x.A_bloom, x.A_sec} - {None}) & ({x.B_bloom, x.B_sec} - {None})), axis=1).mean()
sec_a_llm = df.apply(lambda x: x.llm_bloom in ({x.A_bloom, x.A_sec} - {None}), axis=1).mean()
sec_b_llm = both.apply(lambda x: x.llm_bloom in ({x.B_bloom, x.B_sec} - {None}), axis=1).mean()
say(f"\nSecondary labels used: A {df.A_sec.notna().sum()}/188, B {both.B_sec.notna().sum()}/{len(both)}. "
    f"Lenient (secondary) agreement: A–B {sec_ab:.2f}, A–LLM {sec_a_llm:.2f}, B–LLM {sec_b_llm:.2f}.")
# secondary is usually adjacent?
adj = df.dropna(subset=["A_sec"]).apply(lambda x: abs(LIDX[x.A_sec] - LIDX[x.A_bloom]), axis=1)
say(f"A's secondary label is adjacent to the primary in {(adj == 1).mean():.0%} of cases (distance counts: {dict(Counter(adj))}).")

# consensus / majority
df["consensus"] = np.where(df.A_bloom == df.B_bloom, df.A_bloom, None)
say(f"\nItems where A and B agree exactly: {df.consensus.notna().sum()} of {len(both)}; LLM matches that consensus on "
    f"{(df.consensus.notna() & (df.llm_bloom == df.consensus)).sum()} of them.")
# 3-way
three = both[(both.A_bloom == both.B_bloom) & (both.B_bloom == both.llm_bloom)]
say(f"All three agree on {len(three)} of {len(both)} items: {dict(Counter(three.llm_bloom))}")
alpha3 = krippendorff.alpha(reliability_data=np.vstack([both.A_bloom_i, both.B_bloom_i, both.llm_bloom_i]).astype(float),
                            level_of_measurement="ordinal")
alpha3n = krippendorff.alpha(reliability_data=np.vstack([both.A_bloom_i, both.B_bloom_i, both.llm_bloom_i]).astype(float),
                             level_of_measurement="nominal")
say(f"Krippendorff α for A, B and LLM together (n=118): nominal {alpha3n:.2f}, ordinal {alpha3:.2f}")

# agreement by exam (A vs B) and A vs LLM by exam
say("\nA vs B by exam:")
for e in EXAMS:
    s = both[both.exam == e]
    say(f"  {e}: exact {(s.A_bloom == s.B_bloom).mean():.2f}, within-1 {(abs(s.A_bloom_i - s.B_bloom_i) <= 1).mean():.2f}, "
        f"κ {cohen_kappa_score(s.A_bloom_i, s.B_bloom_i):.2f}")
say("A vs LLM by exam:")
for e in EXAMS:
    s = df[df.exam == e]
    say(f"  {e}: exact {(s.A_bloom == s.llm_bloom).mean():.2f}, within-1 {(abs(s.A_bloom_i - s.llm_bloom_i) <= 1).mean():.2f}, "
        f"κ {cohen_kappa_score(s.A_bloom_i, s.llm_bloom_i, labels=list(range(6))):.2f}, LLM higher {(s.llm_bloom_i > s.A_bloom_i).mean():.2f}")

# per-level agreement: when LLM says X, what do humans say
say("\nHuman level given LLM level (A, n=188): mean human level and share ≤ understand")
for l in LEVELS:
    s = df[df.llm_bloom == l]
    if len(s):
        say(f"  LLM={l:<10} n={len(s):<3} A mean {s.A_bloom_i.mean() + 1:.2f}  A≤understand {(s.A_bloom_i <= 1).mean():.2f}"
            + (f"  B mean {s.B_bloom_i.mean() + 1:.2f} (n={s.B_bloom_i.notna().sum()})" if s.B_bloom_i.notna().any() else ""))

# ================================================================ 3. LLM difficulty vs p
say("\n## Table 4. LLM difficulty label vs student difficulty\n")
t4 = df.groupby("llm_diff").agg(n=("p", "size"), p_mean=("p", "mean"), p_sd=("p", "std"), p_median=("p", "median"),
                                disc_mean=("disc", "mean")).reindex(DIFF)
t4.to_csv(OUT / "table4_llm_difficulty.csv")
say(t4.round(3).to_markdown())
kw = stats.kruskal(*[df[df.llm_diff == d].p for d in DIFF])
sp = stats.spearmanr(df.llm_diff_i, df.p)
say(f"Kruskal–Wallis H={kw.statistic:.2f}, p={kw.pvalue:.3f}; Spearman ρ(difficulty, p)={sp.statistic:.3f}, p={sp.pvalue:.3f}")
say("Within exam:")
for e in EXAMS:
    s = df[df.exam == e]
    sp_e = stats.spearmanr(s.llm_diff_i, s.p)
    say(f"  {e}: " + ", ".join(f"{d} {s[s.llm_diff == d].p.mean():.2f} (n={len(s[s.llm_diff == d])})" for d in DIFF)
        + f"; ρ={sp_e.statistic:.2f}, p={sp_e.pvalue:.2f}")
# stratified: Spearman on within-exam percentile ranks
sp_s = stats.spearmanr(df.llm_diff_i, df.p_rank_in_exam)
say(f"Spearman using within-exam percentile rank of p: ρ={sp_s.statistic:.3f}, p={sp_s.pvalue:.3f}")
# also mixed: medium vs hard only (easy n small)
mh = df[df.llm_diff.isin(["medium", "hard"])]
mw = stats.mannwhitneyu(mh[mh.llm_diff == "medium"].p, mh[mh.llm_diff == "hard"].p)
say(f"Mann–Whitney medium vs hard: U={mw.statistic:.0f}, p={mw.pvalue:.3f}; "
    f"rank-biserial r={1 - 2 * mw.statistic / (len(mh[mh.llm_diff == 'medium']) * len(mh[mh.llm_diff == 'hard'])):.2f}")
# difficulty vs discrimination
kwd = stats.kruskal(*[df[df.llm_diff == d].disc.dropna() for d in DIFF])
say(f"Discrimination by LLM difficulty: KW H={kwd.statistic:.2f}, p={kwd.pvalue:.3f}")

# ================================================================ 4. Bloom vs p and discrimination
say("\n## Table 5. Bloom level vs student difficulty and discrimination\n")
t5rows = []
for src, col, sub in [("LLM", "llm_bloom", df), ("A", "A_bloom", df), ("B", "B_bloom", both)]:
    for l in LEVELS:
        s = sub[sub[col] == l]
        if len(s):
            t5rows.append({"source": src, "level": l, "n": len(s), "p_mean": s.p.mean(), "p_sd": s.p.std(),
                           "p_median": s.p.median(), "disc_mean": s.disc.mean(), "disc_n": s.disc.notna().sum()})
t5 = pd.DataFrame(t5rows)
t5.to_csv(OUT / "table5_bloom_vs_p.csv", index=False)
say(t5.round(3).to_markdown(index=False))
say("")
for src, col, sub in [("LLM", "llm_bloom_i", df), ("A", "A_bloom_i", df), ("B", "B_bloom_i", both)]:
    s = sub.dropna(subset=[col])
    groups = [s[s[col] == i].p for i in range(6) if (s[col] == i).sum() >= 3]
    kw = stats.kruskal(*groups)
    sp = stats.spearmanr(s[col], s.p)
    sp_r = stats.spearmanr(s[col], s.p_rank_in_exam)
    sd = s.dropna(subset=["disc"])
    spd = stats.spearmanr(sd[col], sd.disc)
    # Jonckheere-Terpstra style trend: use Spearman as monotone trend test
    say(f"{src}: Spearman ρ(level, p)={sp.statistic:.3f} (p={sp.pvalue:.3f}); within-exam rank ρ={sp_r.statistic:.3f} "
        f"(p={sp_r.pvalue:.3f}); KW H={kw.statistic:.2f} (p={kw.pvalue:.3f}, groups n≥3); "
        f"ρ(level, discrimination)={spd.statistic:.3f} (p={spd.pvalue:.3f}, n={len(sd)})")
# consensus items only
cons = df.dropna(subset=["consensus"]).copy(); cons["c_i"] = cons.consensus.map(LIDX)
sp = stats.spearmanr(cons.c_i, cons.p); spr = stats.spearmanr(cons.c_i, cons.p_rank_in_exam)
say(f"Consensus (A=B, n={len(cons)}): ρ(level, p)={sp.statistic:.3f} (p={sp.pvalue:.3f}); within-exam ρ={spr.statistic:.3f} (p={spr.pvalue:.3f})")
say("  consensus level means: " + ", ".join(f"{l} {cons[cons.consensus == l].p.mean():.2f} (n={len(cons[cons.consensus == l])})" for l in LEVELS if (cons.consensus == l).any()))
# binary: recall (remember) vs higher, humans
for src, col, sub in [("A", "A_bloom_i", df), ("B", "B_bloom_i", both)]:
    s = sub.dropna(subset=[col])
    lo, hi = s[s[col] == 0].p, s[s[col] > 0].p
    mw = stats.mannwhitneyu(lo, hi)
    say(f"{src}: remember (n={len(lo)}, p̄={lo.mean():.2f}) vs higher (n={len(hi)}, p̄={hi.mean():.2f}): MWU p={mw.pvalue:.3f}")
# LLM: evaluate/analyze vs remember/understand
lo = df[df.llm_bloom_i <= 1].p; hi = df[df.llm_bloom_i >= 3].p
say(f"LLM: remember/understand (n={len(lo)}, p̄={lo.mean():.2f}) vs analyze+ (n={len(hi)}, p̄={hi.mean():.2f}): MWU p={stats.mannwhitneyu(lo, hi).pvalue:.3f}")

# ordinal regression-ish: OLS of p on human level with exam fixed effects (robustness)
import statsmodels.formula.api as smf
for src, col in [("LLM", "llm_bloom_i"), ("A", "A_bloom_i"), ("B", "B_bloom_i")]:
    s = df.dropna(subset=[col])
    m = smf.ols(f"p ~ {col} + C(exam)", data=s).fit(cov_type="HC3")
    say(f"OLS p ~ {src} level + exam FE: slope {m.params[col]:.3f} per level (95% CI {m.conf_int().loc[col, 0]:.3f} to {m.conf_int().loc[col, 1]:.3f}, p={m.pvalues[col]:.3f})")

# ================================================================ 5. duplicates
say("\n## Duplicate items (identical stems in Exam 1 and Exam 2)\n")
dup = df[df.duplicated("stem", keep=False)].sort_values(["stem", "exam"])
drows = []
for stem, g in dup.groupby("stem"):
    g = g.sort_values("exam")
    drows.append({"Exam 1 no": int(g.iloc[0].no), "Exam 2 no": int(g.iloc[1].no),
                  "p Exam 1": g.iloc[0].p, "p Exam 2": g.iloc[1].p,
                  "LLM": g.iloc[0].llm_bloom, "A (E2/M3)": f"{g.iloc[0].A_bloom}/{g.iloc[1].A_bloom}",
                  "B (E2/M3)": f"{g.iloc[0].B_bloom}/{g.iloc[1].B_bloom}"})
dd = pd.DataFrame(drows)
dd.to_csv(OUT / "table_duplicates.csv", index=False)
say(dd.round(2).to_markdown(index=False))
say(f"Intra-annotator consistency on duplicates: A {sum(r['A (E2/M3)'].split('/')[0] == r['A (E2/M3)'].split('/')[1] for r in drows)}/5, "
    f"B {sum(r['B (E2/M3)'].split('/')[0] == r['B (E2/M3)'].split('/')[1] for r in drows)}/5; "
    f"mean |Δp| = {np.mean([abs(r['p Exam 1'] - r['p Exam 2']) for r in drows]):.2f}")

# ================================================================ 6. surface features
say("\n## Surface features\n")
say(f"Stem length words: mean {df.stem_words.mean():.1f} (SD {df.stem_words.std():.1f}); correct option {df.correct_words.mean():.1f}, "
    f"distractors {df.incorrect_words.mean():.1f}.")
longest = sum(1 for q in data["questions"] if len(q["correct_answer"].split()) > max(len(o["text"].split()) for o in q["incorrect_answers"]))
shortest = sum(1 for q in data["questions"] if len(q["correct_answer"].split()) < min(len(o["text"].split()) for o in q["incorrect_answers"]))
say(f"Correct option longest: {longest}/188; shortest: {shortest}/188 (chance ≈ 47 each).")
for src, col in [("LLM", "llm_bloom_i"), ("A", "A_bloom_i"), ("B", "B_bloom_i")]:
    s = df.dropna(subset=[col])
    say(f"ρ(stem words, {src} level) = {stats.spearmanr(s.stem_words, s[col]).statistic:.2f}; ρ(stem words, p) = {stats.spearmanr(s.stem_words, s.p).statistic:.2f}")

# ================================================================ 7. comments
say("\n## Comments\n")
say(f"A wrote {df.A_comment.notna().sum()} comments, B wrote {both.B_comment.notna().sum()}.")
with open(OUT / "comments_full.txt", "w") as fh:
    for _, r in df.iterrows():
        for a in ["A", "B"]:
            if isinstance(r[f"{a}_comment"], str) and r[f"{a}_comment"]:
                sec = r[f"{a}_sec"] if isinstance(r[f"{a}_sec"], str) else ""
                fh.write(f"[{a}] {r.exam} Q{r.no} | {a}={r[f'{a}_bloom']}{'/' + sec if sec else ''} LLM={r.llm_bloom} p={r.p:.2f}\n{r[f'{a}_comment']}\n\n")

# ---- theme coding of comments (coded by hand; see comment_codes.py)
from comment_codes import CODES, THEMES
code_rows = []
for (exam, no, who), themes in CODES.items():
    r = df[(df.exam_orig == exam) & (df.no == no)].iloc[0]
    code_rows.append({"exam": exam, "no": no, "annotator": who, "themes": ";".join(themes),
                      "B_level": r.B_bloom, "A_level": r.A_bloom, "llm_level": r.llm_bloom, "p": r.p,
                      "llm_minus_B": (LIDX[r.llm_bloom] - LIDX[r.B_bloom]) if isinstance(r.B_bloom, str) else np.nan})
cc = pd.DataFrame(code_rows)
cc.to_csv(OUT / "comment_codes.csv", index=False)
say("Theme counts over %d comments:" % len(cc))
for t, desc in THEMES.items():
    say(f"  {t:<14} {cc.themes.str.contains(t).sum():>3}  {desc}")
# items flagged by B for weak distractors vs. B's other items
flag = set((r.exam, r.no) for r in cc.itertuples() if "distractors" in r.themes and r.annotator == "B")
both_ = both.copy(); both_["flag"] = [(e, int(n)) in flag for e, n in zip(both_.exam_orig, both_.no)]  # CODES use export exam names
f1, f0 = both_[both_.flag], both_[~both_.flag]
say("Flagged vs not, per exam (n flagged, mean p flagged vs not, MWU p): " + "; ".join(
    f"{e}: {both_[(both_.exam == e) & both_.flag].shape[0]}, {both_[(both_.exam == e) & both_.flag].p.mean():.2f} vs "
    f"{both_[(both_.exam == e) & ~both_.flag].p.mean():.2f}, p={stats.mannwhitneyu(both_[(both_.exam == e) & both_.flag].p, both_[(both_.exam == e) & ~both_.flag].p).pvalue:.3f}"
    for e in EXAMS))
spr_flag = stats.mannwhitneyu(f1.p_rank_in_exam, f0.p_rank_in_exam)
say(f"Flagged vs not on within-exam percentile rank of p: mean rank {f1.p_rank_in_exam.mean():.2f} vs {f0.p_rank_in_exam.mean():.2f}, MWU p={spr_flag.pvalue:.3f}")
say(f"B flagged weak distractors on {len(f1)} of {len(both_)} items. On those: B mean level {f1.B_bloom_i.mean() + 1:.2f} vs {f0.B_bloom_i.mean() + 1:.2f} elsewhere; "
    f"LLM mean level {f1.llm_bloom_i.mean() + 1:.2f} vs {f0.llm_bloom_i.mean() + 1:.2f}; LLM higher than B on {(f1.llm_bloom_i > f1.B_bloom_i).mean():.0%} vs {(f0.llm_bloom_i > f0.B_bloom_i).mean():.0%}; "
    f"p̄ {f1.p.mean():.2f} vs {f0.p.mean():.2f} (MWU p={stats.mannwhitneyu(f1.p, f0.p).pvalue:.2f}); "
    f"A–B exact agreement {(f1.A_bloom == f1.B_bloom).mean():.0%} vs {(f0.A_bloom == f0.B_bloom).mean():.0%}")
say(f"B's level on flagged items: {dict(Counter(f1.B_bloom))}; LLM's: {dict(Counter(f1.llm_bloom))}")

# ================================================================ 8. Fable 5 labels
F_COND = {"bloom_instructed": "F_bloom_i", "bloom_bare": "F_bloom_b", "diff_instructed": "F_diff_i", "diff_bare": "F_diff_b"}
FL = HERE / "data" / "llm_labels_structured_claude-fable-5.json"  # structured outputs, provider-default temperature
if FL.exists():
    lab = json.loads(FL.read_text())
    for cond, col in F_COND.items():
        m = {(RENAME[v["exam"]], v["question_no"]): v["label"] for k, v in lab.items() if v["condition"] == cond}
        df[col] = [m.get((e, n)) for e, n in zip(df.exam, df.no)]
    for col in ["F_bloom_i", "F_bloom_b"]:
        df[col + "_n"] = df[col].map(LIDX)
    for col in ["F_diff_i", "F_diff_b"]:
        df[col + "_n"] = df[col].map({d: i for i, d in enumerate(DIFF)})
    both = df.dropna(subset=["A_bloom", "B_bloom"])
    say("\n## Fable 5 as a classifier\n")
    say(f"coverage: " + ", ".join(f"{c} {df[c].notna().sum()}" for c in F_COND.values()))
    say("Bloom distributions:\n" + pd.DataFrame({
        "Fable instructed": df.F_bloom_i.value_counts().reindex(LEVELS, fill_value=0),
        "Fable bare": df.F_bloom_b.value_counts().reindex(LEVELS, fill_value=0),
        "Gemini": df.llm_bloom.value_counts().reindex(LEVELS, fill_value=0),
        "A": df.A_bloom.value_counts().reindex(LEVELS, fill_value=0),
        "B": df.B_bloom.value_counts().reindex(LEVELS, fill_value=0)}).to_markdown())
    say(f"mean level: instructed {df.F_bloom_i_n.mean() + 1:.2f}, bare {df.F_bloom_b_n.mean() + 1:.2f}")
    fb = []
    pairs = [("Fable-instr vs A", df, "F_bloom_i_n", "A_bloom_i"), ("Fable-instr vs B", both, "F_bloom_i_n", "B_bloom_i"),
             ("Fable-instr vs Gemini", df, "F_bloom_i_n", "llm_bloom_i"), ("Fable-bare vs A", df, "F_bloom_b_n", "A_bloom_i"),
             ("Fable-bare vs B", both, "F_bloom_b_n", "B_bloom_i"), ("Fable-bare vs Gemini", df, "F_bloom_b_n", "llm_bloom_i"),
             ("Fable-instr vs Fable-bare", df, "F_bloom_i_n", "F_bloom_b_n")]
    for name, sub, c1, c2 in pairs:
        s_ = sub.dropna(subset=[c1, c2])
        rec, cm = agreement_block(name, s_[c1], s_[c2]); fb.append(rec); confs[name] = cm
    fbt = pd.DataFrame(fb).set_index("pair")
    fbt.to_csv(OUT / "table6_fable_bloom_agreement.csv")
    say("\n### Table 6. Fable Bloom agreement (second_higher = share where the *second* source is higher)\n")
    say(fbt[["n", "exact", "within1", "kappa", "kappa_lo", "kappa_hi", "kappa_quad", "kappa_quad_lo", "kappa_quad_hi", "alpha_ordinal", "second_higher", "second_lower", "wilcoxon_p"]].round(3).to_markdown())
    # three humans-ish: A, B, Fable-instr alpha
    a3 = krippendorff.alpha(reliability_data=np.vstack([both.A_bloom_i, both.B_bloom_i, both.F_bloom_i_n]).astype(float), level_of_measurement="ordinal")
    a3n = krippendorff.alpha(reliability_data=np.vstack([both.A_bloom_i, both.B_bloom_i, both.F_bloom_i_n]).astype(float), level_of_measurement="nominal")
    say(f"α over A, B, Fable-instructed (n=118): nominal {a3n:.2f}, ordinal {a3:.2f}")
    cons = both[both.A_bloom == both.B_bloom]
    say(f"Fable-instructed matches human consensus on {(cons.F_bloom_i == cons.A_bloom).sum()} of {len(cons)}; Gemini on {(cons.llm_bloom == cons.A_bloom).sum()}")
    s_ = df.dropna(subset=["F_bloom_i_n"])
    say("Per-level specific agreement Fable-instr vs A:\n" + per_level(s_.F_bloom_i_n, s_.A_bloom_i).round(2).to_markdown())
    s_ = both.dropna(subset=["F_bloom_i_n"])
    say("Per-level specific agreement Fable-instr vs B:\n" + per_level(s_.F_bloom_i_n, s_.B_bloom_i).round(2).to_markdown())
    # relation with p
    say("\nFable Bloom vs p:")
    for col, lab_ in [("F_bloom_i_n", "instructed"), ("F_bloom_b_n", "bare")]:
        s_ = df.dropna(subset=[col])
        sp = stats.spearmanr(s_[col], s_.p); spr = stats.spearmanr(s_[col], s_.p_rank_in_exam)
        groups = [s_[s_[col] == i].p for i in range(6) if (s_[col] == i).sum() >= 3]
        kw = stats.kruskal(*groups)
        sd = s_.dropna(subset=["disc"]); spd = stats.spearmanr(sd[col], sd.disc)
        say(f"  {lab_}: ρ(level,p)={sp.statistic:.3f} (p={sp.pvalue:.3f}); within-exam ρ={spr.statistic:.3f} (p={spr.pvalue:.3f}); KW p={kw.pvalue:.3f}; ρ(level,disc)={spd.statistic:.3f}")
        say("    p̄ by level: " + ", ".join(f"{LEVELS[i]} {s_[s_[col] == i].p.mean():.2f} (n={(s_[col] == i).sum()})" for i in range(6) if (s_[col] == i).sum()))
        lo, hi = s_[s_[col] == 0].p, s_[s_[col] > 0].p
        if len(lo) >= 3:
            say(f"    remember (n={len(lo)}, p̄={lo.mean():.2f}) vs higher (n={len(hi)}, p̄={hi.mean():.2f}): MWU p={stats.mannwhitneyu(lo, hi).pvalue:.3f}")
    # ---- difficulty
    say("\n### Fable difficulty labels\n")
    say("distributions:\n" + pd.DataFrame({"Fable instructed": df.F_diff_i.value_counts().reindex(DIFF, fill_value=0),
                                          "Fable bare": df.F_diff_b.value_counts().reindex(DIFF, fill_value=0),
                                          "Gemini": df.llm_diff.value_counts().reindex(DIFF, fill_value=0)}).to_markdown())
    say("by exam (instructed):\n" + pd.crosstab(df.exam, df.F_diff_i).reindex(EXAMS).reindex(columns=DIFF, fill_value=0).to_markdown())
    say("by exam (bare):\n" + pd.crosstab(df.exam, df.F_diff_b).reindex(EXAMS).reindex(columns=DIFF, fill_value=0).to_markdown())
    fd = []
    for col, lab_ in [("F_diff_i_n", "instructed"), ("F_diff_b_n", "bare"), ("llm_diff_i", "Gemini")]:
        s_ = df.dropna(subset=[col])
        sp = stats.spearmanr(s_[col], s_.p); spr = stats.spearmanr(s_[col], s_.p_rank_in_exam)
        kw = stats.kruskal(*[s_[s_[col] == i].p for i in range(3) if (s_[col] == i).sum() >= 3])
        means = {DIFF[i]: s_[s_[col] == i].p.mean() for i in range(3) if (s_[col] == i).sum()}
        per_exam = {e: stats.spearmanr(s_[s_.exam == e][col], s_[s_.exam == e].p).statistic for e in EXAMS}
        fd.append({"source": lab_, "n": len(s_), **{f"p_{k}": v for k, v in means.items()}, "rho_all": sp.statistic, "rho_all_p": sp.pvalue,
                   "rho_within": spr.statistic, "rho_within_p": spr.pvalue, "kw_p": kw.pvalue, **{f"rho_{e}": v for e, v in per_exam.items()}})
    fdt = pd.DataFrame(fd).set_index("source"); fdt.to_csv(OUT / "table7_difficulty_vs_p.csv")
    say("\n### Table 7. Difficulty labels vs p\n" + fdt.round(3).to_markdown())
    # agreement between difficulty sources
    for n1, c1, n2, c2 in [("Fable-instr", "F_diff_i_n", "Gemini", "llm_diff_i"), ("Fable-bare", "F_diff_b_n", "Gemini", "llm_diff_i"), ("Fable-instr", "F_diff_i_n", "Fable-bare", "F_diff_b_n")]:
        s_ = df.dropna(subset=[c1, c2]); a, b = s_[c1].astype(int), s_[c2].astype(int)
        say(f"{n1} vs {n2}: exact {(a == b).mean():.2f}, κ {cohen_kappa_score(a, b, labels=[0, 1, 2]):.2f}, κ-lin {cohen_kappa_score(a, b, labels=[0, 1, 2], weights='linear'):.2f}, {n2} higher {(b > a).mean():.2f}, lower {(b < a).mean():.2f}")
    # calibration of instructed bands
    s_ = df.dropna(subset=["F_diff_i"])
    band = {"easy": lambda p: p > 0.85, "medium": lambda p: 0.60 <= p <= 0.85, "hard": lambda p: p < 0.60}
    hit = np.mean([band[d](p) for d, p in zip(s_.F_diff_i, s_.p)])
    say(f"instructed bands: share of items whose observed p falls in the predicted band = {hit:.2f} (chance from marginals ≈ {sum((s_.F_diff_i == d).mean() * np.mean([band[d](p) for p in s_.p]) for d in DIFF):.2f})")
    # rationales mentioning distractors/elimination
    rat = [v.get("rationale") or "" for v in lab.values() if v["condition"] == "diff_instructed"]
    kw_ = ["eliminat", "distractor", "implausib", "obvious", "rule out", "process of elimination"]
    say(f"instructed-difficulty rationales mentioning elimination/distractor quality: {sum(any(k in r.lower() for k in kw_) for r in rat)} of {len(rat)}")
    rat = [v.get("rationale") or "" for v in lab.values() if v["condition"] == "bloom_instructed"]
    say(f"instructed-Bloom rationales mentioning elimination/distractor quality: {sum(any(k in r.lower() for k in kw_) for r in rat)} of {len(rat)}")
else:
    say("\n(no Fable labels found)")

# ================================================================ 9. Gemini 3.1 Pro classifying cold (same conditions)
GL = HERE / "data" / "llm_labels_structured_gemini-3.1-pro-preview.json"
if GL.exists() and "F_bloom_i_n" in df:
    glab = json.loads(GL.read_text())
    G_COND = {"bloom_instructed": "G_bloom_i", "bloom_bare": "G_bloom_b", "diff_instructed": "G_diff_i", "diff_bare": "G_diff_b"}
    for cond, col in G_COND.items():
        m = {(RENAME[v["exam"]], v["question_no"]): v["label"] for k, v in glab.items() if v["condition"] == cond}
        df[col] = [m.get((e, n)) for e, n in zip(df.exam, df.no)]
    for col in ["G_bloom_i", "G_bloom_b"]:
        df[col + "_n"] = df[col].map(LIDX)
    for col in ["G_diff_i", "G_diff_b"]:
        df[col + "_n"] = df[col].map({d: i for i, d in enumerate(DIFF)})
    both = df.dropna(subset=["A_bloom", "B_bloom"])
    say("\n## Gemini 3.1 Pro classifying cold\n")
    say("coverage: " + ", ".join(f"{c} {df[c].notna().sum()}" for c in G_COND.values()))
    say("Bloom distributions:\n" + pd.DataFrame({
        "Gemini-cold instructed": df.G_bloom_i.value_counts().reindex(LEVELS, fill_value=0),
        "Gemini-cold bare": df.G_bloom_b.value_counts().reindex(LEVELS, fill_value=0),
        "Gemini self": df.llm_bloom.value_counts().reindex(LEVELS, fill_value=0)}).to_markdown())
    say(f"mean level: cold-instructed {df.G_bloom_i_n.mean() + 1:.2f}, cold-bare {df.G_bloom_b_n.mean() + 1:.2f}, self {df.llm_bloom_i.mean() + 1:.2f}")
    gb = []
    for name, sub, c1, c2 in [("Gemini-cold-instr vs A", df, "G_bloom_i_n", "A_bloom_i"), ("Gemini-cold-instr vs B", both, "G_bloom_i_n", "B_bloom_i"),
                              ("Gemini-cold-instr vs Gemini-self", df, "G_bloom_i_n", "llm_bloom_i"), ("Gemini-cold-instr vs Fable-instr", df, "G_bloom_i_n", "F_bloom_i_n"),
                              ("Gemini-cold-bare vs A", df, "G_bloom_b_n", "A_bloom_i"), ("Gemini-cold-bare vs B", both, "G_bloom_b_n", "B_bloom_i"),
                              ("Gemini-cold-bare vs Gemini-self", df, "G_bloom_b_n", "llm_bloom_i")]:
        s_ = sub.dropna(subset=[c1, c2])
        rec, cm = agreement_block(name, s_[c1], s_[c2]); gb.append(rec)
    gbt = pd.DataFrame(gb).set_index("pair"); gbt.to_csv(OUT / "table8_gemini_cold_agreement.csv")
    say("\n### Table 8. Gemini-cold Bloom agreement\n" + gbt[["n", "exact", "within1", "kappa", "kappa_lo", "kappa_hi", "kappa_quad", "kappa_quad_lo", "kappa_quad_hi", "second_higher", "second_lower"]].round(3).to_markdown())
    cons = both[both.A_bloom == both.B_bloom]
    say(f"Gemini-cold-instructed matches human consensus on {(cons.G_bloom_i == cons.A_bloom).sum()} of {len(cons)} (bare: {(cons.G_bloom_b == cons.A_bloom).sum()})")
    s_ = df.dropna(subset=["G_bloom_i_n"])
    say("Per-level specific agreement Gemini-cold-instr vs A:\n" + per_level(s_.G_bloom_i_n, s_.A_bloom_i).round(2).to_markdown())
    s_ = df.dropna(subset=["G_bloom_i_n"])
    say("Per-level specific agreement Gemini-cold-instr vs Gemini-self:\n" + per_level(s_.G_bloom_i_n, s_.llm_bloom_i).round(2).to_markdown())
    for col, lab_ in [("G_bloom_i_n", "cold-instructed"), ("G_bloom_b_n", "cold-bare")]:
        s_ = df.dropna(subset=[col]); sp = stats.spearmanr(s_[col], s_.p); spr = stats.spearmanr(s_[col], s_.p_rank_in_exam)
        say(f"  Bloom {lab_} vs p: ρ={sp.statistic:.3f} (p={sp.pvalue:.3f}); within-exam ρ={spr.statistic:.3f} (p={spr.pvalue:.3f})")
    say("difficulty distributions:\n" + pd.DataFrame({"Gemini-cold instr": df.G_diff_i.value_counts().reindex(DIFF, fill_value=0),
                                                     "Gemini-cold bare": df.G_diff_b.value_counts().reindex(DIFF, fill_value=0)}).to_markdown())
    for col, lab_ in [("G_diff_i_n", "cold-instructed"), ("G_diff_b_n", "cold-bare")]:
        s_ = df.dropna(subset=[col]); sp = stats.spearmanr(s_[col], s_.p); spr = stats.spearmanr(s_[col], s_.p_rank_in_exam)
        means = {DIFF[i]: round(s_[s_[col] == i].p.mean(), 3) for i in range(3) if (s_[col] == i).sum()}
        per_exam = {e: round(stats.spearmanr(s_[s_.exam == e][col], s_[s_.exam == e].p).statistic, 3) for e in EXAMS if s_[s_.exam == e][col].nunique() > 1}
        say(f"  difficulty {lab_}: n={len(s_)} p̄ by label {means}; ρ all={sp.statistic:.3f} (p={sp.pvalue:.3f}); within-exam ρ={spr.statistic:.3f} (p={spr.pvalue:.3f}); per exam {per_exam}")
    for n1, c1, n2, c2 in [("Gemini-cold-instr", "G_diff_i_n", "Gemini-self", "llm_diff_i"), ("Gemini-cold-instr", "G_diff_i_n", "Fable-instr", "F_diff_i_n")]:
        s_ = df.dropna(subset=[c1, c2]); a, b = s_[c1].astype(int), s_[c2].astype(int)
        say(f"  {n1} vs {n2}: exact {(a == b).mean():.2f}, κ-lin {cohen_kappa_score(a, b, labels=[0, 1, 2], weights='linear'):.2f}, {n2} higher {(b > a).mean():.2f} lower {(b < a).mean():.2f}")

# ================================================================ figures
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 120,
})
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---- Figure 1: confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
for ax, (name, cm), lab in zip(axes, confs.items(), "ABC"):
    m = cm.values.astype(float)
    ax.imshow(m, cmap="Greys", vmin=0, vmax=max(m.max(), 1))
    for i in range(6):
        for j in range(6):
            v = int(m[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 * m.max() else "black")
    ax.plot([-0.5, 5.5], [-0.5, 5.5], color="#888", lw=0.6, ls="--")
    r_, c_ = name.split(" vs ")
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT); ax.set_yticks(range(6)); ax.set_yticklabels(SHORT)
    ax.set_xlabel(f"{c_} level"); ax.set_ylabel(f"{r_} level")
    ax.spines[:].set_visible(True)
    n = int(m.sum()); ex = np.trace(m) / n; w1 = sum(m[i, j] for i in range(6) for j in range(6) if abs(i - j) <= 1) / n
    ax.set_title(f"{lab}  {name} (n = {n})\nexact {ex:.0%}, within one level {w1:.0%}", loc="left", fontsize=7.5)
fig.tight_layout(w_pad=1.5)
save(fig, "fig1_confusion")

# ---- Figure 2: p by Bloom level for LLM / A / B (strip + median/IQR), plus discrimination
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharey="row")
for col_i, (src, col, sub) in enumerate([("LLM", "llm_bloom_i", df), ("Annotator A", "A_bloom_i", df), ("Annotator B", "B_bloom_i", both)]):
    s = sub.dropna(subset=[col])
    for row_i, (y, ylab) in enumerate([("p", "Proportion correct"), ("disc", "Item–rest correlation")]):
        ax = axes[row_i, col_i]
        for lv in range(6):
            v = s[s[col] == lv][y].dropna()
            if len(v) == 0:
                continue
            x = lv + RNG.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=7, color="#555", alpha=0.45, lw=0)
            if len(v) >= 3:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([lv - 0.28, lv + 0.28], [med, med], color="black", lw=1.6)
                ax.plot([lv, lv], [q1, q3], color="black", lw=0.8)
            if row_i == 0:
                ax.text(lv, 0.10, f"n = {len(v)}" if lv == 0 else f"{len(v)}", ha="center", va="center", fontsize=6, color="#444")
        ax.set_xticks(range(6)); ax.set_xticklabels(SHORT)
        if row_i == 0:
            ax.set_title(f"{'ABC'[col_i]}  {src}", loc="left")
            ax.set_ylim(0.03, 1.05)
        else:
            ax.set_ylim(-0.55, 1.0); ax.axhline(0, color="#aaa", lw=0.6)
            ax.set_xlabel("Bloom level")
        if col_i == 0:
            ax.set_ylabel(ylab)
fig.tight_layout(h_pad=1.2, w_pad=1.0)
save(fig, "fig2_bloom_vs_difficulty")

# ---- Figure 2b: 2 x 3 version (p only) for the paper; short source names as in the paper's tables
PANELS = [("Gemini self", "llm_bloom_i", df), ("Rater A", "A_bloom_i", df), ("Rater B", "B_bloom_i", both)]
if "F_bloom_i_n" in df:
    PANELS += [("Fable instr.", "F_bloom_i_n", df), ("Fable bare", "F_bloom_b_n", df)]
if "G_bloom_i_n" in df:
    PANELS.append(("Gemini cold", "G_bloom_i_n", df))
ncol = 3; nrow = int(np.ceil(len(PANELS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(14.8 / 2.54, 0.90 * nrow), sharey=True)  # printed 1:1 at 14.8 cm
axes = np.atleast_2d(axes)
for k, (src, col, sub) in enumerate(PANELS):
    ax = axes[k // ncol, k % ncol]; s_ = sub.dropna(subset=[col])
    for lv in range(6):
        v = s_[s_[col] == lv].p
        if len(v) == 0:
            continue
        ax.scatter(lv + RNG.uniform(-0.18, 0.18, len(v)), v, s=7, color="#555", alpha=0.45, lw=0)
        if len(v) >= 3:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([lv - 0.28, lv + 0.28], [med, med], color="black", lw=1.6); ax.plot([lv, lv], [q1, q3], color="black", lw=0.8)
    ax.set_xticks(range(6)); ax.set_ylim(0.2, 1.04); ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_xticklabels([f"{SHORT[lv]}\n{(s_[col] == lv).sum() or ''}" for lv in range(6)], linespacing=1.0)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.set_title(f"{'ABCDEF'[k]}  {src}", loc="left", fontsize=8, pad=2)
    if k % ncol == 0:
        ax.set_ylabel("Facility")
for k in range(len(PANELS), nrow * ncol):
    axes[k // ncol, k % ncol].axis("off")
fig.tight_layout(w_pad=0.6, h_pad=0.3, pad=0.2)
save(fig, "fig2_bloom_vs_p")

# ---- Figure 5: agreement graph (co-author suggestion): sources as nodes, edge width ~ Cohen's kappa
SOURCES = [("Rater A", "A_bloom_i"), ("Rater B", "B_bloom_i"), ("Gemini self", "llm_bloom_i")]
if "F_bloom_i_n" in df:
    SOURCES += [("Fable instr.", "F_bloom_i_n"), ("Fable bare", "F_bloom_b_n")]
if "G_bloom_i_n" in df:
    SOURCES.append(("Gemini cold", "G_bloom_i_n"))
say("\n## Agreement graph (all pairs, unweighted kappa)\n")
kap = {}
for (n1, c1), (n2, c2) in combinations(SOURCES, 2):
    s_ = df.dropna(subset=[c1, c2])
    kap[(n1, n2)] = cohen_kappa_score(s_[c1].astype(int), s_[c2].astype(int), labels=list(range(6)))
    say(f"  {n1} vs {n2}: n={len(s_)} κ={kap[(n1, n2)]:.2f}")
pd.Series(kap).to_csv(OUT / "table9_kappa_all_pairs.csv")
# drawn as a lower-triangle matrix (the graph version had overlapping edge labels at print size)
ORDER = [n for n in ["Rater A", "Rater B", "Fable instr.", "Fable bare", "Gemini cold", "Gemini self"] if n in dict(SOURCES)]
K = lambda a, b: kap.get((a, b), kap.get((b, a)))
fig, ax = plt.subplots(figsize=(7.4 / 2.54, 5.0 / 2.54))  # printed 1:1 at 7.4 cm
for i, r_ in enumerate(ORDER[1:], start=1):
    for j, c_ in enumerate(ORDER[:i]):
        k = K(r_, c_)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 1.5), 1, 1, color=plt.cm.Greys(0.08 + 0.85 * min(max(k, 0) / 0.8, 1)), lw=0))
        ax.text(j, i - 1, f"{k:.2f}", ha="center", va="center", fontsize=7.5, color="white" if k > 0.45 else "black")
ax.set_xlim(-0.5, len(ORDER) - 1.5); ax.set_ylim(len(ORDER) - 1.5, -0.5)
ax.set_xticks(range(len(ORDER) - 1)); ax.set_xticklabels([n.replace(" ", "\n") for n in ORDER[:-1]], fontsize=7, linespacing=0.95)
ax.set_yticks(range(len(ORDER) - 1)); ax.set_yticklabels(ORDER[1:], fontsize=7)
ax.xaxis.tick_top(); ax.tick_params(length=0, pad=2); ax.spines[:].set_visible(False)
ax.plot([-0.5, len(ORDER) - 1.5], [3.5, 3.5], color="black", lw=0.8)  # Gemini self row below the line
fig.tight_layout(pad=0.1)
save(fig, "fig5_agreement_graph")

# ---- power: smallest Spearman/Pearson correlation detectable with 80% power (two-sided alpha .05), Fisher z
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
say("\n## Power (Fisher z): smallest |ρ| detectable at 80% power, α = 0.05 two-sided")
for n_ in [188, 54, 64, 70]:
    say(f"  n={n_}: |ρ| ≈ {np.tanh(z / np.sqrt(n_ - 3)):.2f}")

# ---- Figure 3: p by difficulty label, per exam (Gemini self-label; Fable instructed)
DPANELS = [("Gemini (self-label)", "llm_diff")]
if "F_diff_i" in df:
    DPANELS.append(("Fable 5, instructed", "F_diff_i"))
fig, axes = plt.subplots(1, len(DPANELS), figsize=(3.4 * len(DPANELS), 2.4), sharey=True)
axes = np.atleast_1d(axes)
markers = {"Exam 1": "o", "Exam 2": "s", "Exam 3": "^"}
offs = {"Exam 1": -0.22, "Exam 2": 0, "Exam 3": 0.22}
for ax, (title, col), lab_ in zip(axes, DPANELS, "AB"):
    for e in EXAMS:
        s_ = df[df.exam == e]
        for i, d in enumerate(DIFF):
            v = s_[s_[col] == d].p
            if len(v) == 0:
                continue
            x = i + offs[e] + RNG.uniform(-0.06, 0.06, len(v))
            ax.scatter(x, v, s=8, marker=markers[e], color="#666", alpha=0.45, lw=0, label=e if (i == 1 and lab_ == "A") else None)
            ax.plot([i + offs[e] - 0.1, i + offs[e] + 0.1], [v.median(), v.median()], color="black", lw=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{d}\n(n = {(df[col] == d).sum()})" for d in DIFF])
    ax.set_xlabel("Difficulty label"); ax.set_ylim(0.15, 1.05); ax.set_title(f"{lab_}  {title}", loc="left")
axes[0].set_ylabel("Proportion correct")
axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5 if len(DPANELS) == 1 else 1.05, 1.0), ncol=3, handletextpad=0.2, columnspacing=1.0)
fig.tight_layout()
save(fig, "fig3_llm_difficulty")

# ---- Figure 4: level distributions (stacked/grouped bars)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
w = 0.26
for i, (lab, col, sub, hatch) in enumerate([("LLM (n = 188)", "llm_bloom", df, ""), ("Annotator A (n = 188)", "A_bloom", df, "///"), (f"Annotator B (n = {len(both)})", "B_bloom", both, "...")]):
    cnt = sub[col].value_counts(normalize=True).reindex(LEVELS, fill_value=0)
    ax.bar(np.arange(6) + (i - 1) * w, cnt.values * 100, w, label=lab, color=["#222", "#999", "#ddd"][i], edgecolor="black", lw=0.5, hatch=hatch)
ax.set_xticks(range(6)); ax.set_xticklabels(SHORT); ax.set_ylabel("Share of items (%)"); ax.set_ylim(0, 48)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig4_distributions")

(OUT / "results.md").write_text("\n".join(md) + "\n")
df.to_csv(OUT / "items_merged.csv", index=False)
print(f"\nwrote {OUT}")
