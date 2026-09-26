#!/usr/bin/env python3
"""Build the camera-ready supplementary material and the comment-coding review sheet.

  1. ../materials/  - the package published at https://github.com/Haffi112/bloom-analysis (Section 3.5):
     prompts, model identifiers and dates, anonymised item-level labels and scores (no item
     texts, no rater comments, no student-level data), the comment codebook, analysis code.
     The folder is a git repository; everything except .git is rebuilt on each run.
  2. ../feedback/coding_check/comment_coding_sheet.xlsx - the 100 rater comments in random order,
     each with its item (stem, correct answer A, distractors B-D) and the first coding. A second
     author marks Agree or Disagree for every theme. The sheet shows no labels, facility, item
     numbers or rater. It contains item texts, so it is not part of the public package. The key
     (row id -> item, rater) goes to data/coding_check_key.csv. Score it with coding_check.py.

Run after analyze.py:  uv run --with openpyxl python make_materials.py
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from comment_codes import CODES
from llm_classify import INSTR, SYSTEM, item_text  # INSTR = the raters' instructions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAT = ROOT / "materials"
REPO_URL = "https://github.com/Haffi112/bloom-analysis"
ANON = json.loads((HERE / "data" / "raters.json").read_text())  # private: login -> rater letter

# The seven themes reported in Table 5: key -> (short title, paper wording, detailed coding rule)
THEMES = {
    "distractors": (
        "Weak distractors",
        "One or more distractors implausible, contradictory or excludable without domain knowledge",
        "The comment says that at least one wrong option (B, C or D) is nonsensical, silly, strange or implausible, "
        "contradicts the stem, or can be ruled out without knowing the machine-learning topic."),
    "recall_dep": (
        "Depends on recall or teaching",
        "Level depends on what was taught; key repeats a statement from the material",
        "The comment says that the answer hinges on remembering a specific statement, definition or example from "
        "the course material (for instance, the key repeats the text), or that the Bloom level depends on how the "
        "topic was taught."),
    "common_sense": (
        "Common sense suffices",
        "Answerable by reading comprehension or common sense",
        "The comment says that the item can be answered by careful reading, general knowledge or common sense, "
        "without any machine-learning knowledge."),
    "flaw": (
        "Flawed stem or key",
        "Stem or key ambiguous or flawed",
        "The comment says that the question or the correct answer is ambiguous, wrong, badly worded or confusing, "
        "or that another option could also be correct. A general complaint about the quality of the item counts."),
    "lower_bound": (
        "Level capped by weak alternatives",
        "Level cannot be evaluate/analyze because the alternatives are indefensible",
        "The comment argues explicitly that the item cannot be at a high level such as evaluate or analyze, "
        "because the alternatives are not defensible choices that a student would have to weigh."),
    "giveaway": (
        "Key gives itself away",
        "Wording of the key gives the answer away",
        "The comment says that the wording of the correct answer reveals it, for example because it repeats words "
        "from the stem or stands out from the other options."),
    "unfamiliar": (
        "Rater unfamiliar with topic",
        "Rater unfamiliar with the topic",
        "The rater says that they are not familiar with the topic of the item."),
}

# ------------------------------------------------------------------ comments and their items
data = json.loads((HERE / "data" / "export.json").read_text())
comments = []
for q in data["questions"]:
    for user, who in ANON.items():
        ann = q["annotations"].get(user)
        if ann and ann.get("comment"):
            comments.append({"exam_orig": q["exam"], "no": q["question_no"], "rater": who,
                             "comment": ann["comment"].strip(), "stem": q["question_text"],
                             "key": q["correct_answer"], "distractors": [o["text"] for o in q["incorrect_answers"]]})
assert len(comments) == len(CODES) == 100, (len(comments), len(CODES))


def coded(c: dict, theme: str) -> bool:
    return theme in CODES[(c["exam_orig"], c["no"], c["rater"])]


def examples(theme: str, k: int = 2) -> list[str]:
    """Shortest comments carrying the theme (and no 'duplicate' note), as codebook examples."""
    pool = [c["comment"] for c in comments
            if coded(c, theme) and "duplicate" not in CODES[(c["exam_orig"], c["no"], c["rater"])]]
    return sorted(pool, key=len)[:k]


# ================================================================ 1. materials package
MAT.mkdir(exist_ok=True)
for p in MAT.iterdir():  # rebuild everything but the git history
    if p.name != ".git":
        shutil.rmtree(p) if p.is_dir() else p.unlink()
for d in ["prompts", "data", "code"]:
    (MAT / d).mkdir()

(MAT / "prompts" / "rater_instructions.txt").write_text(INSTR + "\n", encoding="utf-8")
(MAT / "prompts" / "classification_prompts.json").write_text(json.dumps({
    "models": {"Fable": "anthropic/claude-fable-5", "Gemini cold": "google/gemini-3.1-pro-preview"},
    "access": "OpenRouter chat completions API, August 2026",
    "sampling": "provider default sampling settings; structured outputs (response_format json_schema, strict), "
                "provider.require_parameters = true",
    "conditions": {"bloom_instructed": "Fable instr. / Gemini cold", "bloom_bare": "Fable bare",
                   "diff_instructed": "Table 4 'instr.' rows", "diff_bare": "Table 4 'bare' rows"},
    "system_prompts": SYSTEM,
    "user_message_format": item_text({"question_text": "<stem>", "correct_answer": "<key>",
                                      "incorrect_answers": [{"text": "<distractor 1>"}, {"text": "<distractor 2>"},
                                                            {"text": "<distractor 3>"}]}),
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(MAT / "prompts" / "generation_prompts.md").write_text(
    "# Generation prompts (question generator)\n\n"
    "TO BE ADDED BY THE AUTHORS from the question generator's database:\n\n"
    "- the system prompt and output schema sent to Gemini 3.1 Pro (model identifier as used through OpenRouter),\n"
    "- the per-exam instructions (Exams 1 and 2: \"exceptionally difficult\", range of Bloom levels, a distractor "
    "longer than the key; Exam 3: \"medium difficult\", push \"for higher levels such as evaluating, analyzing and "
    "applying\"),\n- generation dates and sampling settings.\n", encoding="utf-8")

m = pd.read_csv(HERE / "out" / "items_merged.csv")


def themes_of(e: str, n: int, who: str) -> str:
    return ";".join(t for t in CODES.get((e, int(n), who), []) if t in THEMES)


items = pd.DataFrame({
    "exam": m.exam, "item": m["no"], "n_students": m.n, "facility": m.p.round(4),
    "item_rest_r": m.disc.round(4),
    "gemini_self_bloom": m.llm_bloom, "gemini_self_difficulty": m.llm_diff,
    "rater_A_bloom": m.A_bloom, "rater_B_bloom": m.B_bloom,
    "fable_instr_bloom": m.F_bloom_i, "fable_bare_bloom": m.F_bloom_b,
    "fable_instr_difficulty": m.F_diff_i, "fable_bare_difficulty": m.F_diff_b,
    "gemini_cold_instr_bloom": m.G_bloom_i, "gemini_cold_bare_bloom": m.G_bloom_b,
    "gemini_cold_instr_difficulty": m.G_diff_i, "gemini_cold_bare_difficulty": m.G_diff_b,
    "rater_B_comment_themes": [themes_of(e, n, "B") for e, n in zip(m.exam_orig, m["no"])],
    "rater_A_comment_themes": [themes_of(e, n, "A") for e, n in zip(m.exam_orig, m["no"])],
})
items["rater_B_flagged_distractors"] = items.rater_B_comment_themes.str.contains("distractors").astype(int)
items.to_csv(MAT / "data" / "items.csv", index=False)

cb = ["# Codebook for the rater comments (Table 5)\n\n",
      "Unit: one free-text comment. A comment can carry several themes or none. Code what the comment says, not "
      "what the coder thinks of the item. The first coding was made by one author with access to the labels and "
      "item statistics (Section 3.5). A second author, one of the raters, checked every code against its comment and "
      "item (question, correct answer and distractors shown) and disagreed with none.\n"]
for t, (title, label, rule) in THEMES.items():
    cb.append(f"\n## {title} (`{t}`)\n\nTable 5: {label}.\n\n{rule}\n\nExamples:\n\n")
    cb += [f"- \"{x}\"\n" for x in examples(t)]
(MAT / "codebook.md").write_text("".join(cb), encoding="utf-8")

for f in ["analyze.py", "camera_ready.py", "camera_ready_glmm.R", "llm_classify.py", "comment_codes.py",
          "make_materials.py", "coding_check.py", "pyproject.toml", "uv.lock"]:
    shutil.copy(HERE / f, MAT / "code" / f)
(MAT / ".gitignore").write_text(".DS_Store\n__pycache__/\n.env\n", encoding="utf-8")

(MAT / "README.md").write_text(f"""# Supplementary material

Einarsson, Sverrisdóttir, Einarsson, Guðmundsson & Lund (2026). Self-assigned Bloom levels and difficulty labels of
LLM-generated exam items: agreement with human raters, cold model ratings and student performance. CELDA 2026.

Repository: {REPO_URL}

## Contents

- `prompts/rater_instructions.txt`: the written instructions shown to the two human raters (also the system prompt
  of the *instructed* Bloom condition).
- `prompts/classification_prompts.json`: system prompts of the four classification conditions, the format of the
  user message, model identifiers, access dates and sampling settings.
- `prompts/generation_prompts.md`: prompts of the question generator (Gemini 3.1 Pro).
- `data/items.csv`: one row per item ({len(items)} items): exam, item number, number of students, facility
  (proportion correct), item-rest correlation, every Bloom and difficulty label used in the paper, and the comment
  themes. Item texts and rater comments are not included, apart from the short example comments in the codebook, so that the items can be reused in teaching.
- `codebook.md`: the themes of Table 5 with coding rules and examples.
- `code/`: analysis code. `analyze.py` produces the main tables and figures, `camera_ready.py` and
  `camera_ready_glmm.R` the analyses added for the camera-ready version (collapsed agreement, per-exam estimates,
  student-and-item bootstrap, mixed logistic models), `llm_classify.py` the model classifications,
  `make_materials.py` this package and `coding_check.py` the check of the comment coding.
  The scripts read the annotation export, the student-level response files and a file mapping the raters' logins
  to A and B, none of which are public. Every item-level quantity they produce is in `data/items.csv`.

## Models

| Role | Model | Identifier | Access |
|---|---|---|---|
| Generation and self-labels | Gemini 3.1 Pro | see `prompts/generation_prompts.md` | spring 2026 |
| Cold classification | Claude Fable 5 | `anthropic/claude-fable-5` | OpenRouter, August 2026 |
| Cold classification | Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | OpenRouter, August 2026 |

All classification calls used the providers' default sampling settings and a strict JSON schema that allowed one
label and a one-sentence rationale.
""", encoding="utf-8")

# ================================================================ 2. coding review sheet
random.seed(20260925)
order = list(range(len(comments)))
random.shuffle(order)

WRAP = Alignment(wrap_text=True, vertical="top")
BOLD = Font(bold=True)
YES_FILL = PatternFill("solid", fgColor="E2EFDA")        # coded as present
HEAD_FILL = PatternFill("solid", fgColor="F2F2F2")
DECIDE_FILL = PatternFill("solid", fgColor="FFF2CC")     # cells the reviewer fills in
THIN = Side(style="thin", color="BFBFBF")

wb = Workbook()
ws = wb.active
ws.title = "Instructions"
lines = [
    ("Review of the comment coding", Font(bold=True, size=13)),
    ("", None),
    ("Each row of the sheet 'Comments' is one free-text comment that a rater wrote while assigning a Bloom level to "
     "a multiple-choice item. The item is shown next to the comment: the question, the correct answer (option A) "
     "and the three distractors (B to D), as the rater saw them.", None),
    ("The first coder assigned each comment to zero or more of seven themes. For every theme, the column "
     "'First coding' shows Yes (the theme is present in the comment) or No (it is not).", None),
    ("Your task: for every theme, choose Agree or Disagree in the yellow column next to it.", BOLD),
    ("- Disagree with a Yes means the comment does not express that theme.", None),
    ("- Disagree with a No means the comment does express that theme and the first coder missed it.", None),
    ("- Judge what the comment says, not what you think of the item. The item is shown only so that remarks such "
     "as 'B-D nonsensical' can be understood.", None),
    ("- Use the Note column to explain a disagreement or to flag a comment you are unsure about.", None),
    ("Please do not consult the paper's Table 5, the item labels or the student results while reviewing. "
     "All 700 decisions (100 comments x 7 themes) are needed.", None),
    ("", None),
    ("Themes", BOLD),
]
for t, (title, _, rule) in THEMES.items():
    lines.append((f"{title}: {rule}", None))
    lines.append(("   Examples: " + " | ".join(f'"{x}"' for x in examples(t)), Font(italic=True, color="595959")))
for i, (line, font) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=line)
    c.alignment = WRAP
    if font:
        c.font = font
ws.column_dimensions["A"].width = 150

cs = wb.create_sheet("Comments")
head = ["id", "Item as the rater saw it: question, correct answer (A, in bold) and distractors (B to D)",
        "Rater's comment"]
for k, (t, (title, _, rule)) in enumerate(THEMES.items(), 1):
    # title and role in bold, the coding rule in plain text; coding_check.py finds columns by "T{k}." and role
    head += [CellRichText(TextBlock(InlineFont(b=True), f"T{k}. {title}\n\n"), f"{rule}\n\n",
                          TextBlock(InlineFont(b=True), "First coding")),
             CellRichText(TextBlock(InlineFont(b=True), f"T{k}. {title}\n\nYour decision:\nAgree / Disagree"))]
head.append("Note")
cs.append(head)
for c in cs[1]:
    if not isinstance(c.value, CellRichText):
        c.font = BOLD
    c.alignment = WRAP
    c.fill = HEAD_FILL
    c.border = Border(bottom=THIN)
cs.row_dimensions[1].height = 195

key = []
for rid, j in enumerate(order, 1):
    c = comments[j]
    item = CellRichText(
        f"{c['stem']}\n\n",
        TextBlock(InlineFont(b=True), f"A (correct): {c['key']}"),
        "".join(f"\n{'BCD'[i]}: {d}" for i, d in enumerate(c["distractors"])))
    row = [rid, item, c["comment"]]
    for t in THEMES:
        row += ["Yes" if coded(c, t) else "No", None]
    row.append(None)
    cs.append(row)
    key.append({"id": rid, "exam_orig": c["exam_orig"], "no": c["no"], "rater": c["rater"]})

n_rows = len(comments) + 1
widths = [5, 55, 32] + [26, 12] * len(THEMES) + [30]
for i, w in enumerate(widths, 1):
    cs.column_dimensions[get_column_letter(i)].width = w
for row in cs.iter_rows(min_row=2, max_row=n_rows):
    for cell in row:
        cell.alignment = WRAP
        cell.border = Border(bottom=THIN)
    row[2].font = BOLD  # the comment itself
    for k in range(len(THEMES)):
        first, decide = row[3 + 2 * k], row[4 + 2 * k]
        first.alignment = Alignment(horizontal="center", vertical="top")
        if first.value == "Yes":
            first.fill = YES_FILL
        decide.fill = DECIDE_FILL
        decide.alignment = Alignment(horizontal="center", vertical="top")
dv = DataValidation(type="list", formula1='"Agree,Disagree"', allow_blank=True,
                    error="Choose Agree or Disagree", showErrorMessage=True)
cs.add_data_validation(dv)
for k in range(len(THEMES)):
    col = get_column_letter(5 + 2 * k)
    dv.add(f"{col}2:{col}{n_rows}")
    cs.conditional_formatting.add(f"{col}2:{col}{n_rows}", CellIsRule(
        operator="equal", formula=['"Disagree"'], fill=PatternFill("solid", fgColor="F8CBAD")))
cs.freeze_panes = "D2"  # item and comment stay visible while scrolling through the themes

out = ROOT / "feedback" / "coding_check"
out.mkdir(parents=True, exist_ok=True)
wb.save(out / "comment_coding_sheet.xlsx")
pd.DataFrame(key).to_csv(HERE / "data" / "coding_check_key.csv", index=False)
print(f"materials -> {MAT}; coding sheet -> {out / 'comment_coding_sheet.xlsx'} ({len(comments)} comments)")
