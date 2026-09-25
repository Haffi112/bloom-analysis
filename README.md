# Supplementary material

Einarsson, Sverrisdóttir, Einarsson, Guðmundsson & Lund (2026). Self-assigned Bloom levels and difficulty labels of
LLM-generated exam items: agreement with human raters, cold model ratings and student performance. CELDA 2026.

Repository: https://github.com/Haffi112/bloom-analysis

## Contents

- `prompts/rater_instructions.txt`: the written instructions shown to the two human raters (also the system prompt
  of the *instructed* Bloom condition).
- `prompts/classification_prompts.json`: system prompts of the four classification conditions, the format of the
  user message, model identifiers, access dates and sampling settings.
- `prompts/generation_prompts.md`: prompts of the question generator (Gemini 3.1 Pro).
- `data/items.csv`: one row per item (188 items): exam, item number, number of students, facility
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
