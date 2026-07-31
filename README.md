# Cassava AI Root Cause Detective — Hackathon Solution

Solution for the [Cassava AI Root Cause Detective](https://zindi.africa) hackathon (Deep Learning Indaba 2026,
run on Zindi). Closes **2026-08-01**, leaderboard reveal **2026-08-03**.

**Open in Colab:**

- Round 1 (submitted, see [SUBMISSIONS.md](SUBMISSIONS.md)):
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashuza11/cassava-network-anomaly-ai/blob/main/notebooks/Cassava_AIRCD_finetune.ipynb)
- Round 2 (scaled synthetic training data, separate Hub adapter, see [Status](#status-2026-07-31)):
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashuza11/cassava-network-anomaly-ai/blob/main/notebooks/Cassava_AIRCD_finetune_v2.ipynb)

## Status (2026-07-31)

Three submissions so far, all clustered around 0.13-0.14 public score — see [SUBMISSIONS.md](SUBMISSIONS.md)
for the full log and analysis. All three were infrastructure fixes (generation token budget, a local-scoring
bug, a non-deterministic seeding bug), not model-quality changes, and none moved the score meaningfully. The
adapter itself hasn't been retrained since round 1's first pass on `train.csv` reformatted as-is (2,400 rows).
Current best evidence points at that being the actual bottleneck: resampling the same question 4x (as the
Pass@1 metric requires) only agrees on the same answer ~2.1/4 times on average, suggesting the model hasn't
reliably learned the underlying decision rule from that little data. Next step is scaling up synthetic
training data over the same rule set (see [Round 2+](#round-2-not-built-yet)) before retraining.

## Task

Given a 5G drive-test scenario (a natural-language question, a slice of user-plane drive-test telemetry,
and a table of cell engineering parameters — all embedded as text in the prompt), pick the most likely root
cause of a throughput drop from a set of candidate explanations and answer with `\boxed{N}`, where `N` is
that option's position/label **as printed in that specific question** (the option order and even the label
style are randomized per question — see [Data format](#data-format) below).

- **Model constraint:** any open-source LLM ≤4B params (Qwen2.5-1.5B-Instruct suggested, not required).
- **Metric:** Pass@1, averaged over 4 independently generated answers per question.
- **Compliance:** final answers must come from actually running inference through the chosen model —
  no deterministic/threshold rule-system as the live predictor, no cross-referencing test questions against
  train/validation to retrieve labels. (Rule-derived reasoning *can* be used to build training data — see
  [Approach](#approach-round-1).) Reproducibility is required: seed everything.
- **H200 access:** Zindi/Cassava only email H200 Jupyter-notebook credentials to participants who already
  have a submission scoring above the leaderboard benchmark — so round 1's job is to clear that bar on a
  free Colab GPU, not to be optimal.

## Data format

| File | Rows | Purpose |
|---|---|---|
| `train.csv` | 2,400 | `ID, question, answer`. Options always listed in fixed canonical order `C1:`.. `C8:`, answer is the bare code (e.g. `"C2"`). |
| `test.csv` | 863 | `ID, question`. What we predict on. See distribution below. |
| `validation_questions.csv` | 864 | `ID, question`. Same shuffled format as test — local Pass@1 harness. |
| `validation_target.csv` | 3,456 | `ID_1`.._4`, `Target` (canonical `C1`-`C8` code) — ground truth for the validation set. |
| `SampleSubmission.csv` | 3,452 | `ID_1`.._4`, `Target`. Shows submission shape; the placeholder Target text is dummy content — only the `\boxed{...}` content is graded. |

`test.csv` is **not** a single uniform format. Breaking down all 863 questions:

- **~79%** — the same 8 canonical root causes as train.csv, but usually a **random subset** (5-8 of the 8)
  with a **randomized label prefix**: plain digits about half the time, otherwise a single random letter +
  digit (e.g. `M1`, `M2`, ... — the letter and even the option count vary per question).
- **~11.6%** — a **disjoint fault domain** never seen in train.csv: LTE-style telemetry (columns like
  `CCE Fail Rate`, `Avg MCS`, `Initial BLER(%)`, `ARFCN`) with 9 different categories labeled `A`-`I`
  (e.g. "Intra-frequency handover threshold too high"). Zero training coverage in round 1 — a known gap.
- **~9.4%** — generic knowledge/math MCQs (4 options, no telecom content) testing whether fine-tuning
  degraded the base model's general knowledge.

`validation_questions.csv` only contains the "full 8 options, plain digits" case — so the local validation
Pass@1 score is an **optimistic upper bound**, not a prediction of the real leaderboard score, since it
doesn't exercise the harder 21% of test.csv.

## Repo layout

```
README.md
.gitignore
data/
├── train.csv                  — 2,400 labeled examples
├── test.csv                   — 863 questions to predict
├── validation_questions.csv   — 864 questions, local Pass@1 harness
├── validation_target.csv      — ground truth for validation_questions.csv
└── SampleSubmission.csv       — submission shape reference
notebooks/
├── Cassava_AIRCD_finetune.ipynb
│               — round 1, submitted. Self-contained Colab notebook: installs deps, builds the SFT
│                 dataset, loads Qwen2.5-1.5B-Instruct in 4-bit + LoRA, trains, validates locally,
│                 writes submission.csv. Pushes/pulls its adapter to `<user>/qwen25-1.5b-aircd-lora`.
└── Cassava_AIRCD_finetune_v2.ipynb
                — round 2. Same pipeline, but section 6 generates several shuffled/subset/prefixed
                  layouts per train.csv row instead of one (~14,400 vs. 2,400 SFT examples), to give
                  the model more exposure to test.csv's actual shuffled-option distribution. Separate
                  Hub adapter (`<user>/qwen25-1.5b-aircd-lora-v2`) — doesn't touch round 1's.
src/
├── data_prep.py — parsing (question tables, options), feature extraction, synthetic CoT generation,
│                  train.csv → test-format conversion. Pure pandas/regex, no GPU needed.
└── scoring.py   — \boxed{} extraction, option-position → canonical-label mapping, Pass@1 scorer.
```

`src/data_prep.py` and `src/scoring.py` are also embedded directly in the notebook (as code cells) so the
notebook is self-contained on Colab; the standalone files exist for local development/testing without a GPU
— run them directly (e.g. `python src/data_prep.py data/train.csv`) to sanity-check parsing/CoT generation
without touching a model.

## Approach (round 1)

Goal: the simplest thing that can plausibly clear the leaderboard benchmark, to unlock H200 access.
Round 2+ backlog is intentionally deferred (see below).

1. **Reformat train.csv to match test.csv's real distribution.** `data_prep.reformat_example` shuffles
   each question's 8 canonical options, keeps a random subset (calibrated to test.csv's observed
   size distribution: 8 most often, sometimes 5-7) with a randomized prefix (digit or LETTER+digit,
   calibrated to test.csv's observed ~48/52 split), and rewrites the "From the following N potential root
   causes" count to match.
2. **Synthesize grounded chain-of-thought training targets.** For each row, `compute_features` derives
   RB average, speed, handover count (serving-PCI transitions), PCI-mod-30 collisions, RSRP/BRSRP
   comparisons, and downtilt/beamwidth geometry directly from that question's own embedded tables.
   `synthesize_reasoning` turns the true label + those features into a short templated explanation ending
   in `\boxed{position}`. This is legitimate SFT-data distillation — the organizer's rule against
   rule-based prediction applies to how *test* answers are produced (`model.generate()`, not a formula),
   not to how training targets are authored.
3. **LoRA SFT** on `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit (QLoRA) on a free Colab T4, using `trl.SFTTrainer`'s
   native `prompt`/`completion` conversational format (automatic completion-only loss masking).
4. **Local validation** against `validation_questions.csv` / `validation_target.csv` before spending a
   Zindi submission.
5. **Test inference**: 4 generations per test question (fixed, index-derived seeds for reproducibility),
   `\boxed{N}` extracted and written to `submission.csv`.

## How to run

1. Open the notebook in Colab (badge above), or manually:
   `https://colab.research.google.com/github/Ashuza11/cassava-network-anomaly-ai/blob/main/notebooks/Cassava_AIRCD_finetune.ipynb`
2. `Runtime > Change runtime type > T4 GPU > Save`.
3. `Runtime > Run all`. The data-loading cell clones this repo automatically (it's public); if that ever
   fails it falls back to a manual upload prompt for the 5 CSVs.
4. Read the printed local validation Pass@1 score before trusting the run further (remember: it's an
   upper-bound estimate, see [Data format](#data-format)).
5. Download `/content/submission.csv` and submit it on Zindi.

## Round 2+ (not built yet)

**In progress / next up**, in priority order (see [SUBMISSIONS.md](SUBMISSIONS.md) for why data scale is
the current top priority):

- **Scale synthetic C1-C8 training data.** `train.csv` gives 2,400 rows, reformatted but not expanded — the
  self-consistency evidence in SUBMISSIONS.md suggests that's too thin for the model to reliably learn the
  8 decision rules already encoded in `compute_features`/`synthesize_reasoning`. Generate many more synthetic
  parameter combinations per rule (not just reformatting the existing rows) across the full valid range for
  each threshold, in `test.csv`-style shuffled/prefixed format, then retrain once.
- Mix in generic MCQ/math examples (same `\boxed{N}` format) to shore up knowledge retention (~9.4% of
  test.csv).
- **LTE 9-category domain (~11.6% of test.csv): known, deliberately deprioritized.** Neither `train.csv` nor
  `validation_questions.csv`/`validation_target.csv` contain a single labeled example of this domain, so
  there's no ground truth to derive or verify synthetic reasoning against — not worth the risk of training on
  guessed labels.

**Later / if time allows:**

- Replace the templated CoT with physics-grounded reasoning for the geometry-heavy categories (C1/C2/C4:
  downtilt/beamwidth → coverage-radius trigonometry) once verified against train.csv's known labels.
- Sweep LoRA rank/epochs, try alternate ≤4B base models, or move to full fine-tuning once H200 access
  is unlocked.
- Revisit generation temperature/sampling strategy once we see how much variance helps vs. hurts Pass@1.
- Mix in the public `netop/TeleLogs` dataset (arXiv:2507.21974) as additional SFT data — same taxonomy,
  likely larger/higher-quality than `train.csv`, schema not yet verified against ours.
- A GRPO reinforcement-learning stage after SFT — the same paper's own ablation shows this is what takes a
  1.5B model from ~20% to ~87% pass@1 on this exact task, far beyond what SFT alone reaches. Out of scope for
  the 2026-08-01 deadline, but the real ceiling if a round 2 happens.
