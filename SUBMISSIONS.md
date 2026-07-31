# Submission log

Team `ZIMBRA` on Zindi. Same LoRA adapter (`qwen25-1.5b-aircd-lora`, pulled from the HF Hub, never
retrained across these three runs) for all three submissions below — only inference-side bugs changed.

| # | Public score | Comment | What actually changed |
|---|---|---|---|
| 1 | 0.129343629 | first baseline submission | `GEN_MAX_NEW_TOKENS=300` — 8.5% of rows had no `\boxed{}` (truncated mid-reasoning) |
| 2 | 0.14092664 | raised token budget 300→450 | Blank-answer rate dropped 8.5%→5.9%. `~+0.012` score gain, roughly matching the ~2.6% of rows recovered from truncation |
| 3 | 0.138030888 | deterministic per-question seeding fix | Fixed a `hash()`-randomization bug (Python randomizes str hashing per-process, so the "fixed" `SEED` wasn't actually reproducible run-to-run). Score is flat vs. #2 (within noise) — confirms this was a reproducibility fix, not an accuracy lever |

## Reading

Scores 1→3 are all clustered around 0.13-0.14, well above the `0.027` benchmark floor but far below the
leaderboard's top tier (rank 5 ≈ 0.985 as of 2026-07-31). All three fixes so far (token budget, scoring-harness
bug, seed determinism) were **infrastructure** fixes — they made measurement/generation correct, not the model
smarter. None moved the score by more than noise.

The strongest signal we have that the *model* itself is the bottleneck: within submission #3, resampling the
same question 4 times (as the Pass@1 metric requires) landed on the same `\boxed{N}` answer in all 4 samples
for only 56/863 questions, and on average only 2.13/4 samples agreed at all. If the model had reliably learned
the underlying decision rule, resampling the same inputs should converge far more often than that.

## Update, 2026-07-31: the real explanation is likely a training bug, not just thin data

A first training attempt for round 2 (`Cassava_AIRCD_finetune_v2.ipynb`) showed `[38/38 steps]` where roughly
1,800 were expected. Measuring actual token counts with the real Qwen2.5 tokenizer showed why: the embedded
drive-test/engineering tables alone run ~2000-2900 tokens per example (median ~2400), against
`SFTConfig(max_length=2048)` in both notebooks. Essentially every example was being truncated down to a
prompt-only fragment with the completion entirely cut off, then dropped as "fully masked" before training
ever saw it — an earlier char-count-based estimate that these tables were "well within context window" was
wrong. **This affects round 1 too** (confirmed 200/200 sampled round-1 examples also exceed 2048 tokens), so
the adapter behind all three submissions above was very likely trained on a tiny sliver of `train.csv`, not
the full 2,400 rows. That's a materially better explanation for the low self-consistency and the score
plateau than "insufficient data diversity" — fixed to `max_length=3072` (0% truncation on the real
14,400-example round-2 dataset) in both notebooks; neither has been retrained with the fix yet.

Also known and *not* yet addressed by any submission: local validation (`validation_questions.csv`) only
covers the "full canonical 8-option" slice of the distribution — the harder ~21% of `test.csv` (letter/subset
shuffled options, the disjoint LTE 9-category domain, generic MCQs) has never been locally measurable.
