# Decisions pending — Paper 2 reframe

These are blocking questions for Phase 1 (PROTOCOL_V2.md). They come from AUDIT.md §11. Phrased so you can answer on a phone — short replies (a/b/c, "default", or one sentence) are fine. I'll keep working on Phase 2 (cell-coverage enumeration) until I have answers.

Reply by editing `DECISIONS_RESOLVED.md` at the repo root with the question number and your answer. I'll pull every ~10 minutes.

---

## Heads-up before you read the questions

While building the Phase 2 cell-coverage map I found something that may change how you want to answer Q1, Q3, and Q6 — and is itself a major finding worth its own §5b in `VALIDATION_REPORT.md`:

**The 582 "diverged" Burgers runs are not optimization divergences — they are NaN-poisoned input data.**

- All 582 are at one resolution: R=48.
- Every other R (16, 32, 64, 96, 128, 192, 256, 384, 512) shows 0 divergences.
- All 582 have `failure_reason=nan_or_inf`, `nan_detected=True`, `best_epoch=-1` (failed before the first epoch finished).
- The Burgers spectral solver in `src/scaling_operator_learning/tasks/burgers.py:_spectral_solve` produces NaN/Inf in roughly 1–4% of generated samples *only* at R=48. Standalone reproducer: `scripts/repro_r48_data_bug.py`.

**Implications you may want to weigh in on (no answer required, but flag if you disagree with my plan):**

- I will exclude R=48 from primary response-surface heatmaps in Phase 3 and call it out separately as a known data-generator artifact.
- The previous draft's "582 diverged" macro should stay numerically the same but be reattributed to *data-generator failure*, not *training instability*. The reviewer would catch this on a re-read.
- Phase 5 (instability) gets a much smaller "true optimization-divergence" budget than the prior draft implied, but a clearer narrative.
- The R=48 data bug itself could be fixed (different dt, larger dealiasing window, or just dropping R=48 from the grid). I have not patched it — that's a separate decision and would require regenerating data + reruns. Default: leave the bug, document it, exclude R=48 from primary plots.

If you want me to instead try to *fix* the R=48 data-gen bug and rerun those 729 cells, add it as a new question in `DECISIONS_RESOLVED.md`. Otherwise I'll proceed with the "document and exclude" plan.

---

## 1. Strictness of "no law fits in main text"

Two readings of your directive:

- **(a)** Law fits live only in an appendix. Main text never names a "winner" and never mentions candidate law families at all.
- **(b)** Law fits live only in an appendix, BUT main text may use descriptive shorthand like "log-like behavior" or "power-like trend" when describing response-surface regions.

**My default:** (b). OK?

---

## 2. Regime taxonomy basis

I need to commit to operational definitions of regime labels in PROTOCOL_V2. My first-pass sketch is **sensitivity-based**:

- *data-limited:* local |∂E/∂N| large relative to slice max, while |∂E/∂R|, |∂E/∂C| small
- *capacity-limited:* analogous on C axis
- *resolution-sensitive:* analogous on R axis
- *saturated:* all three local sensitivities small in absolute terms
- *unstable:* divergence_rate > 10% OR seed CV > 0.5 (thresholds TBD)

**Question:** are you OK with sensitivity-based definitions, or do you want a different basis (e.g. fitted local slope, IC differences, fitted-exponent magnitude)?

**My default:** sensitivity-based, with thresholds calibrated empirically in Phase 3.

---

## 3. Darcy's role

Darcy was only swept over R ∈ {32, 64, 128, 256} — half of Burgers's R range. This limits cross-resolution conclusions.

- **(a)** Accept Darcy as a secondary case; do not draw cross-resolution conclusions about it.
- **(b)** Re-run Darcy at higher R (cost TBD in Phase 2).
- **(c)** Drop Darcy from main text; move to appendix.

**My default:** (a).

---

## 4. Diffusion's data axis

Diffusion uses only 5 N values vs Burgers/Darcy's 11.

- **(a)** Densify (5 → ~11 N values) — adds runtime, requires reruns.
- **(b)** Keep sparse, write the caveats, note in Limitations.

**My default:** (b).

---

## 5. Seed parity (G8)

Currently 2 data seeds globally; 2–3 train seeds per task. New framing emphasizes seed variability.

- **(a)** Re-run to bring data_seeds to 3 across the board.
- **(b)** Work with what we have (2 data seeds) and report seed variability honestly with its actual sample size.

**My default:** (b). Reruns optional, can be added later if Phase 5 results justify.

---

## 6. Optimizer ablations repositioning

The 657 SGD-cosine + 756 Adam-2x-patience runs were originally a "robustness check" for law fits. In the new framing they fit naturally as evidence of optimization-limited regimes (Phase 5).

**Plan:** reposition them as instability/optimization-limited evidence, not as "robustness checks." Confirm?

**My default:** yes, reposition.

---

## 7. Where deliverables live

`AUDIT.md`, `PROTOCOL_V2.md`, `VALIDATION_REPORT.md`, `RESULTS_*.md`, `CLAIM_AUDIT.md`.

- **(a)** Repo root, next to `ROADMAP.md` and `INDEPENDENT_ANALYSIS_REPORT.md`.
- **(b)** A new `paper2/` or `notes/` directory.

**My default:** (a) repo root.

---

## 8. Paper draft target

- **(a)** Wholesale replacement of `paper/main.tex`.
- **(b)** New file `paper/main_v4.tex`, matching the v1/v2/v3 snapshot pattern. Leave `main.tex` unchanged until v4 is ready, then swap in a single commit.

**My default:** (b).

---

## Summary of defaults (if you reply "all defaults")

1. (b) — descriptive shorthand allowed
2. sensitivity-based, empirically calibrated thresholds
3. (a) — Darcy secondary
4. (b) — keep Diffusion sparse, caveat it
5. (b) — work with 2 data seeds
6. yes — reposition optimizer ablations
7. (a) — repo root
8. (b) — `paper/main_v4.tex`
