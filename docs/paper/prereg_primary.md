# Pre-specified confirmatory primary analysis

> **WITHDRAWN AS A PREREGISTRATION.** This document specifies one model completely and was
> written before that model's accuracy on Julia2018 was computed, which is all it ever did.
> It was written *partway through* the analysis of these same participants, as part of an
> effort to find a configuration that decoded well — so it constrained nothing, informed no
> decision, and licenses no uncorrected test. Manuscript section 2.4 says this in full. The
> model is reported as one more exploratory cell (0.477, p = 0.0999) inside the family-wise
> null. Kept as a record of what was specified and when, not as a preregistration.


This document freezes the primary analysis of the LEMON-pretrained model on the Julia2018
AVGP/NVGP contrast **before any part of that analysis has been executed**. Its purpose is to
license one uncorrected significance test (manuscript §2.4; `plan.md` §1): the licence holds
only if nothing below changes after results are seen, and only if the result is reported
whichever way it comes out. The repository owner commits this file before the run; the commit
timestamp is the public form of the freeze.

## Disclosure: what has been seen before this declaration

Everything already computed on Julia2018 at the time of writing, so the reader can judge
independence for themselves:

- The 36-specification connectivity-space grid (SVM/XGBoost over six statistics, two
  aggregations, with and without LEMON-fitted transforms): accuracies, family-wise nulls
  (best 0.660, p = 0.244), attribution agreement, variance decomposition
  (`outputs/agreement/`, `outputs/permutation/`).
- Descriptive group effect sizes for CON-SMN / FPN-SMN / CON-FPN, demographics, the
  brain-behaviour null, the carryover split (`outputs/descriptives.json`,
  `outputs/brain_behavior.json`).

**No timeseries-level model and no torch model of any kind has ever been fitted to Julia2018
labels.** The primary below operates on the timeseries, in a learned latent basis — a feature
space disjoint from everything listed above.

## Architecture selection, on LEMON only

Two candidate encoders were pretrained on MPI-LEMON (220 subjects, no labels available in the
cached data), and selection used held-out reconstruction on 22 subjects split by subject:

| candidate | held-out relative error | verdict |
|---|---|---|
| Seq2SeqAE (LSTM, latent 32) | 1.004 — no better than predicting the mean | rejected |
| TimepointAE (per-timepoint MLP bottleneck, latent 4, hidden 16) | **0.099** | selected |

Records: `outputs/pretrain/lemon_seq2seq_1d6444.json`, `outputs/pretrain/lemon_timepoint_e96664.json`.

## The frozen analysis

**Pretrained encoder.** `checkpoints/lemon_timepoint_e96664/last.pt` — TimepointAE with
`n_channels = 6`, `latent = 4`, `hidden = 16`, pretrained for 50 epochs on 1,980
half-overlapping 124-timepoint windows of network-scale (Dosenbach 2010, six networks)
MPI-LEMON resting timeseries, each window z-scored per channel. Checkpoint id `e96664`;
hyperparameters inside `outputs/pretrain/lemon_timepoint_e96664.json`.

**Model.** `LatentConnectivityClassifier` (`src/acnets/deep/timepoint_ae.py`): the pretrained
encoder, followed by latent summary features (4 latent temporal means + 10 upper-triangle
latent covariances = 14 features), followed by one linear layer to 2 classes. No other
parameters.

**Target data.** Julia2018 `timeseries_dosenbach2010.nc5`, aggregated to network scale with
`TimeseriesAggregator(strategy='network')`, each subject's 124-timepoint run z-scored per
channel (the same operation the LEMON windows received). Labels: AVGP = 0, NVGP = 1 from the
subject prefix. n = 32.

**Cross-validation.** `fixed_splits(n_subjects=32, n_splits=4, n_repeats=25, random_state=0)`
— the identical label-independent partition used by the grid. 100 folds.

**Fine-tuning, per fold, from the checkpoint each time.**
1. Head-only: encoder frozen, 30 epochs, Adam lr 1e-2.
2. Joint: all parameters, 30 epochs, Adam lr 1e-3.
Batch size 8, cross-entropy loss, CPU (`ACNETS_DEVICE` unset), torch seed `1000 + fold`.
No early stopping, no per-fold or per-run tuning of any kind.

**Statistic.** Mean test accuracy over the 100 folds.

**Confirmatory test.** B = 1000 label permutations; for each, the entire fine-tuning
procedure is rerun on the permuted labels with the same splits and seeds; one-sided
p = (1 + #{null mean ≥ observed mean}) / (B + 1). Significance level 0.05. This tests the
primary alone — its uncorrected test is what this document licenses. The same observed
statistic is *additionally* reported inside the family-wise corrected multiverse (plan E1),
where it enjoys no protection.

**Reporting commitment.** The observed mean accuracy, its Wilson interval at n = 32, the
permutation p, and the null distribution are reported in manuscript §3.4.1 regardless of
outcome. If p ≥ 0.05 the primary is reported as a null result and no alternative analysis
inherits its licence.

**Attribution (exploratory, not part of the confirmatory claim).** Haufe-transformed head
weights over the 14 latent features, averaged across folds, plus the latent-to-network
encoder loadings needed to interpret them. These do not share a vocabulary with the
connectivity-space grid and do not enter the agreement analysis.

**Stopping rule.** `scripts/run_primary.py` executes exactly this document, refuses to run if
this file is absent, and writes `outputs/primary/`. It is run once for the observed statistic.
Any deviation discovered later is reported as a deviation, not silently amended here.
