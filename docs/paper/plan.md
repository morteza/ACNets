# Research plan: the pretrain–finetune multiverse

**Phases A-E are all executed; this is now a record, not a plan.** It is kept for the
status log in section 5, which is the compressed account of what was run and what it found,
and because scripts and the manuscript cite its phase labels (`plan.md D1`, `D4`). It also
doubles as the pilot protocol for the OmniData/OmniModel proposal, which is the other reason
not to fold it into the manuscript.

Read section 5 first. Sections 1-4 are the original design and are preserved for provenance:
where execution diverged from them, the divergence is noted in place.

## 0. Where things stand

Two nulls have been run on a 24-specification subset (Dosenbach 2010 atlas; six statistics;
network and random-network aggregation; SVM and XGBoost; 4-fold × 5-repeat CV, identical for
observed and null):

| analysis | result | verdict |
|---|---|---|
| Decoding, family-wise max-statistic null (B = 200) | best 0.725 vs critical value 0.732; p = 0.114; 0/24 specifications significant | **null** |
| Attribution agreement vs random rankings (RBO) | excess +0.376, p = 0.017 | **survives** |
| Same, on size-matched random networks (control) | excess +0.131, p = 0.162 | correctly fails |

The dissociation — a reproducible attribution without reliable decodability — is currently the
paper's central claim (manuscript §3.4.3, title, abstract). The programme below is the attempt
to *earn back* a decoding claim without giving up the correction that killed the first one.

Written and tested but not yet wired into any analysis: `acnets/analysis/brain_behavior.py`,
`acnets/multiverse/variance.py`. Wired this session: `run_agreement.py`, `run_permutation.py`.
Deep-learning stack rewritten in plain PyTorch with tests (`acnets/deep/`), currently outside
the multiverse.

## 1. The design

**Claim 1 of the proposal, piloted: resting activity as a shared alignment subspace.**
MPI-LEMON provides 220 subjects × 652 timepoints of resting fMRI already parcellated with the
same two atlases as Julia2018 (`data/LEMON/timeseries_{aal,dosenbach2010}.nc`). Models are
pretrained on LEMON without ever seeing a label, then fine-tuned inside Julia2018's
cross-validation folds. Because the datasets share no subjects and pretraining is
self-supervised, the transfer is leakage-free by construction; because rest is the most
co-recorded condition in neuroimaging, the same recipe scales beyond this pair of datasets —
which is exactly the proposal's Claim 1, and these results become its preliminary data.

**Claim 2, piloted: behaviour grounds the representation.** Behavioural performance from the
attentional task (`preprocessing/beh.py`, working since the pandas fix) enters twice: as the
association target of the brain–behaviour analysis (§3.3 of the manuscript), and as an
optional-inclusion input modality of the fused model, so the learned latent is anchored to an
interpretable cognitive quantity rather than to sensor statistics alone.

**The multiverse stays the arbiter.** Every new model is a point in the same specification
space, subject to the same family-wise null and the same attribution-agreement analysis. Two
statistical registers, decided in advance:

1. **Pre-specified confirmatory primary.** One pretrained fine-tuned model, fully specified in
   `docs/paper/prereg_primary.md` and `SpecificationRegistry.primary_confirmatory()` *before*
   any Julia2018 fine-tuning accuracy is computed. The declaration must be committed (user's
   git) before the run; the ordering is what licenses an uncorrected test.
2. **The corrected multiverse.** Everything, the primary included, also runs under the
   family-wise max-statistic null. Adding specifications widens this null; that cost is
   accepted and stated.

**Transfer for classic ML, same logic.** A `pretraining ∈ {none, lemon}` axis on
`Specification`: the tangent statistic's reference point estimated on LEMON instead of the
training fold; LEMON-fitted standardisation/PCA in front of SVM/XGBoost. Cheap, runs
everywhere the DL runs, and isolates *how much of any gain is the external data rather than
the deep model* — without it, a DL improvement would be uninterpretable.

**Fusion where the forks are.** The multiverse treats connectivity statistics as competing
paths; the multi-view model treats them as complementary views of one subject (shared-latent
encoder over {correlation, partial correlation, tangent, chatterjee} + structural volumes +
behaviour, with modality dropout so any subset suffices at inference). One phenomenon, two
readings — competition for inference, fusion for representation.

**Attribution parity.** DL specifications produce attribution vectors via
`shap.GradientExplainer`/`DeepExplainer` (shap is already a dependency), feeding the *same*
agreement machinery as the Haufe patterns and TreeExplainer values. The paper's surviving
result extends to the new models or measurably fails to.

## 2. Phases

### A. Fill what is answerable now — days, no new ML

| task | source → destination | manuscript |
|---|---|---|
| A1 demographics: sex/age × group | `data/julia2018/participants.tsv` (verified: age, sex present; 16/16) → `outputs/demographics.json` | §6.1; Appendix E item 9 |
| A2 brain–behaviour | `scripts/run_brain_behavior.py`: `beh.py` trials → accuracy/RT/IES per subject → join CON-SMN, FPN-SMN partial-corr features → `brain_behavior.pooled_and_within` | §3.3; item 6b |
| A3 variance decomposition | wire `variance_decomposition` (balanced subgrid) into `run_agreement.py` → `outputs/agreement/variance.json` | §3.4.2 |
| A4 effect sizes | d + bootstrap CI for CON-SMN, FPN-SMN, CON-FPN (descriptive, no p-values — the features were selected on these data) | §3.1 |
| A5 carryover check | early vs late 62-timepoint halves, primary spec | §4.5 rest-after-task TODO |

### B. Feature cache and the classic transfer axis

- **B1 fold-feature cache.** Features never see `y`, so per-(specification-minus-classifier,
  fold-split) matrices are identical across label permutations. Cache them content-addressed
  under `outputs/cache/`; the permutation null then pays only for classifier refits.
  Acceptance: cached and uncached results bit-identical; null wall-clock ~10× down; enables
  B = 5000 and DL-in-null.
- **B2 `pretraining` axis** on `Specification` + registry validity rules. `spec_id` hashes
  change; stores are regenerable, noted in the migration.
- **B3 LEMON-fitted transforms** for classic specs (tangent reference; scaler/PCA).
  Acceptance: a leakage test proving fold results are invariant to permuting Julia2018 labels
  at transform-fit time.

### C. Deep pretrain→finetune

- **C1 windowing.** LEMON 652 tp → five non-overlapping 124-tp windows (stride option),
  matching Julia2018's length; ~1,100 pretraining samples. Per-window z-scoring is the
  TR/site-mismatch mitigation.
- **C2 pretraining objectives.** (a) Reconstruction: `Seq2SeqAE` on timeseries windows,
  `MaskedVAE` on connectivity — both exist, plain PyTorch, tested; checkpoints via
  `checkpoint_id`. (b) *Rest-vs-task contrastive alignment* — the proposal's mechanism — is
  **deferred**: the DVC manifest holds task-attention BOLD for exactly one subject
  (`participants.tsv` says `preprocessed_tfmri = FALSE` for nearly all), so the objective
  needs task fMRI to be tracked and fmriprep'd first (cluster job). Recorded here so the
  pilot's scope is honest.
- **C3 `src/acnets/deep/multiview.py`.** Shared-latent fusion over the statistic views +
  structure + behaviour, modality dropout, reusing `mvae.py`/`multihead_model.py` blocks.
- **C4 structural modality.** Fetch the fmriprep `anat` subset (extend
  `scripts/fetch_test_fixtures.py --what anat`; `aparc+aseg` segmentations verified in the
  manifest); per-region volumes = label voxel counts × voxel volume
  (`src/acnets/analysis/structural.py`). No new toolchain for Julia2018.
- **C5 `TorchClassifier`** sklearn-compatible wrapper; `runner._make_classifier` gains
  `mlp`, `mlp-pretrained`, `multiview-pretrained`; torch stays an optional import with a
  clean error.
- **C6 DL attribution** branch in `_fold_attribution` (GradientExplainer/DeepExplainer).
- **C7 the prereg gate.** Write `prereg_primary.md` + `primary_confirmatory()` with exact
  architecture, hyperparameters, CV budget; **stop; hand to the user to commit; only then
  compute the primary's accuracy.** Tuning happens on LEMON and on Julia2018 *training folds
  of non-primary specs* only.
- **C8 LEMON noise ceiling** (sex at n = 32): blocked — the cached `.nc` carries no
  metadata; needs a cluster/MPI fetch. Flagged, not blocking.

### D. gong2015 structural arm

`data/gong2015/`: 24 AVG-expert + 23 AVG-nonexpert, raw GE DICOM, T1 (`Ax 3D FSPGR`, 1 mm,
protocol `Brain_gong_dota`) — the structural arm of Gong et al. (2015), the closest
precedent. **D1 provenance: confirmed by that paper's authors.** D2 DICOM→BIDS via
`acnets.preprocessing.utils.dcm2bids` (dcm2niix; nipype path exists). D3 segmentation → per-atlas
grey-matter volumes. D4 expert/non-expert classification through the *same* multiverse + null
machinery → manuscript §3.6. No pooling with Julia2018 inside the arm: disjoint subjects,
different scanner.

> **How this differs from what was planned.** Three things changed once it ran. (i) D3's
> containerised tool (FastSurfer/SynthSeg) was replaced by a pure-Python ANTs pipeline,
> `scripts/segment_t1.py`, used identically for both datasets — no FreeSurfer install.
> (ii) "Replication arm" was the wrong name and is retired: the arm returns a **null**, in
> the sample, in six external-validation transfers, and pooled at n = 79. It replicates the
> *absence* of decodability, not a positive finding. (iii) The agreement half of D4 is
> **withdrawn**, not reported — see the row for D in §5.

> **Compute note from Phase D**: exact TreeExplainer attribution on ~166-feature XGBoost
> models costs tens of seconds per fold — the gong2015 observed grid spent ~1.5 CPU-hours
> mostly there. (Moot for that arm now: the rerun drops attribution entirely and finishes in
> minutes. The lesson still applies to E1.) For E1's region-scale specifications (12,880 features), exact SHAP is not
> viable per fold: use `TreeExplainer(..., approximate=True)`, a background subsample, or
> attribute on a per-repeat rather than per-fold basis. The null phases are unaffected
> (attribution is skipped there by design).

### E. One corrected inference, then the manuscript

- **E1** rerun agreement + permutation over the final universe; B ≥ 1000 locally
  (target 5000 with the B1 cache; cluster if needed); identical CV budget for observed and
  null; primary reported both uncorrected (licensed by C7) and within-family.
- **E2** fill the manuscript's `[[TODO]]` markers per the mapping (§3 below); write the 6c
  resolution; extend Methods §6 with pretraining, the transfer leakage argument, DL
  attribution, structural extraction, and the gong2015 arm.
- **E3** citation verifications from the References table (Appendix E item 2) — flagged for
  the authors; nothing gets invented.

## 3. What each phase fills in the manuscript

| phase | manuscript location |
|---|---|
| A1 | §6.1 sex/age table; closes item 9 |
| A2 | §3.3 `[[TODO: did/did not]]` and abstract counterpart; closes 6b |
| A3 | §3.4.2 variance table |
| A4 | §3.1 d + CI placeholders |
| A5 | §4.5 carryover `[[TODO: result]]` |
| C7 | §2.4 — the designation is **withdrawn**; see the note below |
| D4 | §3.6 structural arm — decoding + external validation only; attribution withdrawn |
| E1 | §3.4.1 final numbers; §3.4.3 final agreement; abstract `[[TODO]]`s |

## 4. Compute and risks

- **Device policy**: CPU by default (`resolve_device`), MPS opt-in; models are small enough
  that determinism is worth more than speed.
- **Budgets**: pretraining on ~1,100 windows — minutes on CPU. Fine-tuning inside
  4×5 CV — seconds per fold. The permutation null dominates; with B1 it is an overnight
  local job at B = 5000, else cluster.
- **Honesty ledger**: DL may well not beat the SVM at n = 32 — the primary is protected
  either way and the result is reported regardless. Widening the universe raises the
  family-wise bar; accepted. transfer_entropy still arm64-blocked (six of seven statistics).
  Rest-vs-task deferred (C2). gong2015 provenance unconfirmed (D1). LEMON metadata absent
  (C8). Agreement is never conditioned on accuracy.
- **Release**: model weights and outputs go to DVC; scripts stay reproducible
  (`uv run python scripts/...`), per the proposal's open-source commitment.

## Status log

Kept short; details in the phase sections above and in `outputs/`.

| item | status |
|---|---|
| Deliverable 1, A1-A5, B1-B3 | **done** — see manuscript §3.1/3.3/3.4.2/4.5/6.1 and `outputs/` |
| C1 windowing, C2a reconstruction pretraining | **done** — `windowed`, `scripts/run_pretrain.py`; TimepointAE selected (held-out rel. err. 0.099), Seq2SeqAE rejected (1.004) |
| C2b rest-vs-task contrastive | deferred — task BOLD tracked for 1 of 32 subjects; needs cluster fmriprep |
| C5 `mlp` classifier, C6 DL attribution | **done** — `TorchMLPClassifier` + gradient×input in the runner; joins the grid in E1 |
| C7 prereg | **written and executed** — `prereg_primary.md`, `PRIMARY_CONFIRMATORY`; observed **0.477**, null 95th pct 0.514, **p = 0.0999 at B = 1000: null result, reported as declared**; user commit pending |
| C4 structural modality | **done** — `--what anat` fetch, `structural.py`; 31 subjects × 267 stable regions, `outputs/structural_volumes.csv` (1 subject lacks native anat; modality dropout covers it) |
| C3 multiview | **model + tests done** (`multiview.py`); exploratory driver lands with E1 |
| C8 LEMON noise ceiling | blocked on LEMON metadata (cluster/MPI fetch) |
| D gong2015 | **done, then superseded** — 47/47 converted and segmented. The first pass reported a decoding null plus strong attribution agreement and concluded *the dissociation replicates across modality*. Two defects invalidated the attribution half: `ants.get_mask` left skull in the volumes (mask 2127 cm³, GM 785) and the Dosenbach sphere atlas gives near-arbitrary cross-dataset volumes (profile r = −0.41 vs +0.87 for AAL). Reprocessed with corrected brain extraction and volumetric atlases: decoding still null and slightly stronger (0.574), external validation 6/6 at chance, pooled n = 79 null, **attribution withdrawn**. See `decodability_ceiling.md` §18–21 and §23 |
| E1 grid correction | **done** — the registry was emitting 5 degenerate cells (`xgboost + lemon` on the 5 stateless statistics is bit-identical to its `none` twin; trees are scale-invariant and only `tangent` carries measure state). Grid is 49, not 54. Fixed in `Specification.is_valid` + test. Agreement, selection frequency, variance and the threshold curve all recomputed; the family-wise p is provably unaffected (max over a set with duplicates = max over the distinct set) |
| E1 agreement, final | **done** — 49 specs, accuracy 0.385–0.660 (median 0.519). Network arm: RBO 0.634 [0.596, 0.673] vs 0.423 floor, **excess +0.366, p = 0.019**; control +0.154, p = 0.135 (correctly fails); Kendall +0.085, p = 0.30 (**fails** — the claim is the top of the ranking, not the ranking). Variance: no-pretraining subset aggregation 44.4% / statistic 6.6% / classifier 3.3%; pretraining subgrid (scale-sensitive classifiers, fully crossed) statistic 75.1% / classifier 5.2% / **pretraining 1.6%** |
| E1 family-wise null | **done** — B = 1000 over the 49-spec grid at 4×25 CV. Observed max 0.660 (`pcorr/network/xgboost`); null max median 0.625, 95th pct **0.739**; **p = 0.291, 0 of 49 significant**. The observed maximum sits beside the null's centre, not in its tail: 29% of permutations beat it. Trajectory across the three runs (24 → 36 → 49 specs): p = 0.114 → 0.244 → 0.291, i.e. widening the grid widened the null exactly as predicted |
| E2 manuscript | **done** for everything the data support — §2.3, §3.1, §3.2, §3.4.1–3.4.4, §3.5, §4.4, §4.5, Methods §6.5–6.11, abstract, how-to-read box. Remaining TODOs are all outside the analyses: title wording, repository URL, reference-list completion, plus two named-and-blocked analyses (LEMON noise ceiling, C8; cerebellar control, needs DiFuMo-256) |
| **New: motion is a real confound** | `run_motion_control.py` + `run_motion_coupling.py`. Gamers move more (FD SD d = +0.89); motion alone decodes at **0.608**, comparable to the best connectivity spec; motion correlates with CON-SMN at r ≈ 0.5 across statistics. CON-SMN group effect **survives** partialling motion out in 4 of 6 statistics; **FPN-SMN does not**. The old §3.5 asserted the opposite and had never been tested. Converges with the independent early/late carryover check, which separates the two features the same way |
| **New: `MultiScaleClassifier` H1 block is zero** | `ExtractH1Features` returns the temporal mean of z-scored time courses (max abs 0.033 vs SD 0.996). §3.2 rebuilt as `scripts/run_multiscale.py` with all blocks from one region-level matrix. Result: every block at chance |
| E2 figures | **done** — `acnets.plotting.specification_curve` + `agreement_figures`, rendered by `scripts/make_figures.py`; the manuscript referenced a `plotting/specification_curve.py` that had never existed |
| **New: aggregation order is the biggest effect measured** | the same 15 between-network quantities give 0.45–0.54 computed as connect-then-average and **0.61–0.67** as average-then-connect. Larger than classifier family, pretraining, or the choice of statistic — and it is not an axis in the registry. Appendix E item 14 |
