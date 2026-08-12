# ACNets Analysis Inventory

One entry per analysis: the question it asks, the **approach** (estimator, folds, null), the **input** (the concrete data consumed), the **output** (what is computed and stored), and the **result**. Accuracies are not comparable across entries — they come from different estimators, feature spaces and participant subsets; each is stated with what produced it. "Critical value" always means the 95th percentile of a family-wise max-statistic permutation null: what the *whole search* produces from permuted labels, not a single fit.

Coverage: all 39 scripted analyses, the notebook-based replications, and the in-progress deep-learning line.

## Datasets and conventions

- **Julia2018** — the primary sample: 32 participants (16 action video game players, 16 non-players; labels from the subject-identifier prefix), resting-state fMRI (Dosenbach-2010 parcellation, 160 regions, 124 timepoints at TR = 3 s) plus attentional-task behaviour. Early runs joined behaviour for only 25 participants until the replacement-identifier fix recovered all 32.
- **gong2015** — the structural dataset: 47 participants (24 expert vs 23 non-expert action gamers), T1-weighted grey-matter volume; pools with Julia2018 to n = 79 under an expert-as-player label mapping.
- **MPI-LEMON** — ~220 external resting-state participants, always used label-free: autoencoder pretraining, normative modelling, reliability estimation.
- **ABIDE** — external control cohorts (CPAC pipeline, Dosenbach-160 ROIs) used only for split-half reliability benchmarking.
- **Network scale** — regional timeseries averaged into 6 network timecourses (FPN, CON, SMN, DMN, occipital, cerebellum), then summarized by a connectivity statistic (correlation, partial correlation, covariance, precision, tangent, Chatterjee) into 21 network-pair couplings; "aggregate-then-connect" = connect averaged timecourses, "connect-then-average" = average region-level edges.
- **Behaviour** — per-subject accuracy on standard (non-catch) trials, median correct-trial RT, and inverse efficiency (IES = RT/accuracy), derived from the attention-task trial tables (`trial_type`, `correct`, `response_time`).

## Cohort, behaviour, and design

*Finding: the sample is clean and well matched, but at n = 32 even a known-real behavioural effect fails the corrected test — which calibrates everything below.*

- **Cohort profile** — What are these participants like?
  - approach: Descriptive only — no folds, no p-values; the couplings were classifier-selected on this same sample, so effect sizes are computed on all 32 at once and declared non-inferential.
  - input: Network-aggregated Dosenbach-2010 resting timeseries, plus `participants.tsv` sex and age.
  - output: Per-group demographics (n, sex counts, age mean/SD/range); Cohen's d with 5000-draw bootstrap CIs for the CON–SMN, FPN–SMN and CON–FPN partial-correlation couplings; an early-vs-late-scan carryover check for the two attributed couplings.
  - result: 16/16, all male, age-matched (20.8 ± 2.1 vs 21.1 ± 2.0).

- **Brain–behaviour coupling** — Does control-to-sensorimotor coupling track attentional performance?
  - approach: Pearson associations with 5000-draw bootstrap CIs, always reported pooled *and* within-group, because a pooled correlation can be manufactured entirely by the group difference.
  - input: The CON–SMN and FPN–SMN network partial correlations plus the per-subject accuracy/RT/IES table, joined for all 32 subjects after re-keying the seven replacement participants.
  - output: An 18-row association table (2 couplings × 3 measures × pooled/within-AVGP/within-NVGP) of r with CIs and a per-pair verdict.
  - result: 3 of 18 bootstrap intervals exclude zero, incl. CON–SMN × IES within AVGP (r = −0.378 [−0.682, −0.070]); the same analysis at n = 25 found nothing.

- **Behavioural benchmark decoding** — Can behaviour alone classify group, and what does that calibrate?
  - approach: A positive control, near-circular by recruitment design: nested grid search inside 4-fold × 3-repeat splits, with a 100-permutation family-wise null over the four classifier grids.
  - input: Behaviour only — the three-column accuracy/RT/IES table, n = 32; no imaging.
  - output: Cross-validated accuracy per family (logistic, linear SVM, RBF SVM, LDA), the best cell, and the family-wise critical value with p.
  - result: best 0.615 (logistic) vs critical 0.679, p = 0.119 — **a known real effect (accuracy d ≈ +0.74, IES d ≈ −0.76) also fails at n = 32**.

## Primary decoding and the specification multiverse

*Finding: no analysis path decodes above what searching that many paths produces by chance, and the pre-registered model fails on its own uncorrected terms.*

- **Specification multiverse** — Does any analysis path decode above what searching that many paths produces by chance?
  - approach: Labels are permuted outside the entire pipeline and the whole grid re-run per permutation (B = 1000) on identical fixed splits, with max-statistic correction; features are cached across permutations because their extraction never sees labels.
  - input: Dosenbach-2010 network and random-network aggregations crossed into 49 runnable specifications — 6 connectivity statistics × 2 aggregations × 3 classifiers (SVM, XGBoost, MLP) × {none, LEMON-pretrained} where valid.
  - output: Per-specification accuracies, the observed maximum, critical value and p, the stored null-max distribution, and the specification-curve table behind Figure 2.
  - result: best 0.660, critical 0.739, p = 0.291, 0 of 49; null max median 0.625.

- **Pre-registered primary model** — Does the one model specified in writing before it was fitted decode?
  - approach: The written declaration (`prereg_primary.md`, committed before any fine-tuning accuracy existed) licenses a single uncorrected test: two-phase fine-tuning per fold (30 head-only epochs, then 30 joint) over 4-fold × 25-repeat splits, with a 1000-permutation null on the same splits.
  - input: Network-aggregated, per-channel z-scored Julia2018 timeseries plus the frozen LEMON-pretrained timepoint-autoencoder checkpoint (latent dim 4).
  - output: Observed accuracy with Wilson CI, permutation p, and an exploratory Haufe attribution over the 14 latent-summary features.
  - result: 0.477, null 95th 0.514, p = 0.0999.

- **Transfer-breadth hypotheses** — Which account of transfer breadth does the connectivity structure support?
  - approach: Each hypothesis becomes one feature block, decoded alone and ablated from the combined model on the multiverse's own fixed splits, with test-fold column-shuffling as a second importance measure.
  - input: Region-level partial correlation reduced three ways — H1 weighted degree per FPN/CON/SMN region (~86 features), H2 mean within-network coupling (6), H3 mean between-network coupling (15) — plus the aggregate-then-connect network matrix as reference.
  - output: Per-classifier accuracy and CI for each block alone, all blocks combined, permutation importance drops, and the reference accuracy.
  - result: all blocks at chance (0.45–0.57); aggregate-then-connect beats connect-then-average by 0.13–0.19.

- **Exploratory feature search** — Is there any feature family that separates the groups?
  - approach: An explicit search with no null — per-subject features computed leak-free, all scaling/reduction fitted in-fold; inference is deferred to the multiverse.
  - input: All nine cached atlas timeseries; families = static network correlations, sliding-window dynamic connectivity (mean and SD, 30-TR windows), signed node strength per network, and full region-edge sets under in-fold PCA or univariate selection.
  - output: One row per atlas × family × classifier with accuracy and Wilson CI, and the best configuration flagged as carrying no inferential weight.
  - result: a candidate list, explicitly labelled a search rather than a result.

## Attribution and explainability

*Finding: analysis paths agree on **which** features matter (CON–SMN) beyond the random floor and beyond what feature geometry explains, partly label-dependently — but nothing attributed survives transfer to another dataset or modality.*

- **Cross-specification attribution agreement** — Do different analysis paths agree on which features matter?
  - approach: One attribution vector per specification (Haufe pattern for SVM, TreeSHAP for XGBoost, model attribution for MLP), compared pairwise by rank-biased overlap and Kendall τ against the agreement random rankings of the same 21-edge vocabulary would reach.
  - input: The same 49-cell grid and cached features as the multiverse, restricted for the primary analysis to specifications sharing the network-scale vocabulary.
  - output: Agreement matrices per scale, per-edge selection frequencies, agreement-vs-accuracy-threshold curves, the variance decomposition of accuracy over specification dimensions, and the stability-comparison verdict (feeds Figures 3–5).
  - result: RBO 0.634 vs random-ranking floor 0.423 (+0.366, p = 0.019); Kendall τ at chance.

- **Label-dependence of agreement** — Does that agreement depend on the labels?
  - approach: Destroys only the labels — B = 200 refits of the grid on permuted labels with identical folds, cached features and attribution methods, recomputing the full pairwise agreement matrix per permutation.
  - input: The network-scale grid (6 statistics; SVM by default, optionally all three classifier families) on the Dosenbach-2010 features.
  - output: The null distribution of mean agreement, observed excess and p, and how often each edge leads the consensus under permutation.
  - result: 3 classifier families 0.657 vs 0.562, **p = 0.035**; linear family alone 0.630 vs 0.601, p = 0.29; consensus leader CON–SMN in 62/1000 permutations (p = 0.063).

- **Feature-geometry control** — Could the agreement be feature covariance rather than group signal?
  - approach: No model fitting — four label-free orderings of the same edges are scored by RBO against the observed consensus, calibrated against a 2000-draw random-ranking floor.
  - input: The 21 network-edge features per statistic, ranked by variance, mean magnitude, top-eigenvector loading, and the mean Haufe pattern of random decoders; the consensus ranking from the agreement analysis.
  - output: Per statistic × criterion, the RBO with the consensus and where CON–SMN and FPN–SMN land in the label-free ranking.
  - result: 0.278–0.461 against the consensus, inside the random floor 0.408 [0.282, 0.611] — geometry does not explain it.

- **Cross-dataset attribution transfer** — Do regions found by XAI in one dataset predict in the other?
  - approach: The Haufe pattern of a shrinkage LDA fitted on the whole source sample ranks parcels; the top-k are re-scored in the target by LOOCV against 1000 random same-size parcel sets — the only baseline that tests localisation rather than dimensionality.
  - input: Schaefer-200 grey-matter volume tables (volume per parcel, normalised by total grey matter) for gong2015 (n = 47) and Julia2018 (n = 32), restricted to shared parcels; both directions.
  - output: Per direction, target accuracy of the discovered parcels vs the all-parcel and random-set references, p, and the network composition of the transferred set.
  - result: does not transfer.

- **Cross-modal network identity** — Does a network identity found in one modality survive into the other?
  - approach: Only the identity of the top-3 attributed networks crosses the gap — no features, no weights; the restricted target is scored by LOOCV shrinkage LDA against the exact null of all C(7,3) = 35 network triples.
  - input: gong2015 Schaefer-200 volumes (structural side) and Julia2018 timeseries re-aggregated to the selected networks with precision connectivity (functional side), linked by a parcel-to-network crosswalk.
  - output: Per direction, the discovered triple, restricted-target accuracy, its rank among all triples, and exact p.
  - result: gong2015 structure → Julia2018 function 0.562 (rank 10/20, p = 0.71); reverse 0.511 (rank 16/20, p = 1.00).

- **Attributed-network replication** — Do the three attributed networks replicate across dataset and modality?
  - approach: LOOCV shrinkage LDA per parcel block, with a 1000-label-permutation null on the discovered block.
  - input: Schaefer-200 volume tables for both datasets, parcels grouped by Yeo-7 networks — discovered block Cont/SalVentAttn/SomMot (the FPN/CON/SMN equivalents), control block Default/Vis/Limbic, reference all parcels.
  - output: Per dataset, accuracy for the discovered, all-parcel and control blocks with permutation p.
  - result: no replication.

## Negative controls

*Finding: head motion alone decodes as well as most connectivity paths and correlates with the star edge; the network restriction is worth nothing over restricting anywhere else; standard bad practices manufacture the headline accuracies other studies report.*

- **Head-motion decoding** — Can head motion alone classify the groups?
  - approach: Motion-only decoding on the same fixed splits as the connectivity analyses, plus per-channel Welch t-tests with Cohen's d under Holm correction.
  - input: fMRIPrep confound timeseries per subject reduced to the mean and SD of each channel — translations, rotations, framewise displacement, DVARS, standardized DVARS, RMSD; no brain features.
  - output: Per-classifier accuracies (SVM, XGBoost, MLP) with Wilson CIs, per-channel group statistics, and the motion feature table.
  - result: 0.608 (SVM) — above the median connectivity specification; 3 of 20 channels nominally significant, 0 survive Holm.

- **Motion contamination of the attributed edges** — Is the attributed coupling a motion artefact?
  - approach: Descriptive subject-level statistics, not cross-validated — per-edge Pearson correlation with motion, Welch group tests, and group effects partialled for motion.
  - input: Network-scale connectivity under all six statistics plus one scalar per subject, the SD of framewise displacement, with participant order asserted identical across loaders.
  - output: Per-statistic edge tables (r with motion, group t, group effect given motion) and a summary for the two attributed couplings.
  - result: CON–SMN × FD r ≈ 0.46–0.57; the CON–SMN group effect survives motion adjustment in 4 of 6 statistics, FPN–SMN in 0 of 4.

- **Network-specificity control** — Is restricting to those networks about *which* networks they are?
  - approach: Connectivity is re-estimated within each 18-region subset (so precision conditions only on the subset), scored by LOOCV shrinkage LDA against 200 size-matched random draws from the discovered-networks pool and from everywhere else.
  - input: The 18 cerebellar regions of the Dosenbach-2010 parcellation from the cached regional timeseries, precision connectivity, n = 32.
  - output: Cerebellar accuracy, its percentile within each matched draw distribution, and the discovered-vs-elsewhere median gap.
  - result: 0.469 against 0.469 — restriction is worth nothing over restricting anywhere else.

- **Overfitting ladder** — What does each bad practice buy?
  - approach: Six protocols of escalating dishonesty on identical data — nested in-fold selection, max over a classifier bank, selection outside CV, best-of-20 fold seeds, scaler/selection leakage under LOOCV, training accuracy — with rungs 1–5 explicitly marked invalid.
  - input: The full region-level correlation connectivity of the Dosenbach-2010 parcellation (~12,880 edges over 32 subjects) and an 8-family classifier bank.
  - output: Accuracy per rung with the method that produced it, and the total accuracy manufactured by stacking shortcuts.
  - result: 0.541 honest → 0.763 circular selection → 0.825 best-of-20-seeds → 1.000 training accuracy.

## Reliability and the ceiling

*Finding: edge reliability in this protocol caps attainable accuracy near 0.72 — below the multiverse's own significance bar. The design cannot succeed; reliability weighting lands exactly on the bar, not over it.*

- **Empirical separability ceiling** — What could *any* classifier reach on these 32 participants?
  - approach: Lachenbruch bias-corrected Mahalanobis distance (ridge-stabilised pooled covariance) converted to the Bayes accuracy Φ(D/2), with stratified-bootstrap CIs and a selection-effect block contrasting fixed-edge LDA with honest in-fold selection.
  - input: The 21 network-edge partial correlations, over three feature sets — all edges, the two attributed couplings, CON–SMN alone — plus the multiverse's stored critical value for the verdict.
  - output: Naive and corrected D², implied ceiling and CI per feature set, the selection-effect gap, and a bar-vs-ceiling verdict.
  - result: over all 21 edges corrected D² is **negative** (Bayes 0.50); the 2 attributed edges give 0.764, but are circularly selected.

- **Theoretical design ceiling** — What is the theoretical ceiling for this design?
  - approach: Four independent analytic ceilings — Bayes, best single edge, reliability attenuation by √ICC, and an estimation (CI-width) ceiling — plus an empirical learning curve from 200 balanced subsample draws per training size.
  - input: The same 21-edge features; literature edge reliability (ICC 0.29 [0.23, 0.36], Noble et al. 2019) as the attenuation constant.
  - output: Ceiling values under each constraint, attainable accuracy as a function of latent effect size and ICC, the learning curve with its slope, and required n per group.
  - result: measured ICC median 0.283 (literature 0.29); attainable accuracy ≈ **0.72**, below the 0.739 threshold.

- **External reliability benchmarks** — Is the reliability estimate specific to this protocol?
  - approach: Per-edge split-half reliability with Spearman–Brown correction estimated on external corpora, reused as feature weights in LOOCV decoding under one 500-permutation family-wise null; ABIDE ROI order is verified by requiring within-network coupling to exceed between-network.
  - input: Network-aggregated Dosenbach-160 timeseries from ~400 ABIDE controls, ~220 LEMON participants and Julia2018, truncated to the target's run length; four connectivity statistics.
  - output: Per-corpus median edge reliabilities and cross-corpus agreement, reliability-estimate precision as a function of corpus size, and plain-vs-weighted target accuracies with the family-wise summary.
  - result: ABIDE 0.732 / 0.570, LEMON 0.367 / 0.230 — brackets Julia2018; reliability is a property of edge × protocol.

- **Reliability-informed weighting** — Can pretraining that targets reliability help?
  - approach: LEMON-estimated per-edge reliabilities as a post-standardisation diagonal feature rescaling under LOOCV shrinkage LDA, with a family-wise null over the 12 cells (4 statistics × {plain, weighted, reliable-edges-only}).
  - input: Julia2018 network timeseries and LEMON regional timeseries aggregated identically and truncated to the same run length; no labels touch the reliability estimates.
  - output: Julia2018-vs-LEMON reliability medians per statistic, LEMON per-edge reliabilities by edge name, and the per-cell accuracies with the family-wise summary.
  - result: no gain beyond the weighting tested below.

- **Closing the gap to the ceiling** — Can reliability weighting close the gap to the ceiling?
  - approach: LOOCV with six heavily regularised linear estimators (LDA variants, ridge, logistic, linear SVC) plus a decision-margin ensemble across statistics, under a single family-wise permutation null over the grid.
  - input: Julia2018 network connectivity under correlation, partial correlation, tangent and precision; the stored reliability and Bayes ceiling constants as reference.
  - output: Accuracy and Wilson CI per statistic × estimator cell and per ensemble, cells clearing the critical value, and the ceiling references.
  - result: peaks at **0.750 against a critical value of 0.750**, p = 0.054 — lands exactly on the bar.

## Structural decoding and cross-dataset transfer

*Finding: grey-matter volume decodes nothing, pooling to n = 79 doesn't help (while scanner site is nearly perfectly decodable), and models don't transfer between samples even though their regional profiles correspond.*

- **Structural decoding** — Is action-game expertise decodable from regional grey-matter volume?
  - approach: Structural decoding under the same discipline as the functional multiverse — a declared 9-cell grid on fixed 4-fold × 25-repeat splits with one 500-permutation family-wise null; attribution explicitly withdrawn after the skull-strip and sphere-atlas defects were found.
  - input: Native-space regional grey-matter volume per subject (ANTs pipeline, normalised by total GM) for 24 experts vs 23 non-experts, under the AAL, Harvard-Oxford and Schaefer-200 volumetric atlases × SVM/XGBoost/MLP.
  - output: The 9-cell specification curve with the family-wise summary and stored null maxima (feeds Figure S7).
  - result: best 0.582, critical 0.664, p = 0.301, 0 of 9.

- **Pooled structural decoding** — Does pooling both samples to n = 79 help?
  - approach: LOOCV shrinkage LDA with per-site mean/SD harmonisation fitted on training rows only, a site-predictability diagnostic, and a 1000-permutation null on the better condition.
  - input: The shared-column union of both GMV tables (AAL by default), pooled to 79 participants (40 players vs 39 non-players) with a two-level site vector identical to the dataset split.
  - output: Site decodability, naive vs harmonised group accuracy with CIs and permutation p, and the recorded label-equivalence assumption.
  - result: best 0.544 (Schaefer-200), p = 0.27; site itself predictable at 0.96–0.99.

- **Cross-dataset model transfer** — Does a model fitted in one sample predict the other?
  - approach: True external validation needing no multiverse correction — a shrinkage LDA fitted on the entire source sample predicts the entire target, with a null that permutes training labels only and re-predicts the held-out dataset (B = 1000); aborts if regional profiles disagree (r < 0.5).
  - input: The shared-column GMV tables per volumetric atlas, three atlases × both directions (gong2015 n = 47 ↔ Julia2018 n = 32).
  - output: Per atlas and direction, the profile correlation, within-dataset accuracies, transfer accuracy and permutation p (feeds Figure S8).
  - result: 0.489–0.511 against null medians 0.500–0.511; profile r = +0.58 to +0.94, so not incomparable features.

## Combining modalities

*Finding: nothing added to functional connectivity helps — structure actively hurts, the widest defensible search stays under its own null threshold, and stacking over all 79 participants finds no signal to pool.*

- **Structure–function fusion** — Does adding structure to function help?
  - approach: LOOCV shrinkage LDA on z-scored in-fold features, all 7 declared cells priced by one 1000-permutation family-wise null.
  - input: The same 32 participants in both modalities — precision connectivity restricted to FPN/CON/SMN edges (functional) and the per-atlas ANTs GMV tables (structural), subject-joined.
  - output: Accuracy per cell (function alone, structure per atlas, concatenation per atlas) and the fusion-minus-function gain.
  - result: function alone 0.812, structure 0.28–0.44, fused 0.41–0.56 — **fusion loses 0.25**.

- **Modality combination census** — How high can accuracy go with more than connectivity?
  - approach: All 15 modality combinations × 8 classifier families on fixed 4-fold × 25-repeat splits; no permutation null here — this census feeds the corrected search below.
  - input: Four Julia2018 modality tables inner-joined per combination — network partial correlation; corrected ANTs AAL grey-matter volumes; mean/SD motion summaries; behavioural accuracy/RT/IES (25 subjects with records).
  - output: One row per combination × classifier with accuracy, its own n and brain-only/uses-behaviour flags, plus best-overall and best-brain-only summaries with an explicit non-comparability note.
  - result: best brain-only cell is connectivity alone; structure never improves it (**stale: the stored census still used the legacy FreeSurfer volume table**).

- **Full multimodal multiverse** — What does the widest defensible search produce, and what is it worth?
  - approach: LOOCV per cell over all modality combinations (up to three blocks) × 7 model families, with one family-wise max-statistic null over the entire grid (B = 500).
  - input: Corrected ANTs GMV (3 atlases), functional connectivity as 4 statistics × {all edges, FPN/CON/SMN only}, motion summaries and behaviour, intersected to 25 participants (**stale: predates the identifier fix that recovers 32**).
  - output: The 833-cell accuracy table and the family-wise summary (observed max, critical value, p, null median).
  - result: max 0.88 vs critical 0.92, p = 0.238 — the null's own median is 0.80.

- **Leave-one-out sensitivity** — Does leave-one-out change the picture?
  - approach: Plain LOOCV over a declared 32-cell search (4 modality combinations × 8 classifiers), labels permuted outside the whole search (B = 200).
  - input: The same multimodal tables restricted to four declared combinations — connectivity; behaviour; connectivity+behaviour; connectivity+motion+behaviour (n = 25, **stale**).
  - output: Per-cell LOOCV accuracies with the family-wise summary and the circularity note on behaviour.
  - result: max 0.840 (connectivity + motion + behaviour, LDA), p = 0.065 — combines a confound with the defining skill.

- **Stacking across block-missing modalities** — Can all 79 participants be used despite block-missing modalities?
  - approach: Stacking — per-modality base learners produce out-of-fold decision scores via an inner 4-fold; a logistic meta-learner reads zero-filled scores plus availability indicators, evaluated on outer 4-fold × 10-repeat splits with separate family-wise nulls per family (B = 200) and an availability-mask-only chance control.
  - input: All 79 pooled participants — GMV for 79/79 (per atlas, in-fold site harmonisation), precision connectivity for the 32 Julia2018 subjects, behaviour for 25; availability pattern identical to site.
  - output: Accuracy for 6 cells (3 atlases × {brain-only, with-behaviour}) with critical values and p, availability counts, and the mask-only control.
  - result: brain-only 0.528 vs critical 0.595 (p = 0.279); with-behaviour 0.515 (p = 0.328); mask-only control 0.448.

- **Modality path weights** — Which modality path carries the signal?
  - approach: Descriptive, no null — stratified out-of-fold base scores feed a single logistic meta-learner fitted on all 79 to read which path carries weight.
  - input: The same three modality blocks (Schaefer-200 structural 79/79, functional 32/79, behaviour 25/79).
  - output: Per-path out-of-fold discriminability (point-biserial r with the label, sign accuracy), pairwise path redundancy, and meta-learner coefficients per score and availability column.
  - result: structural r = +0.05 (n = 79), functional +0.50 (n = 32), behaviour +0.26 (n = 32); paths non-redundant (|r| ≤ 0.19).

## Model capacity and priors

*Finding: neither properly tuned classifiers, hand-built inductive biases, external pretraining, nor normative modelling buys anything — the one cell that reaches 0.812 gets there by circular feature selection.*

- **Hyperparameter-tuning audit** — Does a better classifier, properly tuned, decode?
  - approach: Fixed 4-fold × 25-repeat splits per cell; the tuned condition runs a 20-draw nested randomised search inside each training fold; one family-wise max-statistic null over the grid.
  - input: 13 classifier families × {tuned, untuned} over three feature sets — the 21 partial-correlation edges, the two attributed edges (circular), and the cached normative deviation scores.
  - output: Per-cell accuracy with Wilson bounds, the tuning-effect summary (mean/median delta, cells improved), and the family-wise summary.
  - result: tuning mean effect **−0.003** (median −0.006, 13 of 37 cells improved).

- **Structured inductive biases** — Can prior knowledge substitute for missing data?
  - approach: Four constrained discriminants against an unconstrained shrinkage-LDA baseline under LOOCV, priced by a 500-permutation family-wise null; the discovered-contrast cell also reports an explicitly-unlicensed uncorrected p.
  - input: Precision connectivity over the 21 network edges; the constraints encode the discovered CON–SMN/FPN–SMN contrast (zero fitted direction parameters), FPN/CON/SMN subspace restriction, Laplacian smoothness over the network graph, and a rank-one network mode.
  - output: LOOCV accuracy per bias with the family-wise summary and a provenance note that the "prior" edges were discovered in this same dataset.
  - result: best 0.812 (discovered-contrast) — circular, since the edges come from these same participants.

- **External self-supervised pretraining** — Does representation learned from external data transfer?
  - approach: Unsupervised reconstruction pretraining of a timepoint autoencoder (a seq2seq variant was the rejected candidate), validated on a held-out 10% of LEMON subjects; its value is then measured inside the multiverse's pretraining axis.
  - input: MPI-LEMON network-aggregated resting timeseries only, cut into 124-timepoint windows (Julia2018's run length) at stride 62; no labels.
  - output: The encoder checkpoint consumed by the pre-registered primary and the multiverse's LEMON cells, plus training history and held-out reconstruction error.
  - result: pretraining contributes 1.6% of accuracy variance and does not appear in the best specification.

- **Normative deviation modelling** — Is an AVGP connectome atypical rather than merely different?
  - approach: A denoising vector-diffusion density fitted to external connectivity defines "typical"; each participant is scored by per-edge denoising error, and atypicality is decoded against raw features, a Gaussian/Mahalanobis normative model and an edge-shuffled control, under one family-wise null.
  - input: ~1,100 windowed LEMON network-connectivity vectors as the unlabelled reference population; Julia2018's 32 × 21-edge partial-correlation vectors as targets.
  - output: Per-participant per-edge deviation scores (cached for the tuning audit) and accuracies for 5 feature families × 3 classifiers with the family-wise summary.
  - result: 0.345–0.435 against 0.607–0.666 for the raw features — **worse than its own input**.

- **Augmentation and weak supervision** — Do augmentation and weak supervision apply here?
  - approach: Multiple-instance learning — shrinkage LDA trained on windows, pooled to a participant decision by mean/max/vote, with strictly participant-level LOOCV and participant-level label permutation.
  - input: Julia2018 network timeseries cut into 60-TR sliding windows (stride 16), precision connectivity restricted to FPN/CON/SMN edges — one bag of instances per participant.
  - output: Whole-run baseline vs the three pooling accuracies, window geometry, and the family-wise summary.
  - result: no gain.

## Deep architectures

*Finding: end-to-end models overfit to perfect training accuracy and decode nothing; letting the model learn the region-to-network map produces assignments no more label-driven than noise.*

- **Multi-view hierarchical encoder** — Can one model read all atlases and statistics at once?
  - approach: An end-to-end shared-encoder classifier over all views pooled into a canonical vocabulary, fold features precomputed leak-free, with accuracy and attribution both priced against permuted labels.
  - input: 9 atlas timeseries × 5 connectivity statistics = 45 views of the same 32 participants, mapped by a hand-declared crosswalk onto 7 canonical networks (28 edges); 75 parameters.
  - output: Test/train accuracy with permutation p, per-canonical-edge attribution, and per-view (atlas × statistic) attention weights.
  - result: 0.375 vs null median 0.494, p = 0.96.

- **Learned parcellation** — What if the region-to-network map is learned rather than assumed?
  - approach: Jointly learns an annealed-softmax region-to-network assignment and a classifier head from region connectivity, with an atlas-init vs random-init ablation and permutation nulls on accuracy, attribution and parcellation drift.
  - input: Dosenbach-2010 region-level partial correlation (n = 32), the atlas's own region-to-network map as initialisation.
  - output: Test/train accuracy, learned-network-edge attributions, and assignment drift, each with its permuted-label reference.
  - result: 0.469 (train 1.000), p = 0.63; drift 0.214 against permuted-label drift 0.198.

- **Multi-atlas learned parcellation** — Does learning it across atlases in one shared space help?
  - approach: One learned assignment per atlas pooling into a shared canonical space, with shared mid-layer, atlas attention and head; every reported quantity priced against permuted labels.
  - input: Seven atlas timeseries (Dosenbach-2010, Gordon, Seitzman, Schaefer-100/200, DiFuMo-64/128), partial correlation, crosswalk-initialised into the 5 canonical networks all seven define; 7,043 parameters.
  - output: Accuracy, canonical-edge attribution in one shared vocabulary, per-atlas attention weights, and per-atlas drift against its null.
  - result: 0.313 (train 1.000), p = 0.99; **every atlas drifts less on real labels than on shuffled ones**.

## Replications

*Status: standalone notebook analyses; their results are not part of the stored outputs, and no manuscript claim rests on them.*

- **Control-network replication** — Do the classic graph-analytic control networks emerge in this resting dataset?
  - approach: Reimplements Dosenbach et al. (2007) faithfully — threshold and binarize seed connectivity, cluster, characterise — plus bootstrapped hierarchical clustering and small-world graph measures from `acnets.connectome` (small-worldness; transfer entropy and Chatterjee correlation as exploratory extras).
  - input: Resting timeseries of the 39 control-related ROIs from the original systematic review, one 39 × 39 correlation matrix per subject.
  - output: Notebook replications of the original Figures 5 and 9, cluster solutions, threshold-selection analyses and graph metrics.
  - result: qualitative replication record only.

- **Task and behaviour replication** — Does the attention task show the expected cue-validity effect?
  - approach: First-level GLM of the attention task and a behavioural validity-effect analysis, fed by the R pipeline that tidies the raw behavioural exports.
  - input: Julia2018 task-fMRI runs and trial-level behavioural tables (R loaders for the NVGP37/VGP04 exports).
  - output: First-level contrast maps and validity-effect estimates in notebook outputs.
  - result: qualitative replication record only.

## Deep-learning line (in progress)

- **Generative and temporal models** — Can generative or temporal architectures learn the connectome distribution?
  - approach: Keras 3 on the PyTorch backend — GAN and conditional GAN, conditional VAE, a hierarchical temporal model (region timecourses → network timecourses → similarity → classifier), and multi-input classifiers — with `pyspi` pairwise-interaction statistics as candidate features.
  - input: Region-level timeseries and pairwise-interaction feature tensors from the Julia2018 and LEMON data modules.
  - output: Model code with tests, training notebooks and architecture figures; no stored decoding results yet.
  - result: in development; not part of the manuscript's quantitative claims.

## Exploratory and supplementary notebooks

- **EDA and sanity checks** — descriptive views and verifications.
  - approach: LDA and UMAP projections, outlier screening, tangent- and covariance-connectivity verification, network factor analysis, AutoML and model-comparison runs, plus the cerebellum and confounds classifier prototypes.
  - input: The same Julia2018 connectivity derivatives as the script suite.
  - output: Notebook-level figures and tables only.
  - result: descriptive, or superseded by the scripted cerebellar, motion, tuning and multiverse analyses; no manuscript claims rest on them.

---

## Known staleness

- The full multimodal multiverse and its leave-one-out variant intersected on behaviour and so ran on 25 participants; the replacement-identifier fix in the brain–behaviour pipeline makes 32 available. The 833-cell family's p = 0.2376 appears in manuscript §3.4.5.
- The stored agreement variance decomposition is empty, so §3.4.2's variance decomposition is not reproducible from stored outputs until the agreement analysis is re-run.
- The modality-combination census and the leave-one-out sensitivity analysis now read the corrected ANTs AAL grey-matter volumes instead of the legacy FreeSurfer-derived table; their stored results predate this swap and need a rerun.
