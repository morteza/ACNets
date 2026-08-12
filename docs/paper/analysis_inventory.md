# Analysis inventory

One entry per analysis: the question it asks, what goes in, what comes out, what it found.
Accuracies are not comparable across entries — they come from different estimators,
feature spaces and participant subsets. Each is stated with what produced it.

---

## Descriptive and design

- **`run_descriptives`** — What are these participants like?
  - in: Julia2018 timeseries, participants.tsv
  - out: `outputs/descriptives.json`
  - result: 16/16, all male, age-matched (20.8 ± 2.1 vs 21.1 ± 2.0)

- **`run_brain_behavior`** — Does control-to-sensorimotor coupling track attentional performance?
  - in: Julia2018 network partial correlation (CON-SMN, FPN-SMN) + task accuracy/RT/IES, n = 32
  - out: `outputs/brain_behavior.json`
  - result: 3 of 18 bootstrap intervals exclude zero, incl. CON-SMN × IES within AVGP (r = −0.378 [−0.682, −0.070]); the same analysis at n = 25 found nothing

- **`run_behaviour_decoding`** — Can behaviour alone classify group, and what does that calibrate?
  - in: accuracy, RT, IES; 4 classifiers with nested grid search, 12 label-independent folds
  - out: `outputs/behaviour_decoding/behaviour_decoding.json`
  - result: best 0.615 (logistic) vs critical 0.679, p = 0.119 — **a known real effect (d ≈ 0.75) also fails at n = 32**

## Primary decoding and the multiverse

- **`run_permutation`** — Does any analysis path decode above what searching that many paths produces by chance?
  - in: 49-specification grid (Dosenbach2010 × 6 statistics × 2 aggregations × 3 classifiers × pretraining), B = 1000
  - out: `outputs/permutation/`
  - result: best 0.660, critical 0.739, p = 0.291, 0 of 49; null max median 0.625

- **`run_primary`** — Does the one model specified in writing before it was fitted decode?
  - in: LEMON-pretrained latent-connectivity classifier from `prereg_primary.md`, run once
  - out: `outputs/primary/`
  - result: 0.477, null 95th 0.514, p = 0.0999

- **`run_multiscale`** — Which account of transfer breadth does the connectivity structure support?
  - in: region-level partial correlation → H1 regional degree / H2 within-network / H3 between-network blocks, identical folds
  - out: `outputs/multiscale/`
  - result: all blocks at chance (0.45–0.57); aggregate-then-connect beats connect-then-average by 0.13–0.19

- **`explore_features`** — Is there any feature family that separates the groups?
  - in: broad sweep over feature constructions
  - out: `outputs/explore/`
  - result: a candidate list, explicitly labelled a search rather than a result

## Attribution and explainability

- **`run_agreement`** — Do different analysis paths agree on which features matter?
  - in: 31 network-scale specifications, one attribution vector each (Haufe / SHAP / gradient×input)
  - out: `outputs/agreement/`
  - result: RBO 0.634 vs random-ranking floor 0.423 (+0.366, p = 0.019); Kendall τ at chance

- **`run_attribution_null`** — Does that agreement depend on the labels?
  - in: same grid refit on permuted labels, same folds/features/attribution
  - out: `outputs/agreement/attribution_null*.json`
  - result: 3 classifier families 0.657 vs 0.562, **p = 0.035**; linear family alone 0.630 vs 0.601, p = 0.29; consensus leader CON-SMN in 62/1000 permutations (p = 0.063)

- **`run_attribution_geometry`** — Could the agreement be feature covariance rather than group signal?
  - in: 21 edges ranked by variance, magnitude, top eigenvector, random-decoder Haufe pattern
  - out: `outputs/agreement/attribution_geometry.csv`
  - result: 0.278–0.461 against the consensus, inside the random floor 0.408 [0.282, 0.611] — geometry does not explain it

- **`run_attribution_transfer`** — Do regions found by XAI in one dataset predict in the other?
  - in: Haufe attribution on gong2015 / Julia2018 structural volumes, both directions
  - out: `outputs/attribution_transfer/`
  - result: does not transfer

- **`run_crossmodal_networks`** — Does a network identity found in one modality survive into the other?
  - in: top-3 network triples, only identity crosses the gap
  - out: `outputs/crossmodal_networks/`
  - result: gong2015 structure → Julia2018 function 0.562 (rank 10/20, p = 0.71); reverse 0.511 (rank 16/20, p = 1.00)

- **`run_network_replication`** — Do the three attributed networks replicate across dataset and modality?
  - in: FPN/CON/SMN restriction applied outside the sample that generated it
  - out: `outputs/network_replication/`
  - result: no replication

## Negative controls

- **`run_motion_control`** — Can head motion alone classify the groups?
  - in: 20 fMRIPrep motion summaries, same folds
  - out: `outputs/controls/motion.json`
  - result: 0.608 (SVM) — above the median connectivity specification; 3 of 20 channels nominally significant, 0 survive Holm

- **`run_motion_coupling`** — Is the attributed coupling a motion artefact?
  - in: per-edge correlation with FD-SD; group effect partialled for motion
  - out: `outputs/controls/`
  - result: CON-SMN × FD r ≈ 0.46–0.57; CON-SMN group effect survives in 4 of 6 statistics, FPN-SMN in 0 of 4

- **`run_cerebellar_control`** — Is restricting to those networks about *which* networks they are?
  - in: within-cerebellum precision connectivity (18 regions) vs 200 size-matched random draws
  - out: `outputs/controls/`
  - result: 0.469 against 0.469 — restriction is worth nothing over restricting anywhere else

- **`run_overfitting_ladder`** — What does each bad practice buy?
  - in: 6 variants over the same 12,880 region-level features
  - out: `outputs/ladder/`
  - result: 0.541 honest → 0.763 circular selection → 0.825 best-of-20-seeds → 1.000 training accuracy

## Reliability and the ceiling

- **`estimate_ceiling`** — What could *any* classifier reach on these 32 participants?
  - in: bias-corrected Mahalanobis distance over candidate feature sets
  - out: `outputs/ceiling/ceiling.json`
  - result: over all 21 edges D² corrected is **negative** (Bayes 0.50); the 2 attributed edges give 0.764, but are circularly selected

- **`run_ceiling_theory`** — What is the theoretical ceiling for this design?
  - in: measured edge reliability + attenuation + sample size
  - out: `outputs/ceiling/theory.json`
  - result: measured ICC median 0.283 (literature 0.29); attainable accuracy ≈ **0.72**, below the 0.739 threshold

- **`run_abide_reliability`** — Is the reliability estimate specific to this protocol?
  - in: split-half edge reliability in ABIDE controls and MPI-LEMON
  - out: `outputs/reliability/`
  - result: ABIDE 0.732 / 0.570, LEMON 0.367 / 0.230 — brackets Julia2018; reliability is a property of edge × protocol

- **`run_reliability_transfer`** — Can pretraining that targets reliability help?
  - in: externally estimated per-edge reliability used to weight features
  - out: `outputs/reliability/`
  - result: no gain beyond the weighting below

- **`run_reach_ceiling`** — Can reliability weighting close the gap to the ceiling?
  - in: 12 cells, 4 statistics × {plain, reliability-weighted, reliable-edges-only}
  - out: `outputs/ceiling/reach.json`
  - result: peaks at **0.750 against a critical value of 0.750**, p = 0.054 — lands exactly on the bar

## Structural arm and cross-dataset

- **`run_gong2015`** — Is action-game expertise decodable from regional grey-matter volume?
  - in: corrected native-space GMV, 3 volumetric atlases × 3 classifiers, B = 500
  - out: `outputs/gong2015/`
  - result: best 0.582, critical 0.664, p = 0.301, 0 of 9

- **`run_pooled_structural`** — Does pooling both samples to n = 79 help?
  - in: gong2015 + Julia2018 GMV, site harmonised inside training folds, expert-as-player mapping
  - out: `outputs/pooled_structural/`
  - result: best 0.544 (Schaefer-200), p = 0.27; site itself predictable at 0.96–0.99

- **`run_cross_dataset`** — Does a model fitted in one sample predict the other?
  - in: 3 atlases × 2 directions, feature-profile correspondence checked first
  - out: `outputs/cross_dataset/`
  - result: 0.489–0.511 against null medians 0.500–0.511; profile r = +0.58 to +0.94, so not incomparable features

## Combining modalities

- **`run_struct_func_fusion`** — Does adding structure to function help?
  - in: 7 declared cells, functional / structural / concatenated, one family-wise null
  - out: `outputs/fusion/`
  - result: function alone 0.812, structure 0.28–0.44, fused 0.41–0.56 — **loses 0.25**

- **`run_multimodal`** — How high can accuracy go with more than connectivity?
  - in: connectivity, GMV, motion, behaviour in every combination
  - out: `outputs/multimodal/`
  - result: best brain-only cell is connectivity alone; structure never improves it

- **`run_full_multimodal`** — What does the widest defensible search produce, and what is it worth?
  - in: 833 cells over 4 modalities, family-wise null (**stale: ran on n = 25 before the identifier fix**)
  - out: `outputs/full_multimodal/`
  - result: max 0.88 vs critical 0.92, p = 0.238 — the null's own median is 0.80

- **`run_loocv_null`** — Does leave-one-out change the picture?
  - in: multimodal search under LOOCV with its own null (**stale: n = 25**)
  - out: `outputs/loocv/`
  - result: max 0.840 (connectivity + motion + behaviour, LDA), p = 0.065 — combines a confound with the defining skill

- **`run_stacked_all`** — Can all 79 participants be used despite block-missing modalities?
  - in: per-modality base learners + availability masks → meta-learner, 3 atlases × 2 families, B = 200
  - out: `outputs/stacked_all/stacked_all.json`
  - result: brain-only 0.528 vs critical 0.595 (p = 0.279); with-behaviour 0.515 (p = 0.328); mask-only control 0.448

- **`run_stacked_weights`** — Which modality path carries the signal?
  - in: out-of-fold path scores, n = 79
  - out: `outputs/stacked_all/stacked_weights.json`
  - result: structural r = +0.05 (n = 79), functional +0.50 (n = 32), behaviour +0.26 (n = 32); paths non-redundant (|r| ≤ 0.19)

## Model capacity and priors

- **`run_hpo`** — Does a better classifier, properly tuned, decode?
  - in: 13 families × {tuned, untuned} × feature sets = 74 cells, nested randomised search
  - out: `outputs/hpo/`
  - result: tuning mean effect **−0.003** (median −0.006, 13 of 37 cells improved)

- **`run_inductive_bias`** — Can prior knowledge substitute for missing data?
  - in: discovered-contrast, network-restricted, Laplacian-smooth, rank-one constraints
  - out: `outputs/inductive_bias/`
  - result: best 0.812 (discovered-contrast) — circular, since the edges come from these same participants

- **`run_pretrain`** — Does representation learned from external data transfer?
  - in: encoder pretrained on 220 MPI-LEMON subjects, no labels
  - out: `outputs/pretrain/`
  - result: pretraining contributes 1.6% of accuracy variance and does not appear in the best specification

- **`run_normative`** — Is an AVGP connectome atypical rather than merely different?
  - in: diffusion model on 1,100 LEMON windows → per-participant deviation scores
  - out: `outputs/normative/`
  - result: 0.345–0.435 against 0.607–0.666 for the raw features — **worse than its own input**

- **`run_weak_supervision`** — Do augmentation and weak supervision apply here?
  - in: the subset of low-label-regime methods this design admits
  - out: `outputs/weak_supervision/`
  - result: no gain

## Deep architectures

- **`run_hierarchical`** — Can one model read all atlases and statistics at once?
  - in: 9 atlases × 5 statistics = 45 views → canonical 28-edge space, shared encoder, 75 parameters
  - out: `outputs/hierarchical/`
  - result: 0.375 vs null median 0.494, p = 0.96

- **`run_learned_parcellation`** — What if the region-to-network map is learned rather than assumed?
  - in: Dosenbach2010 region connectivity → annealed-softmax assignment → network bottleneck → head, end to end
  - out: `outputs/learned_parcellation/learned_parcellation.json`
  - result: 0.469 (train 1.000), p = 0.63; drift 0.214 against permuted-label drift 0.198

- **`run_learned_parcellation_multi`** — Does learning it across atlases in one shared space help?
  - in: 7 atlases → 5 common canonical networks, per-atlas assignments, 7,043 parameters
  - out: `outputs/learned_parcellation/multi_atlas.json`
  - result: 0.313 (train 1.000), p = 0.99; **every atlas drifts less on real labels than on shuffled ones**

---

## Supporting utilities (not analyses)

- **`build_parcellations`**, **`prepare_atlases`** — extract regional timeseries; build atlas definitions
- **`segment_t1`**, **`gong2015_segment`**, **`gong2015_convert`** — corrected ANTs structural pipeline; gong2015 ingestion
- **`make_figures`** — render figures from stored outputs, fitting nothing
- **`fetch_test_fixtures`** — test data

## Known staleness

- `run_full_multimodal` and `run_loocv_null` intersected on behaviour and so ran on 25 participants; the identifier fix in `run_brain_behavior.py` makes 32 available. The 833-cell family's p = 0.2376 appears in manuscript §3.4.5.
- `outputs/agreement/variance.json` is empty, so §3.4.2's variance decomposition is not reproducible from stored outputs.
