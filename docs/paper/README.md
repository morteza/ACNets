# Paper working directory

## Files

| File | Role |
|---|---|
| `manuscript.md` | The paper. Single source; edit this one. |
| `plan.md` | Record of the executed research plan: phases A–E, prereg gate, structural arm. Read its status log; the forward-planning sections are preserved for provenance. Doubles as the OmniData/OmniModel pilot protocol. |
| `supplementary.md` | Editorial record: provenance, coauthor comment disposition, text cut from the original. |
| `literature_review.md` | Background review the introduction and discussion draw on. Integrated as of this revision; its verification flags are carried into the References section and Appendix E item 2. |
| `decodability_ceiling.md` | 24 sections asking how far accuracy can honestly be pushed, and why it stops where it does. Contains the reliability-ceiling analysis, the overfitting ladder, and the pipeline defects found along the way. |
| `publication_strategy.md` | What is publishable without pre-registration, by tier, with the required framing. |

There used to be two manuscripts — `manuscript_transfer_framing.md` and
`manuscript_multiverse_framing.md` — differing only in which of the two stories led. The README
described the second as generated from the first by `scripts/build_framings.py`, checked by
`tests/test_framings_current.py`. **Neither script ever existed**, so the "generated" file was a
hand-maintained copy that had already begun to drift. Both are now collapsed into
`manuscript.md`, which leads with the transfer question and treats the multiverse as the method
that makes the adjudication trustworthy.

## Conventions

`[[TODO: ...]]` marks a value the analyses have yet to produce. Nothing is invented; everything
outside a marker is submittable prose. Specification counts are computed from
`acnets.multiverse.SpecificationRegistry().summary()` rather than asserted.

## Where the results currently stand

The grid is **49 specifications** (one atlas, six statistics, two aggregation scales, three
classifier families, with and without LEMON pretraining). It was 54 until five cells were found
to be exact duplicates — `xgboost + lemon` on a stateless statistic is bit-identical to its
`none` twin, since trees are scale-invariant and only `tangent` carries measure state.

| | result | verdict |
|---|---|---|
| Decoding accuracy, family-wise permutation null | best 0.660, critical value 0.739, **p = 0.291** at B = 1000, 0 of 49 significant | **does not survive** |
| Attribution agreement vs random rankings (RBO) | excess +0.366, p = 0.019 | **survives** |
| Same, dimensionality-matched random-network control | excess +0.154, p = 0.135 | correctly fails |
| Same, Kendall tau over the full ordering | excess +0.085, p = 0.30 | **fails** — only the top of the ranking is stable |
| Real vs shuffled network assignment, 18 matched pairs | 0.564 vs 0.455, real ahead in 15/18 | **the clearest positive result** |
| Motion-only classifier | 0.608 | **a real confound, see below** |

The null is not a near miss. Refitting all 49 specifications under permuted labels and keeping
the best gives a maximum accuracy of 0.625 at the median and 0.739 at the 95th percentile —
searching 49 paths over 32 participants produces about 0.63 from nothing, and produces more
than the observed 0.660 in 29% of permutations.

So the paper's centre of gravity has moved from "AVGPs and NVGPs differ in connectivity" to
"the attribution is reproducible across the multiverse even where the accuracy is not", and the
claim is now stated narrowly: the *leading* features are reproducible, not the ranking.

**Two further sections landed in this revision, and they carry most of the methodological
weight.** §3.7 reports the reliability ceiling — median edge ICC 0.283 against the
literature's 0.29, an implied accuracy cap near 0.72, and the observation that at n = 32 this
cap sits *below* the 0.739 family-wise critical value, so the design is close to
unfalsifiable by construction. §3.8 records everything else that was tried and did not help:
13 classifier families with nested HPO, multimodal combination, structure-function fusion,
multiple-instance learning, diffusion normative modelling, and the six-rung overfitting
ladder that runs from an honest 0.541 to 1.000.

**Three findings from this pass weaken earlier claims, and all three are now in the draft.**
Head motion differs by group and decodes at 0.608, so an earlier §3.5 sentence asserting it did
not was withdrawn; CON-SMN survives controlling for motion in four of six statistics but
FPN-SMN does not. `MultiScaleClassifier`'s regional feature block is the temporal mean of
z-scored time courses, i.e. zero by construction, so §3.2 was rebuilt from scratch. And the
largest effect measured anywhere in this project is an axis the registry does not contain:
averaging region time courses into networks before estimating connectivity, rather than after,
is worth 0.13 to 0.19 in accuracy.

Regenerate with:

```bash
uv run python scripts/run_agreement.py      # attribution agreement + its nulls
uv run python scripts/run_permutation.py    # family-wise max-statistic null
uv run python scripts/run_motion_control.py # motion negative control
uv run python scripts/run_motion_coupling.py # motion vs the attributed features
uv run python scripts/run_cerebellar_control.py # spatially-displaced negative control

# sections 3.7-3.8: the ceiling, and everything else that was tried
uv run python scripts/run_reliability_transfer.py   # split-half reliability + weighting
uv run python scripts/run_abide_reliability.py      # the same, on two external corpora
uv run python scripts/estimate_ceiling.py           # bias-corrected Mahalanobis, Bayes accuracy
uv run python scripts/run_hpo.py                    # 13 classifier families, tuned and not
uv run python scripts/run_multimodal.py             # modality combinations
uv run python scripts/run_struct_func_fusion.py     # structure-function fusion
uv run python scripts/run_weak_supervision.py       # multiple-instance learning
uv run python scripts/run_normative.py              # diffusion normative model
uv run python scripts/run_overfitting_ladder.py     # the methodological demonstration
uv run python scripts/run_multiscale.py     # section 3.2 feature blocks

# structural arm, one pipeline for both datasets
uv run python scripts/prepare_atlases.py                 # volumetric atlases via nilearn
$ANTSENV/bin/python scripts/segment_t1.py --nifti ... --out ... --name ...
uv run python scripts/run_cross_dataset.py --atlas aal   # external validation
uv run python scripts/run_pooled_structural.py --atlas aal
```

The remaining `decodability_ceiling.md` analyses that did not reach the manuscript have their
own drivers: `run_ceiling_theory.py`, `run_reach_ceiling.py`, `run_inductive_bias.py`,
`run_full_multimodal.py`, `run_crossmodal_networks.py`, `run_loocv_null.py`,
`run_pretrain.py`, `run_primary.py`, `explore_features.py`.

## Before submission

Appendix E lists the open items. Items 6b, 6c and 9 — which used to gate everything — are all
resolved. What remains needs a decision or an outside source rather than a run:

1. **Item 15, the motion confound.** Disclosed and quantified, but the response is a choice:
   leave it as a limitation (current), attempt motion-matched subsamples (loses too many
   participants at n = 32), or add scrubbing as a multiverse axis (defensible, doubles the grid).
2. **Item 14, aggregation order.** The largest effect measured in the project is not an axis of
   the registry. Adding it changes every `spec_id` and needs the family-wise null rerun.
3. **Item 2, the reference list.** Incomplete, and the second table of entries still needs
   confirming or dropping.
4. **Repository URL, and one blocked analysis** — the LEMON noise ceiling, which needs
   participant metadata from a fresh MPI fetch. The cerebellar control was never blocked:
   Dosenbach2010 carries 18 cerebellar regions, and it has now been run (§3.5).

## A connection worth following up

The literature review names **Gong et al. (2015)**, *Enhanced functional connectivity and
increased gray matter volume of insula related to action video game playing* (Sci Rep 5:9763),
as "the single most on-point precedent" — action-game experts versus amateurs, increased
connectivity between attentional and sensorimotor networks, concentrated in the left insula.

`data/gong2015/` contains 24 `AVG-expert` and 23 `AVG-nonexpert` subjects of raw GE DICOM,
T1-weighted (`Ax 3D FSPGR`, 1 mm isotropic), protocol name `Brain_gong_dota`. DotA is an action
title, the group labels match, and the sample sizes are in the right range. **CONFIRMED: this is the structural data from that paper**, shared by its authors, i.e. the
grey-matter-volume arm of the closest precedent to the present study.

That makes it a **reanalysis of the closest precedent** rather than a generic extra sample: an
independent pipeline and a multiverse discipline applied to the same T1 data, at n = 47, in a
modality the resting arm cannot reach. Nothing is pooled with Julia2018 — different subjects,
scanner and modality — so it extends the paper by replication, not by sample size.

Processed in Phase D, then **reprocessed** after two defects were found in the first
pipeline: skull-contaminated volumes and a sphere atlas unusable for cross-dataset
volumetry (`decodability_ceiling.md` §18–19, §23).

Headline as it now stands: decoding null within gong2015 — 0.574 leave-one-out, and 0.582
as the best of its own nine-cell grid against a 0.664 critical value (p = 0.30, 0 of 9) —
**6 of 6 external-validation tests at chance** across three volumetric atlases and both
directions, pooled n = 79 at 0.544 (p = 0.27). **The attribution result is
withdrawn** — the earlier "sensorimotor cortex leads, insula not recovered" came from the
contaminated volumes, and attribution ranks from a chance-level classifier describe noise.
The claim that the dissociation replicates across modality is withdrawn with it.
