# What is publishable without pre-registration

The question this answers: given that the decoding claim fails its family-wise
null, and given that pre-registration is not on the table, how much of this work
can be published and under what framing?

**Short answer: most of it, and the strongest parts do not depend on a positive
decoding result at all.** What is not publishable is one specific sentence —
"AVGPs and NVGPs differ in resting-state connectivity" — and the paper does not
need it.

---

## Tier 1: publishable as-is, no caveats beyond ordinary limitations

These claims survive every check in `decodability_ceiling.md` and are the paper.

| claim | evidence |
|---|---|
| **The dissociation.** Attribution is reproducible across the multiverse where accuracy is not. | RBO +0.366, p = 0.019; accuracy p = 0.291, 0/49 specifications |
| **The atlas aggregation is meaningful.** Real network assignment beats size-matched shuffles. | 0.564 vs 0.455, real ahead in 15 of 18 matched pairs |
| **The structural arm is a clean, externally-validated null.** | gong2015 reanalysed with a corrected pipeline: within-sample 0.574, 6/6 cross-dataset transfer tests at chance, pooled n = 79 gives 0.544 (p = 0.27) |
| **CON-SMN is the stable feature.** | top-5 in 31/31 specifications; recovered by independent in-fold univariate selection in 94/100 folds |
| **A reliability ceiling explains the field's accuracy band.** | ICC 0.28 measured, 0.29 literature; ceiling ≈ 0.72; needs latent d = 3.1 for 80% |
| **At n = 32 the ceiling coincides with the null's 95th percentile.** | 0.719 vs 0.720 — the design is unfalsifiable by construction |
| **Quantified cost of common shortcuts.** | the ladder: 0.541 → 1.000, +0.459 manufactured |
| **The same model priced by search width.** | 0.812 at p = 0.0010 (7 cells), 0.0066 (20), 0.0399 (80); 0.880 at p = 0.24 (833). Publish the *shape* — how much a declared family width is worth — never any single row as our result: the three networks came from §3.1, so no family we can declare is valid (§15.1) |
| **Parcel size governs cross-dataset volumetry.** | profile agreement −0.41 (spheres) to +0.94 (Harvard-Oxford), monotone in parcel size |
| **A large search yields 80% from noise half the time.** | 833-cell null: median maximum 0.800 at n = 25 |

The last three are, in our view, the most novel and the most citable. They are
methodological results with relevance well beyond action video games, and none
of them requires the group contrast to be decodable.

## Tier 2: publishable as explicitly exploratory

| claim | required framing |
|---|---|
| Inductive bias reaches 0.812, p = 0.040 corrected over statistic x network triple | Must state that the correction covers the final step and not the analytic path, that `CON+DMN+SMN` and `FPN+SMN+VIS` tie, and that motion control costs 0.125 (0.812 → 0.688, below the 0.781 critical value) |
| The three networks do **not** replicate in gong2015 | Cross-modal transfer fails in both directions (p = 0.71, p = 1.00); the structural analogue puts the *control* networks above the discovered ones. Report as a failed replication attempt, not omitted |
| Reliability-weighted transfer, 0.750 | Label it a method demonstration; p = 0.054 |
| Behaviour-inclusive multimodal, 0.840 LOOCV | Only alongside its null (p = 0.065) and the note that behaviour is the trained skill defining the groups |

These belong in a clearly-marked exploratory section, with the confirmatory
analyses reported first and the exploratory ones not mentioned in the abstract.
That is standard practice and reviewers accept it when the labelling is honest.

## Tier 3: not publishable without new data

- "Action video game players differ from non-players in resting connectivity."
- Any accuracy presented as a validated marker.
- 0.812 as a confirmed effect rather than a hypothesis.

Improving the pipeline cannot move a claim from Tier 3 to Tier 1. Each
improvement is chosen after seeing the data, so the family-wise correction must
widen to cover it — which is why 279 configurations, 13 tuned classifier
families, a diffusion prior and four modalities all landed at the same place.
The only routes are an independent sample or a registered report.

## Recommended framing

Lead with the method and the dissociation, not with the group contrast. The
title already does this. Concretely:

1. **Abstract**: dissociation first, ceiling second, group contrast reported as
   null. Do not put 0.812 in the abstract.
2. **Results**: confirmatory (sections 3.1-3.6) then a separate, clearly labelled
   exploratory section for the ceiling-directed analyses.
3. **Discussion**: the ceiling result is the general contribution — it predicts
   where any similar study will land, and explains a published pattern.
4. **Limitations**: motion, n, acquisition length, and the path-dependence of the
   exploratory analyses. All already drafted.

## Venue

`Imaging Neuroscience` remains the best fit: it publishes well-powered null
results and methodological work, and the specification-curve figure needs room.
`Human Brain Mapping` and `NeuroImage` are reasonable. A methods-forward venue
would also take the ceiling analysis on its own.

The current publishing climate helps here rather than hurting: an honestly
corrected null with a strong methodological contribution is more publishable
today than a marginal positive would be, and considerably more durable.

## What would change the answer

One independent AVGP sample. The specification is already fixed by section 15 —
precision connectivity, FPN/CON/SMN restricted, shrinkage LDA, leave-one-out —
so a replication needs no correction at all, and the motion confound could be
handled by design. That converts Tier 3 into Tier 1 in a single study.
