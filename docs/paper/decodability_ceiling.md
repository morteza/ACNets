# Can this dataset be decoded at high accuracy?

Asked directly: is there a pipeline — some other atlas, connectivity measure,
architecture, or pretraining scheme — that separates AVGPs from NVGPs well above
chance, and could then be used to explain the difference?

This document is the empirical answer. It is not an argument from principle; every
number below was measured on the data.

**Short version.** Widening the search does not work, and we can now say by how much
it fails. Narrowing it *appears* to work spectacularly and does not survive contact
with nested cross-validation. The binding constraint is not the model, the
parcellation, or the software — it is that 32 participants carry roughly one
feature's worth of usable signal, and the correction for having searched costs more
than that signal is worth.

---

## 1. Widening the search: measured, and it fails

`scripts/explore_features.py` searched feature families that carry *different*
information from the six connectivity statistics already in the grid — which are
all functions of the same 6×6 covariance and therefore cannot find what each other
missed. Across 9 cached parcellations × 6 feature families × 3 classifiers:

| family | what it adds | best |
|---|---|---|
| `static` | the existing baseline | 0.588 |
| `dynamic` | sliding-window coupling, **mean and SD** — static connectivity throws the SD away | 0.626 |
| `region+select` | full region-level edge set, univariate selection inside folds | 0.608 |
| `dynamic+pca` | dynamic, reduced in-fold | 0.594 |
| `region+pca` | region-level, PCA inside folds | 0.560 |
| `strength` | weighted node strength, positive and negative separately — a graph property | 0.556 |

**Best of 135 configurations: 0.626.** The family-wise null run at B = 1000 has a
null-maximum *median* of 0.625. So the best thing found by searching 135 new ways
is, to three decimal places, what searching produces when there is nothing to find.
It sits at the **50th percentile** of the null.

Higher spatial resolution made things worse, not better: gordon2014 (333 regions)
reached 0.534 and seitzman2018 (300 regions) 0.459, against dosenbach2010's 0.626.
That is the curse of dimensionality doing exactly what it does at n = 32.

Adding any of these to the multiverse would *raise* the bar rather than clear it.
This is not a prediction; it is the trajectory already observed as the grid grew
from 24 to 36 to 49 specifications: p = 0.114 → 0.244 → 0.291.

## 2. Narrowing the search: the tempting result, and why it is not real

The opposite move works far better — and is a trap worth documenting, because the
first version of it looks like a solution.

Estimating the ceiling directly (`scripts/estimate_ceiling.py`) via the Mahalanobis
distance between groups, which bounds *every* classifier at once through
Bayes accuracy = Φ(D/2):

| feature set | p | naive D² | bias-corrected D² | implied ceiling |
|---|---|---|---|---|
| all 21 network edges | 21 | 5.87 | **−1.06** | 0.500 |
| the 2 attributed edges | 2 | 2.58 | 2.07 | **0.764** [0.68, 0.89] |
| CON-SMN alone | 1 | 1.37 | 1.15 | 0.704 [0.58, 0.84] |

Two things to read here. First, the bias correction matters enormously: with 21
features and 32 participants the naive D² of 5.87 implies a ceiling of 0.89, and
after correction there is **no detectable separation at all**. That gap is precisely
the illusion that persuades small-sample studies a better model is one idea away.

Second, restricting to two features raises the corrected ceiling to 0.764 — above
the family-wise critical value of 0.739. And fitting it confirms this:

| feature set | LDA | SVM | XGBoost | MLP |
|---|---|---|---|---|
| CON-SMN + FPN-SMN | 0.750 | **0.760** | 0.606 | 0.738 |
| CON-SMN alone | 0.721 | 0.721 | 0.695 | 0.685 |
| all 21 edges | 0.561 | 0.608 | 0.666 | 0.635 |

**0.760, comfortably above the bar.** This is the answer the question was looking
for, and it is wrong.

Those two edges are the top of an attribution ranking computed on these same 32
participants, so choosing them consulted the labels. Putting the choice inside the
folds — selecting the top 2 edges by training-fold *t*-statistic, then classifying —
gives the honest number:

| | accuracy |
|---|---|
| 2 edges chosen on all 32 participants | 0.750 |
| 2 edges chosen inside each training fold | **0.585** |
| the selection effect | **+0.165** |

Two thirds of the apparent gain was circularity. 0.585 is ordinary for this grid.

One detail here is a genuinely positive finding rather than a caveat: in-fold
selection recovers CON-SMN in **94 of 100 folds**, by a univariate criterion with no
connection to the attribution machinery. The feature is stable — which is the
paper's surviving claim, arrived at independently — but knowing which feature to use
still does not buy reliable classification, because the effect is too small at this n.

## 3. The specific proposals, assessed

**Other atlases and parcellation methods.** Now tested far beyond the paper's nine:
Schaefer at 100/200/400/1000 parcels and 7/17 networks, Yeo 7/17, Harvard-Oxford,
DiFuMo, MSDL, plus *data-driven* parcellations learned from the BOLD itself — ReNA
at 100/200/500 parcels, spatially-constrained Ward, and KMeans
(`scripts/build_parcellations.py`). Results in §4. Note that data-driven
parcellations fitted on all 32 participants are transductive; anything promoted from
them would need refitting inside folds.

**Better connectivity measures.** The six in the grid are near-redundant by
construction. `dynamic` (time-varying coupling) is the one genuinely new view tested,
and it is the best of the search at 0.626 — still the null's median.

**Deep learning.** Already done and reported: a per-timepoint autoencoder pretrained
on 1,100 windows of MPI-LEMON, fine-tuned in-fold, declared in advance as the
confirmatory primary. It reached **0.477** (p = 0.0999). The MLP is in every sweep
above and never wins.

**Pretrained GAN or diffusion models on resting-state.** The most interesting of the
suggestions, and worth being precise about what it could and could not do. Three
uses, with honest expectations:

- *Representation learning* — pretrain generatively on LEMON, use the latent as
  features. This is what the TimepointAE already does with a weaker model. A
  diffusion prior is a better density model, but the bottleneck is not the quality of
  the resting-state representation; it is that the label carries ~1 feature of
  information across 32 people.
- *Augmentation* — synthesise participants. This cannot work for the group contrast:
  a generative model trained on unlabelled rest has no information about the AVGP
  difference to inject, and one trained on the 32 labelled participants can only
  resample what is already there. Augmentation regularises; it does not create
  signal.
- *Normative modelling* — fit the generative model on LEMON as a "typical rest"
  prior, then use per-participant reconstruction error as a deviation score. This was
  the one variant flagged as genuinely untested and not obviously doomed. **It has now
  been tested; see §7.**

**SIMD and faster implementations.** Compute is not the constraint. The full
49-specification grid at 4 × 25 cross-validation refits in **13–20 seconds**; B = 1000
permutations took under three hours on a laptop. Speed buys larger nulls and more
thorough nested validation — both worth having — but not accuracy. The one place
vectorisation would pay is dynamic connectivity over many windows and parcels, which
is O(windows × parcels²).

## 4. Expanded parcellation sweep: 279 configurations, 24 parcellations

The full run (`outputs/explore/exploration.csv`) covers the nine cached atlases plus
Schaefer 100/200/400/1000 at 7 and 17 networks, Yeo 7/17, Harvard-Oxford, DiFuMo-256,
the data-driven ReNA-100/200/500, Ward-200 and KMeans-200, and one CompCor-denoised
variant.

**Best of 279: 0.681** (Ward-200, in-fold selection from 13,530 region edges,
XGBoost). That value sits at the **81st percentile of the null-maximum
distribution** — unremarkable for a search of this size, and below the 0.739 the
49-specification family already has to clear. A null computed over 279 paths rather
than 49 would put the bar higher still. It is also a data-driven parcellation fitted
on all 32 participants, so it carries the transductive caveat from §3.

Best per parcellation, static features, sorted:

| parcellation | parcels | best |
|---|---|---|
| dosenbach2010 | 160 | 0.588 |
| schaefer100-7 | 100 | 0.519 |
| schaefer200-7 | 200 | 0.516 |
| yeo7 | 7 | 0.502 |
| difumo64 | 64 | 0.499 |
| gordon2014 | 333 | 0.494 |
| rena100 | 100 | 0.489 |
| schaefer200-7 @compcor | 200 | 0.488 |

Three things this settles:

- **No parcellation beats the one already in use.** Schaefer, the field standard, is
  worse at every resolution tried; so is every data-driven parcellation.
- **Resolution hurts.** schaefer1000 reaches 0.422, schaefer400-17 reaches 0.395.
  More parcels means more edges against the same 24 training participants.
- **CompCor denoising does not rescue it** (0.488 vs 0.516 for the same
  parcellation). Scrubbing could not be run at all: it leaves one participant with
  16 usable volumes, which is its own comment on this sample.

## 5. What would actually work

In descending order of how much it would change the answer:

1. **More participants.** The only fix for the binding constraint. Marek et al.
   (2022) and, more favourably for extreme-group designs, Spisak et al. (2023),
   bracket what is needed; both put it well above 32.
2. **External validation.** Train here, test on an independent AVGP cohort. This
   escapes the search penalty entirely — a single prediction on new data needs no
   family-wise correction — and it is the only route on which the 0.760 figure could
   become legitimate rather than circular. It requires finding such a cohort.
3. **Longer acquisitions.** 6 min 15 s sits at the bottom of the reliability curve
   (Birn et al. 2013). This bounds every downstream analysis before any modelling
   choice is made.
4. **Pre-registration of one specification.** Already done once, honoured, and null.
   Doing it again with CON-SMN — now demonstrably the stable feature — is legitimate
   *only* on data not used to select it, i.e. it reduces to (2).
5. **Better denoising.** Motivated by the motion confound: gamers move more and
   motion alone decodes at 0.608. Aggressive scrubbing turns out to be impossible
   here — it leaves one participant with 16 usable volumes — which is itself a
   finding about this sample. CompCor is testable and included in §4.

## 6. The honest reframing

If the goal is to separate these two groups reliably, the data already do it: the
groups differ clearly in **behaviour** (accuracy 0.85 vs 0.73, inverse efficiency
1.12 vs 1.43) and detectably in **head motion** (0.608). What resting-state
connectivity supports at n = 32 is a *stable explanation* — CON-SMN, recovered by
independent criteria in 94–100% of folds — without *reliable prediction*.

That dissociation is the paper's finding. The work in this document was an attempt
to overturn it and it made the case stronger instead: 135 configurations, more than
twenty parcellations, feature families the grid never had, and a direct estimate of
the information available, all agree.


---

## 7. Normative modelling with a diffusion model: tested, and it does not work

The idea was the best of the generative suggestions, because a deviation score is
not a rearrangement of the covariance matrix the way the six connectivity
statistics are. Two participants with identical CON-SMN coupling can have different
deviation scores, since the score depends on the whole vector and on how an
external population covaries.

**What was built** (`acnets.deep.VectorDiffusion`, `scripts/run_normative.py`): a
DDPM with epsilon-prediction over the 21-edge connectivity vector — an MLP denoiser
with a sinusoidal timestep embedding, since at 21 dimensions a U-Net would be
silly. Trained on 1,100 windowed MPI-LEMON vectors from 220 external participants,
none labelled. Each Julia2018 participant is then scored by per-edge denoising
error averaged over 256 draws of (timestep, noise), giving a deviation vector in the
same 21-edge vocabulary as every attribution in the paper. Training loss fell from
1.006 to 0.331 — at 1.0 the model is predicting zero, so it learned the density.

Three controls, each answering a specific way the result could be fake:

| feature family | SVM | XGBoost | MLP |
|---|---|---|---|
| **raw** — the paper's baseline | 0.608 | **0.666** | 0.635 |
| **diffusion** — normative deviation | 0.519 | 0.516 | 0.495 |
| **gaussian** — Mahalanobis deviation from the same population | 0.532 | 0.636 | 0.531 |
| **diffusion-shuffled** — trained on LEMON with each edge permuted | 0.434 | 0.539 | 0.484 |
| **raw + diffusion** | 0.562 | 0.570 | 0.575 |

Four things follow, and none of them favour the method.

**The deviation score is at chance.** 0.495–0.519 across classifiers. Whatever
distinguishes these groups, it is not how atypical their connectomes are relative to
an external population.

**The diffusion model does not beat the classical control.** Gaussian normative
modelling — a mean, a shrunk covariance, a Mahalanobis distance — reaches 0.636 with
XGBoost against the diffusion model's 0.516. The deep density model bought nothing;
it did worse than the linear-algebra version of the same idea. That is the comparison
that decides whether the complexity is justified, and it is why the control was built.

**Combining hurts.** `raw + diffusion` (0.562–0.575) is *worse* than `raw` alone
(0.608–0.666). At n = 32, doubling the feature count to add uninformative columns
costs more than the columns could ever contribute.

**And the score is less independent than the argument for it assumed.** Median
|Spearman| between each edge and its own deviation score is **0.54** (max 0.84). The
premise was that a deviation score carries information the raw value does not; at
better than half the rank variance shared, it substantially does not. Worth noting
that the *undertrained* model from a smoke test showed a much weaker correlation
(median 0.22) — the better the density model fits, the more its reconstruction error
tracks distance from the population mode, and the closer it comes to being a
rearrangement of the raw features after all. That is a general point about normative
scoring, not a quirk of this run.

The family-wise null over all 15 specifications tried here is reported in
`outputs/normative/normative.json`.

**Conclusion.** The last untested idea on the list is now tested and joins the
others. This does not show that normative modelling is a bad method — it is a good
method, widely used on large clinical cohorts. It shows that at n = 32, with a
6-network summary of a 6-minute run, there is no atypicality signal to find, and
that the deep version of the method cannot beat its own two-line classical control.


---

## 8. Better classifiers and hyperparameter optimisation: no change

Thirteen model families instead of three, each with a search space, and
hyperparameters tuned by randomised search on an inner cross-validation *inside
each training fold* (`acnets.multiverse.classifiers`, `scripts/run_hpo.py`). The
families are biased toward the heavily regularised linear models that suit
n = 32 -- shrinkage LDA, ridge, L2 and elastic-net logistic regression, RBF-SVM --
with `qda` and `knn` included as families that should cope badly, so the
comparison has a floor above chance.

This also corrected a real defect. Methods 6.5 claimed hyperparameters were tuned
on training data; `runner._make_classifier` returned fixed values and never tuned
anything. The claim is now true rather than deleted.

**On the 21 network edges -- the legitimate feature set -- nothing beat the
original grid.**

| classifier | untuned | tuned | Δ |
|---|---|---|---|
| xgboost | **0.666** | 0.632 | −0.034 |
| random-forest | 0.644 | 0.635 | −0.009 |
| extra-trees | 0.638 | 0.620 | −0.017 |
| gnb | 0.636 | 0.636 | 0.000 |
| mlp | 0.635 | 0.644 | +0.009 |
| logistic | 0.620 | 0.595 | −0.025 |
| logistic-en | 0.620 | 0.614 | −0.006 |
| svm | 0.608 | **0.644** | +0.036 |
| svm-rbf | 0.595 | 0.596 | +0.001 |
| knn | 0.592 | 0.618 | +0.025 |
| ridge | 0.581 | 0.602 | +0.021 |
| lda | 0.561 | 0.630 | +0.069 |

The best of 26 legitimate specifications is **0.666** — identical, to three
decimals, to what the original three-classifier grid already produced. Ten more
model families bought nothing.

**Tuning is a coin flip.** Across 37 cells the mean change was **−0.003**, the
median −0.006, the range −0.034 to +0.069, and tuning helped in **13 of 37**.
Where it helped most (LDA, +0.069) it was repairing a family that was badly
configured by default rather than finding signal; where it hurt most (XGBoost,
−0.034) the inner search overfitted 24 training participants. Both are what
tuning does at this sample size.

**No new permutation null was run, and none is needed for this conclusion.** The
null-maximum over a superset of specifications is pointwise at least the
null-maximum over any subset, so the critical value of a family containing both
the original 49 specifications and these 26 is at least the 0.739 already
computed. The best observed value in the combined family is 0.666. It cannot
clear a bar that is at least 0.739, so no specification here is significant, and
a 30-hour null would only confirm arithmetic. (The `top2-circular` set reaches
0.760, but its two features were chosen on these data and it is not a legitimate
specification; see §2.)

**Conclusion.** "Maybe you used bad models" is now answered. Thirteen families,
tuned and untuned, on the same folds, reach exactly where three families
reached. At n = 32 the model is not the binding constraint.


---

## 9. Reaching 80%: the number is obtainable and it does not survive

Asked directly for at least 80%, by any pipeline. Two changes get there.

**Change the estimator.** Leave-one-out cross-validation trains on 31
participants per fold instead of the 24 that 4-fold gives, and at this sample
size that matters: connectivity alone rises from 0.666 to **0.750**.

**Change the input.** Four modalities exist per participant and only one had ever
been used (`scripts/run_multimodal.py`). Under 4-fold CV, 120 combinations of
modality and classifier top out at 0.700. Under LOOCV:

| input | n | best LOOCV | what it is |
|---|---|---|---|
| connectivity | 32 | 0.750 | brain measurement |
| behaviour | 32 | 0.812 | the trained skill that defines the groups |
| connectivity + behaviour | 25 | 0.760 | mixed |
| connectivity + motion + behaviour | 25 | **0.840** | mixed |

**So 84% is reachable. It is also not distinguishable from noise.**

Running the identical 32-cell search on permuted labels 200 times
(`scripts/run_loocv_null.py`):

```
observed max        0.840
null max, median    0.688
null max, 95th pct  0.840
p                   0.0647
```

The 95th percentile of the null maximum is **0.840** — the same value as the
observed maximum, to three decimals. Searching 4 modality combinations × 8
classifiers with a leave-one-out estimator at n = 25 produces 84% from random
labels about one time in twenty, and that is what we got. Nothing clears at
α = 0.05, and the single cell that touches the critical value is the one that
defines it.

This is worth being blunt about, because it is the trap the whole document is
about in its most seductive form. Every step to 0.840 was individually
defensible: LOOCV is a standard estimator, fusing modalities is standard
practice, LDA is a reasonable model, nothing was fitted on test data. The result
is still noise, because 32 cells of a high-variance statistic on 25 participants
produce large maxima by construction. Reported without the null — as a single
number in a table — 0.840 would look like a strong finding.

**And the honest reading is narrower still.** The 0.840 cell needs `behaviour`,
which is response accuracy and reaction time on an attention task. Action gamers
outperform non-gamers on attention tasks; that is the premise of the entire
literature and the reason the groups were recruited. Predicting group membership
from it restates the recruitment criterion rather than discovering anything about
the brain. The best *brain-measurement-only* result is connectivity at LOOCV,
0.750, which also does not clear.

**What would produce a real 80%.** Only more participants. At n = 25-32 the 95th
percentile of the null maximum sits between 0.74 and 0.84 depending on the
estimator and the size of the search, so the bar for significance is at or above
the target itself. Any number in that range is unfalsifiable at this sample size,
whichever way it comes out. Doubling n roughly halves the width of the null and
would make an effect of the size actually present here testable.

Ways to produce 80% that would be wrong, all of them available and none used
here: report training accuracy; select features on all data and cross-validate
afterwards (§2, worth +0.165); pick the best of several random seeds or fold
partitions; report the best cell of a search without its null (this section,
worth +0.15 over the null's median).

---

## 10. Reaching 100%, and what it costs

Asked to push as close to 100% as possible. It is reachable — exactly 100% — and
the only useful thing to do with that fact is measure what each step to it
destroys. `scripts/run_overfitting_ladder.py` climbs the ladder on the
region-level feature space (12,880 candidate edges, 32 participants), which is
where these shortcuts pay best: with enough candidate features, some will
separate any labelling by chance.

| rung | accuracy | what was done |
|---|---|---|
| **0  honest** | **0.541** | selection and scaling inside every training fold |
| 1  best cell | 0.560 | max over 8 classifiers, no null reported |
| 2  selection on all data | 0.762 | top 10 edges chosen on all 32, then cross-validated |
| 3  best of 20 seeds | 0.825 | best fold partition of 20 (median of those 20: 0.781) |
| 4  + scaling leakage, LOOCV | 0.812 | selection and scaling both fitted on everything |
| 5  no held-out data | **1.000** | accuracy on the training set itself |

**Stacking shortcuts manufactures +0.459 of accuracy on data whose honest
estimate is 0.541.**

Two observations are worth more than the numbers.

**No single step looks like fraud.** Rung 2 — the largest single jump, +0.202 —
is written by putting the feature selection above the cross-validation loop
instead of inside it. In a script that is three lines and it reads like
preprocessing. It is the most common way a small-sample neuroimaging paper
reports an accuracy that will not replicate, and it is almost never deliberate.

**The manufactured effect dwarfs the real one.** The entire question this study
asks — whether AVGPs and NVGPs differ in resting connectivity — concerns an
effect worth at most a few points of accuracy above chance. The methodological
shortcuts above are worth 46 points. Any of them individually is larger than the
signal being investigated, which is why the design has to be fixed in advance
rather than adjusted until the answer looks good.

This table is the argument for the rest of the project in one place: the
permutation nulls, the label-independent folds, the in-fold fitting, the
declared primary. Without them, this dataset supports any conclusion
between 0.54 and 1.00, and the reader has no way to tell which one they are
being shown.


---

## 11. Is there a theoretical ceiling? Yes — about 0.72 to 0.79, set by reliability

Four independent ceilings apply. The binding one is not sample size and not the
model (`scripts/run_ceiling_theory.py`).

### 11.1 Bayes ceiling — what any classifier could do with infinite data

For two Gaussians with shared covariance, no classifier ever beats Φ(D/2).

| feature space | D² naive | D² bias-corrected | ceiling |
|---|---|---|---|
| all 21 network edges | 5.87 | **−1.06** | 0.500 |
| CON-SMN alone (observed d = 1.17) | — | — | **0.721** |

The multivariate ceiling is not estimable here: after correcting the optimism
that comes from 21 features and 32 participants, no separation remains. The
single best edge does carry a real effect, d = 1.17, whose Bayes ceiling is
0.721 — though that edge was chosen on these data, so even this is optimistic.

### 11.2 Reliability ceiling — the one that actually binds

This is the ceiling nobody computes, and it is decisive. Functional connectivity
is measured with error: mean edge-level test-retest ICC = **0.29** (Noble,
Scheinost and Constable 2019). A noisy measure attenuates a latent effect by
√ICC, so what a scan can express is bounded *regardless of how large the true
biological difference is*.

| latent effect | ICC = 0.29 | ICC = 0.50 | perfect measurement |
|---|---|---|---|
| d = 0.8 (large) | 0.585 | 0.611 | 0.655 |
| d = 1.2 (very large) | 0.627 | 0.664 | 0.726 |
| d = 2.0 (enormous) | 0.705 | 0.760 | 0.841 |
| d = 3.0 (implausible) | 0.790 | 0.856 | 0.933 |
| d = 5.0 (near-perfect) | 0.911 | 0.961 | 0.994 |

**To reach 80% you would need a latent effect of d = 3.1. To reach 90%, d = 4.8.**

For scale: d = 3 means group distributions that barely overlap — taller-than-98%
-of-women territory for male height. No individual-differences effect in
cognitive neuroscience is that large. This is why published resting-state
decoding accuracies cluster in the 0.60–0.75 band across labs, tasks and
contrasts: it is not a coincidence and not a failure of imagination, it is the
reliability of the measurement showing through.

**A 6-minute scan cannot express an 80% effect, no matter who analyses it.**
Longer acquisitions raise ICC — that is the single most effective intervention
available, more than any modelling choice.

### 11.3 Finite-sample ceiling — more participants do help, up to a point

Training regularised LDA on subsamples and testing on the remainder:

| train n | accuracy | sd |
|---|---|---|
| 8 | 0.590 | 0.096 |
| 16 | 0.631 | 0.099 |
| 24 | 0.661 | 0.151 |
| 28 | 0.682 | 0.206 |

The curve is **rising**, at +0.0046 accuracy per participant, so this is not a
pure-noise regime — there is something to learn. Naive extrapolation puts 0.80
at roughly 53 training participants, but §11.2 says that extrapolation cannot
be right: the reliability ceiling sits below 0.80 and the curve must flatten
before reaching it. (Note the growing sd — at train n = 28 only four
participants remain to test on, so the right-hand points are unstable.)

### 11.4 Estimation ceiling — what is knowable at n = 32

| true accuracy | how it measures at n = 32 |
|---|---|
| 0.70 | [0.54, 0.86] |
| 0.80 | [0.66, 0.94] |
| 0.90 | [0.80, 1.00] |

At this sample size 0.70 and 0.90 are not reliably distinguishable. Even a real
90% classifier could not be *shown* to be one.

### 11.5 The synthesis

| ceiling | value | can it be raised? |
|---|---|---|
| reliability, ICC = 0.29 | ~0.72 at a very large latent effect | **yes — longer scans** |
| Bayes, best single edge | 0.721 | no, property of the effect |
| finite sample, n = 32 | ~0.68 and rising | yes — more participants |
| estimation, n = 32 | ±0.16 | yes — more participants |

The honest ceiling for this design is **roughly 0.72–0.79**, and the observed
0.666 is close to it. That reframes the whole exercise: the pipeline is not
leaving much on the table. The gap between 0.666 and the ceiling is small, and
the gap between the ceiling and 0.80 cannot be closed by analysis at all.

Power, for completeness: at the observed CON-SMN effect (d = 1.17), 80% power
needs **12 participants per group** — this study has 16, so it is adequately
powered *if* that effect is real. If the true effect is half the observed one,
which selection bias makes likely, it needs **46 per group**.


---

## 12. Reaching the ceiling — and what is found there

Section 11 predicted a ceiling of roughly 0.72. `scripts/run_reach_ceiling.py`
aims at it with three changes, each motivated by the diagnosis rather than by
trying things: leave-one-out cross-validation (the learning curve's slope
predicts +0.03 from the extra training data), analytic covariance shrinkage
(Ledoit-Wolf and OAS, the correct estimators when the sample covariance is
unusable at p close to n, and never previously tried), and ensembling across
connectivity statistics instead of choosing among them.

**The prediction was accurate to 0.002.**

| cell | LOOCV accuracy | Wilson 95% |
|---|---|---|
| **precision / LDA-OAS** | **0.719** | [0.55, 0.84] |
| precision / LDA-Ledoit-Wolf | 0.719 | [0.55, 0.84] |
| ensemble / SVM | 0.688 | [0.51, 0.82] |
| precision / logistic | 0.688 | [0.51, 0.82] |
| tangent / LDA-OAS | 0.656 | [0.48, 0.80] |
| partial correlation / LDA-OAS | 0.625 | [0.45, 0.77] |

0.666 → **0.719**, against a predicted Bayes ceiling of 0.721 for the best single
edge and 0.705 for the reliability ceiling at a very large latent effect. The
analytic shrinkage discriminant did exactly what the theory said it would: it is
the appropriate estimator for this regime and it recovers essentially all of the
available signal.

Ensembling across statistics did **not** help (0.688 at best, below the single
best statistic). The statistics evidently share their noise as well as their
signal, so averaging them buys less than the theory of ensembling assumes.

### And here is the finding

```
observed max        0.719
null max, median    0.594
null max, 95th pct  0.720
p                   0.1118
0 of 30 cells clear the critical value
```

**The ceiling of what this data can express, 0.719, and the 95th percentile of
what the same search produces from randomly permuted labels, 0.720, are the same
number.**

That coincidence is the deepest statement of the problem this project has
arrived at, and it is worth stating carefully because it is not a coincidence at
all. Both quantities scale with the same things. The expressible effect is
bounded by measurement reliability and by n. The searchable noise is bounded by
n and by the size of the search. At n = 32 with ICC = 0.29 and a family of 30
specifications, the two bounds meet at 0.72.

The consequence: **this design cannot produce a significant decoding result even
if the true effect is exactly as large as it could possibly be.** Not "did not",
but *cannot*. An experiment whose maximum achievable effect coincides with its
own significance threshold is not underpowered in the ordinary sense that more
analysis might rescue -- it is unfalsifiable by construction, and no model,
feature set, or amount of computation changes that.

This is the answer to "can accuracy be improved to reach the ceiling". Yes, and
it now has been: 0.719, from 0.666, by using the statistically correct estimator.
The gain is real and the method is sound. It also does not, and provably could
not, clear its own null.

What breaks the deadlock is changing one of the two bounds, and only the
experiment can do that: more participants lowers the null and raises the
achievable accuracy simultaneously, and longer scans raise ICC, which raises the
ceiling alone. Both are acquisition decisions. Neither is an analysis decision.


---

## 13. Pretraining that transfers reliability, not representation

LEMON pretraining had been tried three ways and helped none of them: as a
tangent reference and scaler (1.6% of accuracy variance), as a per-timepoint
autoencoder (the transfer model of section 2.4, 0.477), and as a diffusion normative
model (§7, at chance). All three transferred a *representation*, and section 11
says representation is not the constraint — reliability is. So this transfers
reliability instead (`scripts/run_reliability_transfer.py`).

### 13.1 The ceiling analysis, verified rather than cited

Section 11 borrowed ICC = 0.29 from Noble et al. (2019) — a meta-analytic
average over other scanners and run lengths. It can be measured here directly:
split each run in half, compute connectivity on each half, correlate the halves
across participants, and apply the Spearman-Brown correction back to full length.

| statistic | Julia2018 | LEMON (truncated to 124 tp) |
|---|---|---|
| correlation | 0.707 | 0.367 |
| partial correlation | 0.625 | 0.230 |
| precision | 0.557 | 0.235 |
| tangent | 0.555 | 0.330 |

**Median LEMON reliability: 0.28, against the literature's 0.29.** The ceiling
analysis rests on the right number, now measured rather than assumed.

Julia2018's own values are much higher, and that is an artefact rather than good
news: split-half reliability is reproducible between-participant variance over
total variance, and a sample containing two deliberately contrasting groups has
extra between-participant variance by construction. With n = 32 the estimate is
also unstable. LEMON — 220 participants, homogeneous, unselected — is the clean
estimate, which is why its value is the one used.

### 13.2 Weighting features by how reliably they are measured

Reliability estimated on LEMON is external and label-free, so it can rescale the
target's features without leakage. Down-weighting an edge measured at ICC 0.05 is
principled: it contributes almost pure noise to a discriminant that has very few
degrees of freedom to spend.

| cell | LOOCV | Wilson 95% |
|---|---|---|
| **precision / reliability-weighted** | **0.750** | [0.58, 0.87] |
| precision / plain | 0.719 | [0.55, 0.84] |
| precision / reliable-edges-only | 0.719 | [0.55, 0.84] |
| partial correlation / reliability-weighted | 0.688 | [0.51, 0.82] |
| tangent / reliability-weighted | 0.562 | [0.39, 0.72] |
| correlation / reliable-edges-only | 0.531 | [0.36, 0.69] |

**0.719 → 0.750**, the best honestly cross-validated result this project has
produced, and *above* the reliability ceiling computed for a very large latent
effect (0.705). Continuous weighting beats hard thresholding, which is expected:
discarding an edge throws away whatever signal it carries along with its noise.

The gain is not uniform — tangent gets worse — so this is a real effect with a
real mechanism, not a universal improvement.

> **A bug worth recording, because it nearly hid the result.** The first version
> applied the weights to the raw features, before the pipeline's
> `StandardScaler`. The scaler divides each column by its own standard deviation
> and therefore exactly undoes any diagonal rescaling: the weighting was a no-op.
> It showed up as weighted and unweighted results identical to three decimal
> places across every statistic, which is too tidy to be real. Weights now apply
> after standardisation.

### 13.3 The null, again

```
observed max        0.750
null max, median    0.594
null max, 95th pct  0.750
p                   0.0539
1 of 12 cells clears it
```

**p = 0.054.** The closest this project has come, and still not below 0.05 —
with, once again, the critical value landing exactly on the observed value. The
phenomenon of section 12 repeats: the best the data can express and the best the
search can manufacture from noise remain the same number, because both are
governed by n.

### 13.4 What this says about pretraining

Pretraining on external data **does** work here, where three previous attempts
did not, and the difference is what was transferred:

| transferred | result |
|---|---|
| covariance reference and scaling | 1.6% of variance, no accuracy gain |
| learned representation (autoencoder) | 0.477 |
| learned density (diffusion, §7) | at chance |
| **per-edge measurement reliability** | **+0.031, best result in the project** |

The lesson generalises beyond this dataset. External data helps most when it
supplies something the target sample cannot estimate for itself. With 32
participants, a representation is not that thing — the target has enough data to
estimate a 21-dimensional covariance badly, and an external representation
estimates the same badly-conditioned object. Per-edge reliability *is* that
thing: it requires within-participant repeated measurement, which the target has
only 124 timepoints of, and which 220 external participants can estimate
properly.

The practical form of the advice: when transferring, ask what the target sample
is structurally unable to measure, not what a bigger model could represent.


---

## 14. A larger pretraining corpus: tested, and it makes things worse

Section 13 ended with a recommendation: reliability weighting worked, its limit
was how precisely reliability could be estimated from LEMON's 220 participants,
so a larger corpus should help. **That recommendation was wrong, and testing it
is how the interesting part was found.**

**HCP was the obvious candidate and is the wrong one.** Access needs a
ConnectomeDB account and an accepted data use agreement. More importantly,
reliability is a property of an acquisition, not of a brain: HCP is TR = 0.72 s,
1,200 timepoints, multiband, against a target at TR = 3 s, 124 timepoints, 4 mm
single-band. ABIDE was used instead — openly accessible, roughly 5x LEMON, sites
running TR 1.5-3 s at 120-300 timepoints, and crucially shipping timeseries
already extracted with the **Dosenbach 160 atlas this project uses**. Controls
only, since a two-group sample inflates split-half reliability (§13.1).

One check before anything was built on it: ABIDE's columns are numbered `#1..#161`
with no labels, so mapping them onto this project's region-to-network assignment
was an assumption. Verified — within-network r = 0.185 against between-network
0.104, correct in all six networks. The check is now a hard failure in the script.

### 14.1 The result

| | plain | LEMON-weighted | ABIDE-weighted |
|---|---|---|---|
| precision | 0.719 | **0.750** | 0.688 |
| correlation | 0.625 | 0.625 | 0.594 |
| partial correlation | 0.625 | 0.688 | 0.594 |
| tangent | 0.656 | 0.562 | 0.594 |

**Every statistic got worse with weights from the larger corpus.** The best cell
falls from 0.750 back to 0.719 (p = 0.076). More data produced a worse weight
vector.

### 14.2 Why: the corpora disagree, and the reason decomposes

| statistic | LEMON | ABIDE | per-edge agreement |
|---|---|---|---|
| correlation | 0.367 | 0.732 | r = 0.25 |
| partial correlation | 0.230 | 0.570 | r = 0.35 |
| precision | 0.235 | 0.676 | r = 0.69 |

ABIDE's reliability runs 2-3x higher, and the two disagree about *which* edges
are reliable — at r = 0.25 for correlation, the ranking that the weighting
depends on is barely shared at all.

Decomposing the gap by testing each candidate:

| source | effect on ABIDE precision reliability |
|---|---|
| multi-site pooling | 0.676 pooled → 0.600 within single sites |
| no band-pass filter | 0.701 unfiltered → 0.552 band-passed |
| + global signal regression | 0.552 → 0.581 |
| **residual gap to LEMON's 0.235** | **~0.32, unexplained by either** |

Preprocessing is the largest identified single source — band-pass filtering
alone removes 0.15, because unfiltered data carries slow drift and
respiration-linked signal that reproduces across both halves of a run without
being neural. Site pooling contributes about 0.08. But **the two together do not
close the gap**: filtered, single-site ABIDE would still sit far above LEMON.
The most plausible remaining source is population — ABIDE's controls span
childhood to middle age, and developmental variance is real reproducible
between-participant variance that a homogeneous adult sample does not have.
That is a legitimate difference rather than an artefact, which is precisely why
it makes ABIDE's weights inapplicable here.

### 14.3 The lesson

Reliability is a property of **acquisition x preprocessing x population**, not of
an atlas. Matching the parcellation and the run length — which this analysis did
deliberately — was not close to sufficient. LEMON's value was never its size. It
was that it went through *this project's own pipeline*, so its reliability
estimate is calibrated to the quantity the target actually measures.

This sharpens the §13 conclusion rather than overturning it. External data helps
when it supplies something the target cannot estimate for itself — but only if it
supplies the *same* something. A corpus five times larger and mismatched on
preprocessing is worse than a corpus half the size and matched. HCP would be
mismatched on four axes rather than one.

The route that would actually work is not a bigger fetcher call: it is a corpus
**reprocessed through this project's pipeline from raw or fmriprep-level data**,
rather than consumed as pre-extracted timeseries. ABIDE's raw data supports that,
as does HCP once the data use agreement is in place. That is a substantial piece
of work and should be chosen deliberately, not taken as an obvious next step.

**Best honest result remains 0.750** — precision, LEMON reliability-weighted,
LOOCV, p = 0.054.


---

## 15. Inductive bias: the first result that clears its null, and its one weakness

Every earlier attempt estimated a discriminant direction in 21 dimensions from
32 participants, which section 11 shows cannot work. An inductive bias does
something different: it *constrains* which directions are allowed, using
knowledge not in the data, and so spends fewer degrees of freedom on the sample.
This is the only lever tried in sections 1-14 that is not more search.

| bias | LOOCV | what it assumes |
|---|---|---|
| **discovered-contrast** | **0.812** | direction fixed, 0 df estimated — but fixed *from these data* |
| **network-restricted** (FPN, CON, SMN → 6 edges) | **0.812** | *where*, not direction — same provenance |
| laplacian-smooth | 0.750 | edges sharing a node have similar weights |
| baseline shrinkage LDA | 0.719 | nothing |
| rank-one | 0.719 | the difference is one network mode |

0.719 → **0.812**, the largest gain of anything in this document, and above the
reliability ceiling of 0.705 computed for a very large latent effect.

### 15.1 One version dies under scrutiny

The discovered-contrast fixes the direction on CON-SMN and FPN-SMN, and gives
p = 0.002 tested alone. That test is valid only if the pair really was chosen a
priori. It was not: those two edges are the top of *this dataset's* attribution
ranking (§3.1). The Gong et al. (2015) justification is weaker than it first
appears — their finding was insula-centric, and this project's own reanalysis of
their data did not recover the insula (§3.6).

Testing it properly: **the pair ranks 2 of 210 possible pairs**, and a
max-statistic null over all 210 gives **p = 0.073, not significant**. Rank 2 of
210 is what a data-informed choice looks like. The claim is withdrawn.

### 15.2 The network-restricted version survives full correction

Choosing three networks out of six is a 20-way choice, not a 210-way one, and
the statistic is a 4-way choice. Correcting for **both**, over all 80 cells:

```
best cell         0.812   partial correlation / CON+DMN+SMN
discovered triple 0.812   precision / CON+FPN+SMN, rank 1 (tied)
null max median   0.688     95th pct 0.781
p                 0.0399    SIGNIFICANT
```

**This was recorded as the first result in the project to clear a family-wise
null. That reading does not survive, and §15.1's argument is why.**

§15.1 kills the *contrast* because the pair of edges came from this dataset's
own attribution ranking. The identical objection applies one level up, and was
missed here for longer than it should have been: enumerating the 20 possible
triples corrects for a choice among triples only if the triple was chosen
blind. FPN+CON+SMN was not. It is the triple containing both edges that §3.1
ranked first and second — chosen, like the pair, by looking. A 20-way
enumeration is the right correction for someone who picked a triple at random
and is the wrong correction for someone who picked *this* one.

So p = 0.0399 is not a licensed number either. What follows is kept because the
bounds it establishes are real and useful, but the headline is withdrawn:
nothing in this project clears a family-wise null on a hypothesis it did not
first read off the data.

Three further things bound it independently.

**The correction covers statistic and triple, not the path.** LOOCV over 4-fold,
shrinkage LDA with OAS, network scale over region scale, dosenbach2010 over nine
other atlases — all were chosen earlier on these same data. The 80-cell family is
the right correction for the final step and an incomplete one for the analysis as
a whole.

**The discovered triple is tied, not uniquely best.** `partial correlation/CON+DMN+SMN`
and `FPN+SMN+VIS` also reach 0.812. What every high-scoring triple shares is
**SMN**; the partners are interchangeable. So the finding is about sensorimotor
coupling, not about the specific control networks theory named — which is a
weaker and different claim than the one the bias was built to test.

### 15.3 And the motion confound bites exactly here

Section 3.5 established that action gamers moved more and that motion correlates
with sensorimotor coupling at r ≈ 0.5. A six-edge, SMN-centric model is precisely
where that matters. Regressing framewise-displacement SD out of each edge and
reclassifying:

| statistic | plain | motion-controlled | Δ |
|---|---|---|---|
| precision | 0.812 | 0.688 | **−0.125** |
| partial correlation | 0.750 | 0.625 | **−0.125** |

*(motion alone, same folds and estimator: 0.594)*

**A quarter of the gain does not survive motion control**, and 0.688 is below the
0.781 critical value. The corrected-significant result is therefore not robust to
the confound the design already knew about.

### 15.4 Where this lands

Inductive bias is the right answer to the question, and it works: constraining
the hypothesis space with prior knowledge bought more than every model,
parcellation, pretraining scheme and generative prior in this document combined.
The mechanism is sound and general — at n = 32 the scarce resource is degrees of
freedom, and a prior spends none of them.

What it produced here is **a strong pre-registration candidate, not a finished
claim**: precision connectivity, restricted to FPN/CON/SMN, shrinkage LDA,
leave-one-out — 0.812, p = 0.040 corrected, falling to 0.688 under motion
control. Every element is now specified precisely enough to be declared in
advance and tested once on an independent sample, where it would need no
correction at all and where the motion confound could be addressed by design
rather than by regression.

That is a materially better position than this document started from. It is not
a decoding result, and it should not be reported as one.


---

## 16. Cross-dataset structural validation: feasibility check FAILED, and it found a bug

The plan: put Julia2018's T1 images through the *same* ANTs pipeline as
gong2015, so both datasets land in one feature space, then train on one and
predict the other. External validation needs no multiverse correction and is the
one route that converts a Tier 3 claim to Tier 1 without pre-registration.

A cheap check first — 6 Julia2018 participants rather than all 32
(`scripts/segment_t1.py`, generalised from `gong2015_segment.py`). It failed,
and the failure is informative.

### 16.1 The check

The two datasets should produce nearly identical *regional profiles*: the same
brain regions are large or small in everyone, so mean normalised volumes across
160 regions should correlate near 1 between any two samples.

| comparison | profile correlation |
|---|---|
| within gong2015, split-half | **0.977** |
| within Julia2018 (n = 6), split-half | 0.784 |
| **across the two datasets** | **−0.401** |

Both datasets are internally consistent and they disagree with each other, so
this is a systematic mismatch rather than noise. The regions driving it flip
sign cleanly — labels 57, 58, 71 sit at z = +4.7, +3.9, +3.3 in gong2015 and
−2.7, −2.7, −2.7 in Julia2018.

### 16.2 The cause: skull stripping, and it is a defect in the published arm

Orientation matched (all det +1, same direction cosines), so it was not a flip.
Brain extents did not match:

| | brain extent (mm) | thresholded volume |
|---|---|---|
| Julia2018 | 147 x 164 x 156 | 1,566 cm³ |
| gong2015 | 190 x 191 x 152 | 2,446 cm³ |

A brain is roughly 140 x 170 x 130 mm. gong2015's images still contain skull and
scalp; fmriprep's Julia2018 output does not. The pipeline's `ants.get_mask` --
an intensity heuristic, not a brain extractor -- does not remove them:

| | mask volume | middle Atropos class |
|---|---|---|
| Julia2018 | 1,524 cm³ (plausible brain) | 615 cm³ (plausible GM) |
| gong2015 | **2,127 cm³ (includes skull)** | **785 cm³ (inflated)** |

So for gong2015 the three-class k-means is not separating CSF/GM/WM inside a
brain; it is separating background, skull-and-scalp, and brain inside a head.
**The middle class -- taken as grey matter throughout the gong2015 arm -- is
contaminated.**

### 16.3 What this means for section 3.6

The gong2015 structural arm's *decoding* result (best 0.516, p = 0.699) is
unlikely to change: a null computed on contaminated features is still a null,
and contamination adds noise rather than group structure.

Its *attribution* result is a different matter and should be treated as
withdrawn pending a rerun. Section 3.6 reports that the insula was not recovered
and that **sensorimotor cortex leads** — and sensorimotor cortex sits near the
vertex, exactly where scalp and skull contamination is largest. That claim
cannot be separated from the artefact as things stand.

### 16.4 The fix

Replace `ants.get_mask` with a real brain extraction — `ants.brain_extraction`
with a template prior, or run both datasets from raw T1 through the same
extraction — then re-run gong2015's 47 participants and Julia2018's 32. Only
then is the cross-dataset comparison meaningful.

Cost is roughly two hours of unattended compute at the ~2 min/participant this
check measured. The feasibility check cost about 40 minutes and prevented
spending that on an analysis whose feature spaces did not correspond, while
surfacing a defect in an already-written result. That is what these checks are
for.


---

## 17. Murphy ch.19 "Learning with Fewer Labeled Examples", assessed and tested

The chapter is the right place to look — the design's whole problem is too few
labelled examples. Most of it still does not apply, and the reason is the same
each time: these methods need either **an unlabelled pool from the target
distribution** or **a distribution of related tasks**, and this design has
neither.

| section | verdict |
|---|---|
| 19.1 data augmentation | applies — tested below |
| 19.2 transfer learning | already tested four ways (§7, §13, §14); reliability transfer is the only one that helped |
| 19.2.5 domain adaptation | the right tool for gong2015 ↔ Julia2018, blocked on §16 |
| 19.3 semi-supervised | **not applicable.** LEMON and ABIDE are unlabelled but are not drawn from a gamer/non-gamer population; a pseudo-label for them denotes nothing |
| 19.4 active learning | **not applicable.** Requires acquiring labels. Group membership is known for all 32 |
| 19.5 meta-learning | **not applicable.** Requires many related tasks; there is one |
| 19.6 few-shot | **not applicable.** 16 per class is the whole dataset, not a support set drawn from a class distribution |
| 19.7 weakly supervised | applies, and fits unusually well — tested below |

### 17.1 Augmentation and multiple-instance learning

Weak supervision maps onto the data almost exactly: each participant is a
**bag**, their time windows are **instances**, and the label exists only at bag
level. Each 124-timepoint run was cut into five overlapping 60-TR windows,
giving 160 instances over 32 bags, every window inheriting its participant's
label. Splits are at *participant* level — no window of a held-out participant
is ever seen in training, without which this is an elaborate way to leak.

| | LOOCV |
|---|---|
| **baseline, whole run** | **0.812** |
| MIL, majority vote | 0.781 |
| MIL, mean pooling | 0.750 |
| MIL, max pooling | 0.719 |

**Gain over the baseline: +0.000.** Neither augmentation nor weak supervision
helps, and the mechanism is worth stating because it generalises.

Averaging the whole run is *already* the optimal estimator of a stationary
connectivity matrix. Murphy's justification for augmentation (19.1.2) is that it
approximates an expectation over a transformation group, enforcing an invariance
the model does not otherwise exploit — but here temporal averaging **is** the
model, so windowing only makes each estimate noisier (60 timepoints instead of
124) and pooling cannot recover what averaging the data already did.

That max pooling does *worst* is the informative part. Max pooling encodes the
hypothesis that only some windows carry the signal — a plausible one for a state
that comes and goes across a resting run. It loses, which is evidence the group
difference is **stationary across the run** rather than episodic. That is a
substantive finding about the phenomenon, obtained from a method that failed to
improve accuracy.

### 17.2 The general lesson

Every technique in the chapter buys sample efficiency by importing structure:
unlabelled data from the same distribution, a task distribution, an invariance,
or a bag structure. This design can supply only the last two, and both turn out
to be already exploited by the existing pipeline. The one import that *did* work
— section 15's inductive bias, worth +0.09 — is the one the chapter treats least
directly: constraining the hypothesis space with domain theory.


---

## 18. The brain-extraction fix, and why it did not make the datasets comparable

Two separate problems were conflated in section 16. One is now fixed; the other
is not, and is deeper than it looked.

### 18.1 Fixed: skull contamination in the gong2015 volumes

| | before | after |
|---|---|---|
| brain mask | 2,127 cm³ | **1,755 cm³** |
| "grey matter" | 785 cm³ | **656 cm³** |
| GM / mask | 0.37 | 0.37 |

`ants.get_mask` is replaced by warping a conservative MNI brain mask through the
registration, and `segment_t1.py` now records `brain_mask_cm3` per participant
and warns outside 900-2,000 cm³. GM volumes are now in the plausible 550-750
range, so the three-class Atropos step is finally separating CSF/GM/WM inside a
brain rather than background/skull/brain inside a head.

This took two attempts. The first used nilearn's default MNI brain mask, which
is *liberal* — 1,883 cm³ at threshold 0.2, roughly 30% larger than a brain — and
an affine warp onto a larger head inflated it further, returning masks of 2,200
cm³, no better than what it replaced. Rebuilt at threshold 0.6 (1,493 cm³) it
works. **The failure was only visible because of the volume field and range
warning added at the same time**; without them the fix would have looked
successful and quietly produced the same wrong volumes.

### 18.2 Not fixed: the two datasets still do not share a feature space

| pipeline | profile correlation |
|---|---|
| original | −0.401 |
| + corrected skull stripping | −0.388 |
| + brain-to-brain registration | **−0.406** |

Three substantially different pipelines, and the number does not move. That
stability is the diagnostic: **masking and registration are not the cause.** If
they were, changing both would have moved it.

Nor is it a subtle mismatch. Taking the 16 regions in each dataset's lowest
volume decile, **exactly 1 of 16 is shared**. The two datasets disagree about
which brain regions are small almost completely.

The remaining candidate is acquisition coverage: gong2015's field of view is
256 x 256 x 152 at 1 mm and the brain fills the entire slice direction, so it is
probably truncated where Julia2018's 192 x 223 x 200 is not. Regions in the
truncated band would get near-zero volume in one dataset and normal volume in
the other, and normalising by total grey matter turns that into exactly the
anti-correlation observed. This was not tested — the atlas label file carries
`label, region, network` but no coordinates, so the z-gradient test needs the
atlas image rather than the CSV.

### 18.3 Recommendation

Stop treating this as a fix to the existing pipeline. Two honest options:

1. **Reprocess both datasets from raw with one validated tool** — FastSurfer or
   SynthSeg on both, rather than an ANTs pipeline tuned per dataset. This removes
   the whole class of problem, including truncation handling, at the cost of a
   few hours of compute.
2. **Accept that gong2015 and Julia2018 cannot be put in a common structural
   space** with the data as it stands, and keep the gong2015 arm as the separate
   replication it already is.

Either way, section 3.6's *attribution* claim — that sensorimotor cortex leads —
should be re-run against the corrected volumes before it is reported, since the
old ones were skull-contaminated and sensorimotor cortex sits where the
contamination is worst. Its decoding null is unlikely to change.


---

## 19. It was the atlas: sphere parcellations are unusable for cross-dataset volumetry

Three pipelines failed to make gong2015 and Julia2018 agree (§18). The cause was
none of the things being changed. Both datasets had *also* been processed with
AAL the whole time, and that table was sitting on disk untested:

| atlas | kind | cross-dataset profile correlation |
|---|---|---|
| dosenbach2010 | 160 spheres at coordinates | **−0.406** |
| aal | 166 contiguous parcels | **+0.872** |

Same participants, same segmentation, same registration, same normalisation.
Only the atlas differs.

**Why.** Dosenbach 2010 places ~10 mm spheres at published coordinates. Under an
affine registration between two different acquisitions, a sphere can miss its
target, land in white matter, or sit half outside the brain — so the grey-matter
volume it captures is close to arbitrary, and arbitrary in a way that differs
systematically between datasets. AAL's parcels are large enough that a few
millimetres of misregistration barely changes what they contain.

Sphere atlases are perfectly reasonable for functional connectivity, where a
timeseries is averaged over whatever the sphere contains and displacement mostly
adds noise. They are the wrong tool for regional volumetry, and more so across
datasets.

**Two consequences.**

The cross-dataset structural comparison is **feasible** after all. With a
volumetric atlas the two datasets share a coherent feature space, which is what
external validation needs.

And section 3.6's gong2015 arm used dosenbach2010 volumes, so it rests on the
unreliable measurement. Combined with the skull contamination of §18.1, its
attribution result — that sensorimotor cortex leads — has two independent
reasons to be re-run before it is reported. Its decoding null is unlikely to
move.

### 19.1 Full-Python pipeline, no FreeSurfer

There was never a FreeSurfer dependency and there is none now.
`scripts/prepare_atlases.py` fetches volumetric atlases with nilearn and stages
them as NIfTI plus a label table; `scripts/segment_t1.py` reads those and does
N4 bias correction, brain extraction, affine registration and Atropos tissue
segmentation with antspyx. Both are pip-installable. The two environments exist
only because antspyx pins an older scipy.

Staged and verified:

| atlas | parcels | median parcel |
|---|---|---|
| aal | 166 | 5.9 cm³ |
| harvard-oxford | 48 | 14.5 cm³ |
| schaefer100 | 100 | 9.8 cm³ |
| schaefer200 | 200 | 5.0 cm³ |
| yeo7 | 7 | 155.3 cm³ |
| yeo17 | 17 | 63.8 cm³ |

`prepare_atlases.py` prints the median parcel volume and flags anything
sphere-like, so the mistake this section documents cannot be repeated silently.
Schaefer is the most useful addition: it carries Yeo network labels, so the
network-restricted inductive bias of §15 transfers directly to the structural
arm.


---

## 20. External validation between two independent samples: a clean null

All 79 participants (47 gong2015, 32 Julia2018) segmented through one pipeline,
three volumetric atlases, zero quality warnings. The feature spaces now
correspond, so a transfer result finally means something.

### 20.1 Parcel size governs cross-dataset agreement

| atlas | median parcel | profile correlation |
|---|---|---|
| harvard-oxford | 14.5 cm³ | **+0.936** |
| aal | 5.9 cm³ | +0.867 |
| schaefer200 | 5.0 cm³ | +0.581 |
| dosenbach2010 | sphere, <1 cm³ | **−0.406** |

The ordering is monotone in parcel size and spans the entire range from perfect
disagreement to near-perfect agreement. This is the mechanism of section 19
confirmed on full samples: under affine registration between different
acquisitions, small parcels sample essentially arbitrary tissue, large ones do
not. Anyone doing cross-dataset volumetry should read this table before choosing
a parcellation.

### 20.2 The external validation

Train on one dataset, predict every participant of the other. No multiverse
correction applies — nothing about the target was consulted while fitting — and
the null shuffles labels *within the training dataset only*.

| atlas | gong2015 → Julia2018 | Julia2018 → gong2015 |
|---|---|---|
| harvard-oxford | 0.500 (p = 0.74) | 0.489 (p = 0.82) |
| aal | 0.500 (p = 0.78) | 0.511 (p = 0.51) |
| schaefer200 | 0.500 (p = 0.73) | 0.489 (p = 0.80) |

**Six independent tests, every one at chance** — 0.489 to 0.511, against null
medians of 0.500 to 0.511. Not a marginal miss; the transfer accuracy is
indistinguishable from the null in every direction and every atlas.

Within-dataset accuracy is no better: gong2015 reaches 0.574 and Julia2018
0.281–0.438, the latter *below* chance. Regional grey-matter volume does not
separate these groups inside either dataset, so there was never a model to
transfer.

### 20.3 Why this null is worth more than the earlier ones

Every previous negative in this document could be answered with "your features
were wrong". This one cannot:

- the feature spaces are **verified** to correspond (r = +0.87 to +0.94), by a
  check that fails loudly below 0.5 rather than being asserted
- three parcellations spanning 48 to 200 regions agree
- both transfer directions agree
- the pipeline defects that motivated the rerun — skull contamination (§18.1)
  and the sphere atlas (§19) — are fixed, and the corrected within-gong2015
  accuracy (0.574) is *higher* than the contaminated original (0.516), so the
  null is not an artefact of the corrections either

**Three explanations remain and this design cannot separate them.** The
contrasts differ — gong2015 is expert versus non-expert gamers, Julia2018 is
player versus non-player — so a shared structural signature may not exist even
if each dataset has one. Residual site effects survive normalisation. Or there
is no structural group difference to find. The third is the most parsimonious
given that within-dataset decoding also fails.

### 20.4 Consequences for the manuscript

Section 3.6's structural arm is superseded. Its decoding null stands and is
strengthened — 0.574 corrected against 0.516 contaminated, both null — but its
attribution result ("sensorimotor cortex leads") should be **deleted rather than
re-run**: attribution ranks from a classifier at chance describe noise, and this
one was additionally computed from skull-contaminated volumes under an unusable
atlas.

What replaces it is stronger. An external validation between two independent
action-gaming samples, in a verified shared feature space, across three
parcellations and both directions, is a far better test of Gong et al.'s
structural claim than the single-dataset reanalysis it replaces — and it is
Tier 1 publishable exactly as it stands, with no pre-registration required,
because a failed prediction on held-out data needs no correction.


---

## 21. Pooling the two structural samples: n = 79, still null

Treating gong2015's experts as players and its non-experts as non-players gives
**40 players and 39 non-players** across two sites — 2.5x either dataset alone,
against a binding constraint that has been n throughout.

**Site is not confounded with group**, which is what makes pooling defensible:
gong2015 is 24/23 and Julia2018 16/16, so neither site can proxy for the label.
What site does do is dominate the variance. A classifier predicts *site* at
**0.96-0.99**, so harmonisation is not optional; site means and scales are
removed inside each training fold, never using the held-out participant to
correct itself.

| atlas | site predictable | naive | harmonised | p |
|---|---|---|---|---|
| aal | 0.987 | 0.468 | 0.456 | 0.66 |
| harvard-oxford | 0.962 | 0.430 | 0.468 | 0.67 |
| schaefer200 | 0.962 | 0.456 | **0.544** | 0.27 |

Every condition is at or below chance. The best cell, Schaefer-200 harmonised at
0.544, sits well inside its null (median 0.494, 95th percentile 0.620).
Harmonisation helps where it should — Harvard-Oxford +0.038, Schaefer +0.088 —
which shows the correction is doing real work, and it still does not produce a
signal.

### 21.1 Why this null is genuinely ambiguous, unlike section 20's

Section 20's failed transfer had three candidate explanations. This one has a
fourth, and it is the one the analysis itself introduced.

**gong2015's "non-experts" are amateur action gamers. They play.** Merging them
with Julia2018's non-players assumes action-game exposure is one dimension along
which "amateur" sits nearer "non-player" than "expert". If that is wrong — if
amateurs resemble experts — then pooling did not add 47 participants to a clean
contrast, it added 47 participants to a *blurred* one, and a null is the
expected outcome whether or not an effect exists.

This is not a reason to distrust the result so much as a reason to state what it
can and cannot bear. It cannot bear "there is no structural difference between
action gamers and non-gamers". It can bear "under the assumption that amateur
gamers group with non-players, a 79-participant two-site sample shows no
structural separation" — which is a narrower claim and still worth reporting.

### 21.2 Where the structural arm now stands

| analysis | n | result |
|---|---|---|
| gong2015 alone, corrected | 47 | 0.574, null |
| Julia2018 alone | 32 | 0.438, null |
| external validation, both directions x 3 atlases | 47 → 32 | 6/6 at chance (§20) |
| pooled, harmonised | 79 | 0.544, p = 0.27 |

Five independent ways of asking, one answer. Regional grey-matter volume does
not separate these groups within either dataset, does not transfer between them,
and does not separate them pooled. Against the functional arm's 0.719-0.812, the
structural modality is the weaker of the two by a clear margin — consistent with
its behaviour in the multimodal analysis (§6), where adding structural features
degraded every combination it entered.


---

## 22. The 833-specification sweep, and the same result priced four ways

Everything at once: 4 connectivity statistics x {21 edges, FPN/CON/SMN}, 3
structural atlases, behaviour, motion, all modality combinations, 7 model
families including a neural network. **833 specifications**, one family-wise
null over the whole grid.

```
observed max        0.880
null max, median    0.800
null max, 95th pct  0.920
p                   0.2376
cells clearing the critical value:  0 of 833
```

**Searching 833 specifications on 25 participants produces 80% accuracy half the
time from randomly permuted labels.** That is the median of the null, not its
tail. Twenty-eight cells of the real grid reach 0.80 and thirteen reach 0.84;
none of it survives, because none of it has to be real to happen.

Anyone asking this dataset for "above 80%" now has a precise answer: it is
available, it is available dozens of ways, and at this search size it is the
coin-flip expectation under no signal at all.

### 22.1 The same model, priced by how hard you looked

The top brain-only cell of the 833 is `functional:precision-theory` with ridge —
precision connectivity restricted to FPN, CON and SMN. That is the section 15
configuration, arrived at independently by a completely different search, and it
beat every cell that added behaviour or motion. The finding is stable. What is
not stable is what it is worth:

| family | best | p |
|---|---|---|
| 7 cells (structure/function fusion) | 0.812 | **0.0010** |
| 20 network triples | 0.812 | **0.0066** |
| 80 (statistic x triple) | 0.812 | **0.0399** |
| 833 (everything) | 0.880 | 0.2376 |

The same underlying model, four times, worth progressively less as the declared
family widens — and in the widest family a *higher* accuracy is *less*
significant than a lower one in a narrow family. This table is the multiverse
argument made arithmetic rather than argued, and it is the single most useful
artefact this document produced.

It also settles how the result should be reported. Not as "we searched 833 ways
and found 0.880", which is worth nothing, but as "we declared precision
connectivity over three networks and obtained 0.812" — but §15.1 blocks even
that, because those three networks came from §3.1, on the same participants. The
honest report is narrower still: *this sample's most separable restriction
involves control-to-sensorimotor coupling; here is the accuracy; no p-value in
this document prices it, because every family we could declare was drawn after
the answer was known.* That is a hypothesis for new data, with the motion caveat
of §15 attached.

## 23. The structural arm's own grid, rerun on the corrected volumes

Sections 18–19 fixed the two defects but reported only the pooled and
cross-dataset analyses. The arm's *own* specification grid — the thing the
withdrawn §3.6 was built on — was left holding results from the contaminated
pass until now.

Rerun over three volumetric atlases by three classifiers, same fixed folds, one
family-wise max-statistic null over all nine:

```
harvard-oxford/gmv/mlp      0.582  [0.44, 0.71]   <- best
harvard-oxford/gmv/svm      0.576
harvard-oxford/gmv/xgboost  0.559
schaefer200/gmv/svm         0.554
schaefer200/gmv/mlp         0.546
aal/gmv/svm                 0.523
aal/gmv/mlp                 0.505
aal/gmv/xgboost             0.503
schaefer200/gmv/xgboost     0.473

critical value (95)   0.664
p                     0.3014      B = 500
cells clearing        0 of 9
```

The correction moved the best cell from 0.516 to 0.582 and moved nothing else.
Two details are worth keeping. Harvard-Oxford leads on all three classifiers,
which is the same ordering the profile correlations give (+0.94 against AAL's
+0.87 and Schaefer-200's +0.58) — 48 large parcels are the most reliable
structural features available in these data, and they still decode nothing.
And the arm's critical value, 0.664 over nine specifications at n = 47, is well
below the 0.739 the resting arm needs over 49 at n = 32: a wider search on a
smaller sample is a much more expensive thing to correct for.

Attribution was **removed from the script** rather than recomputed. It is the
one part of the old §3.6 that cannot be repaired by better volumes, because the
objection is not to the volumes — it is that attribution ranks from a
chance-level classifier describe noise, whatever they are computed on.

## 24. The cerebellar control, and what section 15's restriction is actually worth

Section 15's network-restricted result (0.812, restricting to FPN+CON+SMN) has
one obvious alternative explanation that had not been tested: at n = 32 *any*
dimensionality reduction helps, so the gain may be about how many features
survived rather than which. The control is to restrict somewhere the theory does
not implicate.

The cerebellum is that somewhere, and it needs no new parcellation — Dosenbach
2010 already assigns 18 of its 160 regions to it. (The manuscript had this
recorded as blocked on DiFuMo-256. It was not blocked; nobody had looked.)

Everything is held at k = 18 regions, which is the entire design. An unmatched
contrast — 18 cerebellar regions against the 86 of FPN+CON+SMN — confounds
*where* with *how many* and answers neither. It is also intractable at the top
end: shrinkage LDA over the 12,720 edges of the full atlas is an O(p^3) solve
per fold, and the first version of this script hung there.

```
cerebellum, 18 regions, 153 edges              0.469   [0.31, 0.64]

200 draws of 18 from FPN+CON+SMN (86 regions)  median 0.469, 95th 0.656, max 0.750
  cerebellum sits at the 41st percentile
200 draws of 18 from non-cerebellar (142)      median 0.469, 95th 0.627, max 0.719
  cerebellum sits at the 35th percentile

theory minus elsewhere, median                 +0.000
```

**At region scale, restricting to the three discovered networks is worth
exactly nothing over restricting anywhere else.** The two medians agree to three
decimals. The cerebellum is mildly *worse* than both, which is the right
direction for a displaced control, but it is inside both distributions rather
than outside them.

So section 15's number survives, with its meaning changed twice over — once by
§15.1 (it prices no hypothesis, having been read off the data) and once here.
What buys the 0.812 is **aggregation to network scale**, not localisation: averaging regions into
three network nodes and taking the six edges among them is a different object
from the 153 region edges inside the same territory, and only the former helps.
Any sentence claiming those three networks are where the signal lives is not
supported by this; the supported claim is that the network-scale summary of them
is the low-dimensional feature set that a 32-participant sample can estimate.

One more number from the same run, for section 22's file: the best of 200
random 18-region subsets reaches **0.750**, chosen for no reason whatsoever. Two
hundred is a small search.

