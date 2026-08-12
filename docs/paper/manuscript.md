# What survives a multiverse? A measurement ceiling, and a stable explanation that survives only because the models disagree — resting-state connectivity in action video game players

## How to read this draft


This is the single working draft. An earlier pair of alternative framings — one leading with
the transfer question, one leading with analytic robustness — has been collapsed into this
file; nothing was lost, since they shared all of their Methods and limitations text.

Most of the prose is carried over from the original manuscript, because most of it still
works. Provenance is recorded once, in Appendix A of `supplementary.md`, rather than annotated
section by section, so the body reads as continuous prose. Editorial bookkeeping a reader does
not need — coauthor comment disposition, text cut from the original — lives in
`supplementary.md`.

Two conventions apply throughout. `[[TODO: ...]]` and `[[VERIFY: ...]]` mark a value the
analyses have yet to produce and a citation not yet confirmed; nothing is invented, and
everything outside a marker is submittable text. Counts of specifications are computed rather
than asserted, from `acnets.multiverse.SpecificationRegistry().summary()`: **495
specifications, of which 390 are substantive and 105 negative controls**, across 10
parcellations, 7 connectivity statistics, 3 aggregation scales and 3 classifier families. The
grid actually *run* is a 49-specification subset on one atlas; see the box below.

**What this paper is about.** Action video game play improves a broad range of cognitive
abilities, and the question is why the improvement is broad rather than narrow. Competing
accounts of that breadth make different predictions about where in the brain the change should
sit, and resting-state connectivity can in principle adjudicate between them. That was the
design.

What the analysis established is that this sample cannot adjudicate, and *why* — a reliability
ceiling that sits below the threshold a corrected search requires (section 3.7) — together with
two things about the machinery used to reach that conclusion: what a multiverse prices and what
it leaves unpriced (sections 2.2, 3.4.5), and which null an attribution-stability claim has to
clear (section 3.4.3). The multiverse began as the method and has ended as the larger of the
two contributions. A reader who wants only the substantive question should read 1.1–1.6, 3.1,
3.3 and 4.1–4.3; a reader who wants the methodological result should read 2.2, 3.4, 3.7 and
4.4.

> **Status of the results, as of this draft.** The family-wise null has been run at B = 1000
> over the final 49-specification grid and **the decoding result does not survive it**: best
> 0.660 against a critical value of 0.739, p = 0.291, 0 of 49 clearing. The structural arm is
> null too, within sample and in six external-validation transfers. Section 3.7 explains both:
> at the measured edge reliability the attainable accuracy caps near 0.72, *below* the 0.739
> the correction requires, so this design could not have produced a positive result for an
> effect of ordinary size.
>
> **The attribution-stability claim has been put to a permuted-label null, which it passes on
> the grid the paper uses and fails on a single-family subgrid. This is the most consequential
> change in the draft.** Agreement clears a random-ranking baseline (+0.366, p = 0.019), but
> that baseline is 0.15 below the real null. Refitting the grid on permuted labels gives 0.562
> against an observed 0.657 across three classifier families (**p = 0.035**, B = 200) and 0.601
> against 0.630 within the linear family alone (**p = 0.29**, B = 1000). Label-free feature
> geometry does not explain the agreement either. The mechanism is shared inductive bias:
> specifications from one model family overfit the same 32 participants the same way.
>
> **Anyone holding a copy from earlier today has the wrong version of this.** An intermediate
> draft reported only the linear subgrid and concluded the claim had failed. It had not; the
> subgrid is the least favourable slice of the grid, and the three-classifier run reverses the
> sign. The title, abstract, sections 3.1, 3.4.3, 3.4.4, 4.4 and 5 have been rewritten twice
> and now reflect the fuller result. Outstanding: extend the three-classifier null to B = 1000
> and all 31 specifications (§3.4.3 scope box).
>
> Four things changed late and matter to anyone holding an earlier copy. **Do not circulate**
> (i) any version of section 3.1 asserting a reliable group difference; (ii) the earlier
> section 3.6, whose attribution result came from skull-contaminated volumes over a sphere
> atlas and is withdrawn; (iii) any text presenting FPN/CON/SMN as an *a priori* restriction —
> those networks were identified in these data, section 1.5 now says so, and section 3.4.5
> works through what it costs; (iv) any version of the abstract, title or discussion offering
> attribution stability as a positive finding.

**Target venue.** Imaging Neuroscience, or NeuroImage. Both accept a transfer-mechanism framing
while giving credit for the analytic work, and neither imposes page pressure on the
specification-curve figure. The trade-off is a reviewer pool that will scrutinise the imaging
more than the cognitive claim, so section 6 has to be complete before submission. Journal of
Cognitive Enhancement and Technology, Mind, and Behavior remain the fallbacks if reviewers there
object to the sample size; that pool cares about the transfer claim and would need less of the
multiverse apparatus.

**Title.** Options 1 and 2 assert a group difference and are only available if the decoding
result reaches significance under the family-wise null. It does not. Option 3 was the choice
until the permuted-label attribution null was run; it asserts attribution *stability* as a
finding, and that assertion is no longer available either (section 3.4.3). Option 4 stands.

1. *Why is transfer broad? Control-to-sensorimotor coupling at rest in action video game
   players.* The original choice. Requires a positive decoding result.
2. *Action video game experience is associated with reconfigured coupling between control and
   sensorimotor networks.* Flatter, safer, but still asserts the association.
3. *What survives a multiverse? Attribution stability without decodability in resting-state
   connectivity of action video game players.* Available again, but incomplete: attribution
   stability does survive a permuted-label null (p = 0.035), and this title does not convey
   that it survives only on a classifier-diverse grid, which is the part worth publishing.
4. *What survives a multiverse? A measurement ceiling, and a stable explanation that survives
   only because the models disagree — resting-state connectivity in action video game players.*
   **Current choice.** What survives is the ceiling computation of section 3.7, the
   dimensionality-matched atlas result of section 3.4.2, and the finding that an
   attribution-stability claim is testable only when the grid varies the model family.
   Shorter variants worth considering: *Explanation stability is a property of multiverse
   diversity, not of multiverse size*, or *A reliability ceiling below the correction
   threshold, and an explanation that shuffled labels reproduce unless the models disagree*.

**Length budget.** The draft is over on every line of this table, deliberately: it is written
to be cut once a venue is chosen, and cutting is safer than reconstructing.

| Section | Target words | Current | Note |
|---|---|---|---|
| Abstract | 200-250 | ~1100 | **must be cut hardest**; the structured version here is the coauthor-review copy. When cutting, the three paragraphs to protect are the reliability ceiling, the permuted-label null, and the sentence saying this bears on no behavioural claim |
| 1. Introduction | 1400 | over | transfer breadth leads |
| 2. Competing accounts of breadth | 700 | near | was "the specification space" |
| 3. Results | 1800 | well over | 3.4 carries the robustness warrant; 3.7-3.8 are the bound and the search |
| 4. Discussion | 1400 | over | leads on transfer, not on method |
| 5. Conclusion | 250 | near | |
| 6. Methods | unlimited | — | multiverse machinery lives here |

---

## Abstract


Action video game play improves a broad range of attentional and cognitive abilities, and the central open question is why the improvement is broad rather than confined to the trained task. Competing accounts locate the underlying change in specific control regions, within a single control network, or in the coordination among networks, and these make different predictions about where in a connectivity matrix a group difference should appear.

We tested them in **two independent samples totalling 79 participants and two imaging
modalities**: resting-state fMRI from 16 action video game players (AVGPs) and 16 non-players (NVGPs), and T1-weighted structural images from 24 action-game experts and 23 non-experts — the anatomical scans acquired alongside Gong et al. (2015), the closest published precedent, shared by that paper's authors and analysed here with an independent pipeline. Because their published finding concerns *functional* connectivity and this arm analyses *structural* volume, it is a new question about their sample rather than a replication attempt of their result. Neither sample was analysed along one pipeline. The functional arm crosses a declared multiverse of 495 specifications — parcellation, connectivity statistic, aggregation scale, classifier family and external pretraining — of which 49 were run against a permutation null; the structural arm runs its own nine-cell grid over three volumetric atlases. Beyond that we ran thirteen classifier families with nested hyperparameter search, a deep model pretrained on 220 external resting-state subjects and fine-tuned within folds, multimodal and structure-function fusion, multiple-instance learning over windows, and diffusion-based normative modelling.

**What the design could have shown, computed first.** Split-half edge reliability in this
sample has a median ICC of **0.283**, matching the 0.29 that Noble et al. report across a decade
of test-retest studies. Attenuation by measurement error caps attainable accuracy near **0.72**,
while a family-wise correction over a search of this width requires **0.739**. A perfectly real
effect of ordinary size, measured as well as these instruments allow, would therefore have
failed to clear the threshold. Every null below should be read against that number, which is
computable before data are collected rather than after they disappoint, and which we regard as
the most portable result here.

**Decoding is null wherever it is corrected.** Functional accuracy varied from 0.385 to 0.660
and no specification exceeded the null (best 0.660, critical value 0.739, p = 0.291, B = 1000).
The structural arm is null within sample (best 0.582, critical value 0.664, p = 0.30, 0 of 9)
and pooled across both datasets (0.544, p = 0.27). Six external-validation tests — training on
one sample and predicting every participant of the other, across three atlases and both
directions — are all at chance (0.489 to 0.511 against null medians of 0.500 to 0.511), with
the two feature spaces verified to correspond first (profile r = +0.58 to +0.94). Additional
modelling does not move it: tuning has a mean effect of −0.003, fusing structure with function
*loses* 0.25 relative to function alone, instance-level learning gains exactly zero, and
diffusion normative scores are worse than the features they are computed from. Nor does using
everything at once: a stacked model that predicts all 79 participants from whichever modalities
each has — structural for 79, connectivity and behaviour for 32 — reaches 0.528 brain-only
(critical 0.595, p = 0.28), *below* connectivity alone on 32, because the participants pooling
adds carry only the modality that does not discriminate (structural path r = +0.05 against
+0.50 for connectivity).

**Three results are positive at the strength this design supports.** The atlas earns its
keep: across 18 exactly matched pairs, real region-to-network assignment beats a
dimensionality-matched shuffle (0.564 against 0.455, real ahead in 15 of 18). Attribution
converges on coupling between the cingulo-opercular (CON) and fronto-parietal (FPN) control
networks and the sensorimotor network (SMN) — CON-SMN in the leading positions of every
specification, FPN-SMN in 81% — and in-fold feature selection recovers CON-SMN in **94 of 100
folds**. And reliability-weighted precision connectivity reaches **0.750 against a critical
value of 0.750** (p = 0.054), the closest anything here comes to clearing a corrected null on a
choice that could have been specified in advance.

**The attribution stability survives a permuted-label null, but only on a grid that varies the
model family, and that dependence is the paper's main methodological result.** Agreement across
specifications is 0.366 above what random rankings of the same features achieve (p = 0.019),
and a dimensionality-matched control correctly fails (p = 0.135) — but a random-ranking
baseline is not the rival worth excluding. Refitting the whole grid on **permuted labels** —
same folds, same features, same attribution methods — gives **0.562 against an observed 0.657
across three classifier families (p = 0.035)** and **0.601 against 0.630 within the linear
family alone (p = 0.29)**. Specifications sharing an inductive bias overfit the same 32
participants the same way and agree whether or not the labels mean anything; specifications
that do not, do not. Since the observed agreement barely moves between the two grids, the
entire inferential content sits in the null — and the classifier axis, which contributes only
3-5% of accuracy variance and is therefore the natural one to economise on, is what makes the
claim testable. Label-free feature geometry does not explain the agreement either (rankings by
variance, mean magnitude, leading covariance eigenvector, or the pattern a chance-level decoder
yields agree with the consensus at 0.278 to 0.461, inside the 0.408 [0.282, 0.611]
random-ranking interval). The identity of the converged-on feature points the same way: permuted
grids put CON-SMN first in 62 of 1000 (p = 0.063), selecting no alternative consistently.
Kendall's tau over the full ordering is at chance independently of all this.

**Related limits on the substantive reading.** Discovering networks by attribution in one
sample and modality and testing them in the other fails in both directions (p = 0.71 and
p = 1.00); restricting to those networks beats no size-matched restriction elsewhere in the
brain (0.469 against 0.469), so the gain is dimensionality rather than localisation. CON-SMN
coupling relates to attentional task performance only weakly: on all 32 participants, three of
eighteen uncorrected bootstrap intervals exclude zero, of which one is the within-group
association the transfer account requires (CON-SMN with inverse efficiency within AVGPs,
r = −0.378 [−0.682, −0.070]). The same analysis on the 25 participants an identifier mismatch
had left available found nothing, so this is suggestive and unestablished rather than a result.

One confound qualifies the substantive reading and we report it prominently rather than in a footnote: action gamers moved more in the scanner, motion summaries alone decode group at 0.61, and motion correlates with sensorimotor coupling at r ≈ 0.5. The CON-SMN group association survives partialling motion out in four of six connectivity statistics; the FPN-SMN association does not.

**The study was not preregistered and is exploratory in whole; there is no confirmatory component.** One transfer model was specified in writing before it was fitted, but that document was written partway through the analysis of these same participants, so it licenses nothing and we claim nothing from it (it is null in any case: 0.477, p = 0.0999). In particular the three networks above are a result of this analysis rather than an input to it. A restriction to them reaches 0.812, the highest accuracy anywhere in this work, but that number selects and tests on the same 32 participants: it is a hypothesis this study generates, not one it evaluates, and we report it as such rather than attaching a family-wise p-value we are not entitled to.

**The multiverse itself has unpriced degrees of freedom, and we quantify rather than assert
them.** A family-wise correction disciplines a search only once the family is declared, and
declaring it is a choice no fitting procedure sees: holding one result fixed at 0.812 and
varying only how widely the family is drawn moves its p-value from 0.0010 to 0.2376. We
declared 495 specifications and ran 49. And the largest single analytic effect we measured sits
on no declared axis at all — whether region time courses are averaged before or after
connectivity is estimated is worth 0.13 to 0.19 in accuracy, more than statistic, classifier
and pretraining combined.

**Nothing here bears on whether action video game play improves cognition.** That question is
behavioural and has been addressed by meta-analysis; our own participants show the expected
behavioural difference in the same session as these scans (accuracy 0.840 against 0.750,
inverse efficiency 1.111 against 1.409; d = +0.74 and −0.76). Neither dataset was collected to answer the question we put to
it, both were shared with us, and the structural arm tests a modality — regional grey-matter
volume — that the study which collected it did not analyse. These are nulls about what a
32-participant, 6-minute resting-state design can decode, not about the phenomenon.

Three conclusions follow. Substantively, control-to-sensorimotor coupling remains a *candidate*
substrate for breadth of transfer, since it alters a resource common to many tasks rather than
the efficiency of one specialised system, and a restriction to those networks reaching 0.812 on
these participants is a hypothesis worth a preregistered test in data that did not generate it.
Methodologically, the worth of an accuracy depends on a declared family that no
cross-validation can see, and the worth of a stable explanation depends on a permuted-label
null that a random-ranking baseline cannot substitute for — both of which are cheap to compute
and neither of which we computed until late. And measurement-wise, reliability sets a ceiling
that a study of this size cannot clear, which is knowable in advance and, we would argue,
belongs in the design stage of any study of this kind rather than in its discussion.

---

## 1. Introduction


### 1.1 Action video games improve cognition, and the improvement is broad

Playing action video games, that is first- or third-person shooter games, has been associated
with a broad range of cognitive abilities, including the ability to switch between
different tasks, filter out irrelevant information, and perceive finer details, while
leaving other abilities unaffected (e.g., bottom-up attention; Bediou et al. 2018; Bediou
et al. 2023). The breadth of that pattern is the striking part and the part that demands
explanation. In the most recent meta-analysis, habitual action gamers outperformed non-gamers
cross-sectionally with a large effect (g = 0.64, 95% CI [0.53, 0.74], 105 studies, 221 effect
sizes), and intervention studies in which participants are assigned to an action game or to a
control game with different mechanics recover a smaller but reliable one (g = 0.30, 95% CI
[0.11, 0.50], 28 studies; Bediou et al. 2023). Among intervention domains only top-down
attention had a confidence interval excluding zero (g = 0.52). Improvement on such a range of
tasks after training on one is not what most cognitive training produces, where gains tend to
remain close to the trained task.

That literature is contested, and the disagreement should be stated rather than managed. Two
meta-analyses in the same journal reached opposing headline conclusions from overlapping
corpora: Bediou et al. (2018) reported a cross-sectional advantage of about half a standard
deviation (g = 0.55) and an intervention effect of about a third (g = 0.34), while Sala,
Tatlidil and Gobet (2018) found small or null effects across three meta-analytic models and no
evidence of a causal game-to-cognition relationship once active controls and study quality were
accounted for, with effect sizes shrinking as design quality rose. The divergence is driven
largely by inclusion criteria, the treatment of active controls, and publication-bias
correction; the broader skeptic position holds that far transfer is rare across trained-skill
domains generally (Sala and Gobet 2017). Bediou et al. flag the bias problem themselves,
estimating published effects to be about 30% larger than those in the full literature. In the
2023 update publication bias was detected in the cross-sectional but not the intervention data
set, which is some reassurance that the smaller causal effect is not purely an artefact of
what gets published.

We take the most defensible reading to be that the cross-sectional advantage is robust, the
causal effect is real but small and most reliable for top-down attention, and that the effect
is broad in the sense of spanning perception, attention and spatial cognition rather than one
trained task. Nothing in this paper depends on resolving that dispute. The neural question —
what about the underlying system is organised differently — is worth asking under either
reading, and is a convergent line of evidence rather than a referendum on the behavioural
meta-analyses.

While the cognitive mechanisms underlying these benefits are not yet fully understood, the broad
impact of action video game play on cognition has been attributed to domain-general abilities.
These include in particular control functions, at times termed cognitive control (Anguera et al.
2013; West et al. 2020), executive functions (Bavelier and Green 2019), or attentional control
(Bavelier et al. 2012; Bavelier and Green 2019), as well as potentially other domain-general
abilities related to reward processing (Nahum and Bavelier 2020; Benady-Chorney et al. 2020;
Gleich et al. 2017; Watanabe et al. 2015). Appealing to a domain-general ability explains breadth
in principle, but it relocates the question rather than settling it: it remains to be shown what
about the underlying system is altered such that many different tasks benefit.

It is critical to note that video games are not all made equal: there is a large variety
of video game genres with distinct combinations of features and mechanics which have very
different effects on cognition, behaviour and the brain (Bediou et al. 2018; Bediou et al.
2023; Cardoso-Leite et al. 2021; Kühn et al. 2014). Here we focus specifically on the
category of video games known as action video games in the scientific literature (Green,
Li and Bavelier 2010). These are first- or third-person shooter games that (1) are
fast-paced, involving rapid motion and transient events such as projectiles and
explosions, (2) require switching between distributed and focused attention, (3) demand a
high attentional load by stressing motor, perceptual, and cognitive systems concurrently,
and (4) contain both temporal and spatial uncertainty. There is substantial evidence that
playing this specific type of video game, as opposed to other types such as puzzle games,
causes a variety of cognitive improvements (Bediou et al. 2023). The mechanisms underlying
these specific improvements remain largely unexplored.

A caveat on how that category is defined. The one-to-one mapping between genre label and
gameplay mechanics has broken down as hybrid titles proliferate, and the field is moving from
labels toward a mechanics-based taxonomy (Dale et al. 2020); a game-factors analysis finds
labels such as "action" or "strategy" to be poor predictors relative to the underlying gameplay
features. We therefore treat "action video game" throughout as shorthand for a constellation of
demands — pacing under time pressure, a sustained load on divided attention across a wide
field, and repeated switching between divided and focused attentional modes (Bavelier and Green
2019) — and characterise our sample by those demands rather than by the marketing category its
games belong to. This matters for the present argument specifically, because the theoretical
pivot in section 2 turns on what action games *require* of the control-to-sensorimotor loop,
not on what shelf they sit on.

### 1.2 Neural correlates of action video game play


At the neural level, playing action video games has been associated with changes in brain
functions related to control (Bavelier et al. 2012; Föcker et al. 2018; Föcker et al.
2019; Gong et al. 2017; Mishra et al. 2011; Richlan et al. 2018; Cui et al. 2021) as well
as with changes in brain structures related to reward processing and automatisation, for
example increased grey matter in the caudate nucleus and decreased grey matter in the
hippocampus (West et al. 2018; Kühn and Gallinat 2014). One study, for example, used
functional magnetic resonance imaging (fMRI) to record the brain activity of participants
while they performed an attention-demanding visual search task in the presence of
distractors (Bavelier et al. 2012). When contrasting action video game players (AVGPs)
with people who do not play video games (non-video game players; NVGPs), the
fronto-parietal brain network, a key actor in cognitive and attentional control, was less
activated by increased attentional demands in AVGPs than in NVGPs. Lesser brain activity
in such control areas in the context of better performance has been interpreted as
resulting from easier, better top-down attentional control abilities in AVGPs compared to
NVGPs.

A smaller resting-state literature exists and is directly relevant here. Action video game
experience has been associated with increased resting-state functional connectivity between
the nucleus accumbens and subgenual anterior cingulate cortex, and between the caudate
nucleus and precentral gyrus (Benady-Chorney et al. 2020), and with altered
resting-state connectivity following video game play more generally (Martínez et al. 2013;
West et al. 2015). Most relevant to what follows, Gong et al. (2015) compared action game
experts with amateurs and reported increased functional connectivity between attentional
and sensorimotor networks, with the effect concentrated in the left insula. Longitudinal
work has shown connectivity changes persisting after intensive first-person shooter
training ([[VERIFY: Brilhault et al. 2020]]). Outside the resting state, Cahill, Jordan and
Dhamala (2024) report enhanced functional and structural connectivity in the dorsal visual
stream in AVGPs, between left superior occipital gyrus and left superior parietal lobule, with
that connectivity tracking response time. Their study uses task fMRI and diffusion imaging
rather than rest, so it corroborates the general claim that gaming experience is visible in
connectivity without being a resting-state precedent.

### 1.3 The brain is organised into interacting large-scale networks


It has become increasingly clear in cognitive neuroscience that the traditional, modular
approach, which assumes a specific cognitive function X is performed by a brain area Y, is
limited (Poldrack 2006; but see Hutzler 2014). Instead, considering networks and how they
may interact with each other to generate behaviour emerges as a more promising approach
(Sporns 2013a; Sporns 2013b; Thomson and Poldrack 2022). This holds particularly true for
cognitive control, where the scientific evidence has not identified a single cognitive
control area but instead highlighted multiple large-scale control networks (Badre 2021;
Reineberg et al. 2015).

Similarly, within the field of attention, a large body of work recording fMRI while humans
perform a variety of visuo-spatial attentional tasks has revealed two broad attentional
systems: a dorsal frontoparietal system involved in top-down attentional control, for
example maintaining attentional focus on a stimulus, and a more ventral system responsible
for bottom-up attention, for example detecting danger (Corbetta and Shulman 2002).
Although these segregated networks are specialised and often regarded as being functionally
separate, their coordination is necessary for adaptive and stable behaviours. For instance,
a subset of the ventral attention system, which includes the temporoparietal junction
(TPJ), is often regarded as a circuit breaker, interrupting activity in the top-down dorsal
system when an important stimulus demands an adjustment of top-down attention (Anticevic
et al. 2010; Corbetta et al. 2008; Dugué et al. 2018; Lynch et al. 2014).

The leading hypotheses regarding the broad effect of action video games on cognition assume
enhanced cognitive control, a domain-general ability believed to be subserved by
large-scale brain networks and their crosstalk rather than by a localised, single brain
area (Badre 2021; Menon and D'Esposito 2022). Since cognitive control involves broad
recruitment of brain networks and their interaction, the study of cognitive control in
action video gaming demands a system-wide approach to the brain.

### 1.4 Why resting state


Measuring at rest rather than during a task can look like a concession — giving up the ability
to see the system doing the thing of interest. Three separate lines of evidence say it is not,
and they are worth stating separately because they license different parts of the argument.

**Rest and task are largely the same architecture.** The whole-brain network organisation
present across dozens of task states closely matches the resting-state organisation, with task
states amounting to the intrinsic architecture plus small and consistent modifications (Cole et
al. 2014). Individual task-evoked activation can be predicted from task-free scans across many
cognitive domains, implying that individual task responses are largely inherent rather than
constructed anew by the task (Tavor et al. 2016). Within visual and dorsal attention networks
specifically, a stable connectivity backbone is preserved from rest into task while attention
adds top-down directed influences on top of it (Spadone et al. 2015). Measuring intrinsic
connectivity therefore does not sacrifice task relevance; it isolates the backbone on which
task states are built.

**That architecture is shaped by experience.** Intensive perceptual learning alters resting
connectivity between the trained sensory cortex and fronto-parietal attention areas, and the
size of the resting change scales with how much was learned (Lewis et al. 2009) — direct
evidence that training leaves a trace visible at rest. The mechanism is a Hebbian one:
spontaneous correlations encode individual biases stored as synaptic efficacies, which
sustained co-activation can durably remodel (Harmelech and Malach 2013; Guerra-Carrillo et al.
2014). The same signature has been found in other high-practice domains, with aspiring
professional musicians showing greater resting network strength than amateurs in a
task-defined auditory network, correlated with performance (Papadaki et al. 2023). Years of
action gaming repeatedly and jointly recruit control and sensorimotor systems; by this logic it
should sculpt their resting coupling.

**And it is stable enough to be a trait.** Individual connectivity profiles are reliable enough
to identify a person across sessions and even between rest and task (Finn et al. 2015), and
functional networks are dominated by stable individual factors rather than by task state or
day-to-day variation (Gratton et al. 2018). Those stable profiles carry behavioural
information: connectome-based models predict sustained attention in novel individuals
(Rosenberg et al. 2016), and resting network connectivity predicts individual differences in
executive function (Reineberg et al. 2015; see also [[VERIFY: Menon and D'Esposito 2022]] for
cognitive control, and Seguin et al. 2020 for behavioural dimensions more broadly).

Taken together: if resting connectivity indexes stable individual differences in exactly the
attentional and control faculties that action gaming is argued to enhance, then a group
difference in control-to-sensorimotor coupling would be a candidate trait-level substrate
rather than an incidental state effect. That is the measurement logic this study rests on. What
it does *not* license is any inference about direction, an issue we return to in section 4.6.

### 1.5 On-line control motivates looking at control-to-sensorimotor coupling


Action video games are closed-loop control tasks. Players continuously monitor a
fast-changing environment, update task sets under temporal and spatial uncertainty, and
translate decisions into motor output under time pressure. On-line control of this kind
requires sustained exchange between control systems and the structures that link cognition
to action, rather than change within either system alone.

This matters for what we should expect to see, because the sensorimotor network as defined
by Dosenbach et al. (2010) is not primary sensory and motor cortex. It comprises the
precentral gyrus, mid-insular cortex, supplementary motor area (SMA), pre-SMA, and superior
parietal cortex, that is, the supplementary motor complex and its parietal partners. These
structures are implicated in learning and executing action sequences, switching between
tasks, and the initiation and inhibition of actions; the supplementary motor complex is
crucial to link cognition to action (Nachev et al. 2008; Gordon et al. 2023). AVGPs show
advantages across precisely these domains, including sensorimotor learning, responsiveness
of the sensorimotor system to input error signals, spatial orienting, and stimulus-response
processes (Gozli et al. 2014). Consistent with this, Gong et al. (2015) reported increased
connectivity between attentional and sensorimotor networks in action game experts.

This reasoning says where it is worth looking. It does not, on its own, name a hypothesis
narrow enough to test, and we want to be exact about the difference, because the distinction
governs how every number in section 3 may be read.

What the literature supplies is a *region of interest in the space of accounts*: that
transfer from action gaming, if it has an intrinsic-connectivity substrate at all, is more
likely to involve coupling between control systems and the structures that link cognition to
action than to sit within either alone. What it does not supply is the specific contrast.
Gong et al. (2015) name the insula and an "attentional" network that is not the Dosenbach CON
or FPN; the dual-network account (Dosenbach et al. 2007, 2008) is about how control is
maintained and adjusted, not about which of six network pairs in a particular parcellation
should separate two groups. Nobody could have written down "CON-SMN and FPN-SMN, in the
Dosenbach 2010 six-network vocabulary" from prior work alone.

**So we did not, and we say so plainly: this study was not preregistered, and no network
restriction was declared before the data were analysed.** The three networks that recur
throughout section 3 are a *result* of this study, arrived at in section 3.1 by asking which
features the classifiers weight. Section 2 sets the accounts of section 2.1 alongside each
other as the frame the analysis works in; it does not convert them into a declared contrast,
and section 3.4.5 prices what that costs.

### 1.6 The present study


This study asks where the change associated with action video game experience sits, and
therefore which account of transfer breadth it supports. We recorded resting-state fMRI in
action video game players and non-players, and asked whether the group difference is carried by
regional features, by connectivity within a single control network, or by connectivity between
networks. The three accounts in section 2.1 make competing predictions about exactly this, so
the comparison is a test rather than a description.

The question is asked twice, in two samples that share no participants, scanner or modality.
The second is the T1-weighted data behind Gong et al. (2015) — action-game experts versus
non-experts, n = 47 — shared by that paper's authors and reanalysed here through a pipeline of
our own. It is not extra sample size; it is the only external check available, and it does
three things a single sample cannot: it tests whether a structural correlate exists at all
(section 3.6), whether a model fitted in one sample predicts the other, and whether the
networks the functional arm identifies are recoverable when nothing but their *identity*
crosses the gap.

Four commitments follow from taking that test seriously. First, the accounts must compete
within one model rather than across separate analyses, since comparing a 12,880-feature regional
model against a 21-feature network model confounds the hypothesis with dimensionality; section
3.2 places all three feature types in a single classifier. Second, if the reconfigured coupling
is what supports improved performance, it should bear some relation to performance, so section
3.3 relates it to attentional task data from the same participants. Third, the conclusion must
not depend on how the connectivity was computed. Resting-state findings are known to move with
analytic choices, so rather than defend one path we enumerate 495 of them and report the whole
distribution (section 3.4). That analysis is a warrant for the claim, not the claim itself.
Fourth, a negative result is only informative if the search behind it was serious, so sections
3.7 and 3.8 report what bounds accuracy — a reliability ceiling that can be computed rather
than guessed — and everything that was tried against that bound and did not move it: thirteen
classifier families with nested tuning, pretraining on external data, modality fusion,
instance-level learning, and generative normative modelling.

---

## 2. Competing accounts of breadth


### 2.1 Three accounts of breadth, as predictions about a connectivity matrix


Accounts of why action video game play transfers broadly differ in what they take the
underlying change to be, and therefore in how wide a range of tasks should benefit. A change
confined to one region or one specialised network predicts improvement on the tasks that draw on
it and little else. A change in how networks coordinate predicts improvement wherever that
coordination is called upon, which is to say across many tasks. Breadth of transfer is thus not
a separate question from locus of change; it is the behavioural consequence of it.

Framing cognitive control in terms of intrinsic brain networks lets us state these accounts in
explicit terms, without appealing to specific tasks or situations, because each makes a
different prediction about *where* in a connectivity matrix a group difference should sit. That
is what makes them testable with resting-state data, and it is why the present design can
discriminate among them.

Under a **regional account**, changes in brain function due to action video game play are
limited to specific control-related regions, such as dorsolateral prefrontal cortex or the
TPJ, without any discernible effect at the system level and possibly without visible
change in resting-state functional connectivity at the network scale. Prediction: regional
features carry the signal, network-level features do not.

Under a **single-network account**, changes occur within individual networks, such as the
top-down dorsal attention network but not the bottom-up ventral attention system, as might
be posited if action video games alter top-down attention but not the initial pull of
bottom-up attention (Hubert-Wallander et al. 2011). This would be in line with the notion
that an individual domain-general subsystem, such as top-down attention, is enhanced by
action video game play. Prediction: within-network connectivity of one control network is
diagnostic; crosstalk terms are not.

Under a **multi-network account**, consistent with the view that action video game play
facilitates learning to learn (Bavelier and Green 2012; Bavelier and Green 2019), the
effects extend beyond individual networks and broadly reconfigure multiple brain networks.
This would translate into differences not only within networks but also in their crosstalk,
as documented in changes of connectivity between components of the FPN and the VAN in
action video game players (Föcker et al. 2018). Prediction: between-network terms dominate
within-network terms, with the directional specification given in section 1.5.

The multi-network account has independent support. Task-induced brain networks become more
integrated or segregated according to task demands, and their integration correlates with
behavioural performance (Cole et al. 2013), which is what makes tighter or weaker crosstalk
between CON, FPN and SMN a plausible substrate rather than merely a possible one.

Discriminating among these accounts therefore bears directly on the breadth of transfer. The
broader the effect on the coordination of brain networks, the broader the range of tasks over
which those neural changes should manifest as improved performance, because coordination is a
resource many tasks draw on. This is the sense in which a neural result speaks to a behavioural
question that no single behavioural experiment settles: meta-analysis establishes that transfer
is broad (Bediou et al. 2018; Bediou et al. 2023), and the locus of change is what explains why.

### 2.2 Analytic degrees of freedom


Any analysis of resting-state connectivity requires a chain of choices for which the field
offers no settled ground truth. Parcellation maps define the spatial units or nodes of the
brain and transform voxels into regions and subsequently into networks, while connectivity
statistics quantify the relationships or edges between these nodes. Both choices can have a
substantial impact on the results and interpretation of network analyses.

The evidence that they do is now strong. Parcellation selection materially changes the
magnitude of associations between functional connectivity and age, poverty, and cognitive
ability, to the point of changing interpretation (Bryce et al. 2021). The choice of edge
statistic is equally consequential: benchmarking 239 pairwise interaction measures on
Human Connectome Project data, Liu et al. (2025) found substantial quantitative and
qualitative variation across methods, with covariance, precision, and distance measures
displaying the most desirable properties. Preprocessing variability alone is sufficient to
change published conclusions from the same data (Botvinik-Nezer et al. 2020). Yet the modal
study reports one parcellation, one statistic, one aggregation scale, and one classifier,
selected by convention rather than argument.

This leaves a claim like ours in an awkward position. Reporting the analysis path that worked
cannot distinguish a real effect from the best of many draws. Reporting only that results vary
says nothing about the brain.

**A multiverse does not remove that problem; it relocates it, and we should say where to.**
Enumerating the grid converts an undisclosed choice among pipelines into a disclosed choice of
*which* pipelines to enumerate — and that second choice is made by the same people, on the same
data, with the same latitude. Three specific residues remain in this study and each is
quantified later rather than waved at. The grid we declared has 495 cells and we ran 49 of
them, so the family-wise correction prices the search we report and not the search we could
have run (section 3.4.1). The largest single analytic effect we measured is not on any declared
axis at all: whether region time courses are averaged before or after connectivity is
estimated moves accuracy by 0.13 to 0.19, more than statistic, classifier and pretraining
combined, and we found it only because a reviewer-style question prompted the comparison
(section 3.2). And the width of the declared family is itself worth a factor of forty in
p-value on a fixed result (section 3.4.5). We report these because a multiverse presented as
though it closed the garden of forking paths is more misleading than a single-path analysis
that never claimed to: the reader who trusts the correction is entitled to know what the
correction does not cover.

### 2.3 The registry


The specification registry (`acnets.multiverse.SpecificationRegistry`) enumerates every
analysis path considered, and is published as machine-readable JSON in the supplement.

| Dimension | Levels | n |
|---|---|---|
| Parcellation | Dosenbach2010, Gordon2014, Seitzman2018, Friedman2020, Yeo2011, DiFuMo (64/128/256), AAL, Harvard-Oxford maxprob | 10 |
| Connectivity statistic | correlation, partial correlation, covariance, precision, tangent, Chatterjee, transfer entropy | 7 |
| Aggregation scale | region, network, random-network (control) | 3 |
| Classifier | L1 linear SVM, XGBoost, multilayer perceptron | 3 |
| Pretraining | none, MPI-LEMON | 2 |

The grid is not a plain Cartesian product, and two validity rules prune it. Network-level and
random-network aggregation require an atlas that defines a region-to-network mapping, so the
five region-only atlases (three DiFuMo resolutions, AAL, Harvard-Oxford) contribute
region-level specifications only. And LEMON pretraining is dropped wherever it would be a
no-op: it alters the connectivity measure only for the tangent statistic, whose `fit` carries
a reference covariance, and alters the feature scaler only where a scale-sensitive classifier
can see it. A tree ensemble splits on order statistics, so `precision + XGBoost + LEMON` is
bit-identical to `precision + XGBoost`, and emitting both would put exact duplicates in the
grid — overstating the number of analysis paths, handing free perfect scores to the
attribution-agreement mean, and diluting the pretraining dimension with cells that are zero by
construction. The rule is enforced in `Specification.is_valid` and asserted by a test that no
two specifications denote the same computation. Together these yield **495 valid
specifications: 210 region-level, 180 network-level, and 105 random-network controls**, of
which 390 are substantive.

The analyses reported below run a declared subset of this universe rather than all of it, for
reasons of compute and of data availability; each result states its own grid size, and section
3.4.1 gives the reasoning for the subset used in the family-wise null.

The seven statistics were chosen to span distinct assumption classes rather than to
maximise count: linear symmetric (correlation, covariance), conditional (partial
correlation), conditional inverse and sparse (precision), Riemannian (tangent), rank-based
and sensitive to non-monotonic dependence (Chatterjee), and directed and information
theoretic (transfer entropy). The last two matter most for the argument: if the stable
signature holds under a directed, nonlinear statistic, that is a considerably stronger
claim than holding across five variants of covariance.

We deliberately did not widen this axis to the full library of several hundred pairwise
interaction statistics. Under the family-wise max-statistic correction used here (section
6.6), adding many mutually correlated variants raises the threshold the observed effect
must clear without contributing independent information, so breadth for its own sake makes
the design strictly less sensitive. A multiverse is informative in proportion to how
different its specifications are, not how many there are.

### 2.4 One model was specified before it was fitted, and that is not a preregistration


This section used to claim a pre-specified confirmatory primary. It is withdrawn, and what
replaces it is worth stating carefully, because the distinction it turns on is one that
readers of this literature are routinely asked to take on trust.

A document exists (`docs/paper/prereg_primary.md`) that specifies one model completely — a
per-timepoint autoencoder pretrained on 1,980 windows of MPI-LEMON resting data (220 external
subjects, no labels), whose 4-dimensional latent induces a *learned* connectivity of 14
features, classified by a single linear layer fine-tuned within each cross-validation fold
from the frozen checkpoint. It was written before that model's accuracy on this dataset was
computed, it names the architecture, hyperparameters and CV budget, and it discloses a
rejected candidate (a sequence autoencoder whose held-out reconstruction never beat predicting
the mean). Architecture selection used LEMON reconstruction only. All of that is true and the
document is included.

**None of it makes the result confirmatory.** The document was written partway through this
project, after substantial analysis of these same participants, as part of an effort to find a
configuration that would decode well. A preregistration constrains what an analysis may do
before the data are seen; this constrained nothing, was not used to make any decision, and
could not have failed in a way that changed the paper. Writing a specification down before
running that one model is ordinary good practice — it prevents the model itself from drifting
mid-fit — and it is not the thing that licenses an uncorrected test. **The honest description
is that this study is exploratory in whole, with no confirmatory component.**

We therefore give the model no exemption. It is entered into the family-wise null alongside
everything else, its accuracy is reported as one more cell, and we do not describe it, or any
other analysis here, as preregistered. Its result appears in section 3.4.1: **0.477 against a
null 95th percentile of 0.514, p = 0.0999** — a null, which is also why nothing rests on how
it is labelled. Had it come out high, the label would have mattered a great deal, and we would
have had to say the same thing about it that we say here.

An earlier draft designated a grid specification as primary (Dosenbach 2010, partial
correlation, network scale, linear SVM, on the grounds recorded in
`SpecificationRegistry.primary()`). That designation predated the family-wise analysis and was
withdrawn for the same reason: once the whole grid has been seen, no choice within it can be
called pre-specified.

### 2.5 Negative controls


Three controls are stated in advance rather than introduced when results demand them.

1. **Random-network aggregation.** Regions are assigned to size-matched random networks,
   preserving the number of networks and the number of regions per network. This is the
   only dimensionality-matched control for the claim that the atlas is doing meaningful
   work, and 105 of the 495 specifications are of this kind.
2. **Motion.** A classifier receiving only the fMRIPrep motion confounds. Participant
   motion is a significant confounding factor in resting-state connectivity analysis,
   particularly when there are systematic group differences (Power et al. 2014), and
   sensorimotor differences between AVGPs and NVGPs have been reported (Gozli et al. 2014).
3. **Cerebellum.** A classifier receiving only connectivity among cerebellar regions,
   which are thought to modulate and coordinate movement (Koziol et al. 2014; Tseng et al.
   2007). This is a *spatially-displaced* control rather than a label-destroying one, so it
   is read against size-matched draws from elsewhere rather than against chance: the
   question is whether the networks section 3.1 identifies beat an arbitrary territory of
   the same dimensionality, not whether the cerebellum beats a coin (section 3.5). Since
   nothing was declared in advance anywhere in this study, this control is doing work that a
   preregistration would otherwise have done.

---

## 3. Results


### 3.1 Control-to-sensorimotor coupling is what the classifiers consistently rely on


**Read this section together with 3.4.1, 3.4.3 and 3.5.** What follows describes *which
features the classifiers weight*, and that description is reproducible across the multiverse.
It is not a claim that the groups are reliably separable: no specification cleared the
family-wise permutation null (section 3.4.1). Earlier drafts of this section asserted a group
difference; that assertion has been withdrawn.

**A second qualification was added late and is bounded rather than fatal.** Reproducibility
across analysis paths is not by itself evidence that the labels are doing the work: grids
refitted on permuted labels reach 0.601 of the observed 0.630 within one classifier family,
though 0.562 of 0.657 across three (p = 0.035, section 3.4.3). The selection frequencies below
are therefore a description of what these models consistently do, and are *not* a significance
test of the features named. What
distinguishes CON-SMN from the rest of the table on evidence that does use the labels is
narrower and reported elsewhere: permuted-label grids put it first in only 62 of 1000
permutations (p = 0.063) while selecting no alternative consistently (section 3.4.3), it is
recovered by in-fold selection in 94 of 100 folds (section 3.7), it retains a nominal group
association after motion is partialled out in four of six statistics where FPN-SMN retains none
(section 3.5), and it is stable across run halves where FPN-SMN attenuates (section 4.6). None
of those four is significant on its own at a corrected threshold, and the case for CON-SMN over
FPN-SMN rests on their agreeing rather than on any one of them.

Section 3.5 adds a second qualification that a reader should carry into this section rather
than meet later: action gamers in this sample moved more in the scanner, and head motion is
strongly correlated with precisely the sensorimotor coupling described below. The CON-SMN
association with group survives controlling for motion in most statistics; the FPN-SMN one
does not.

Features appearing in the top five across the 31 network-scale specifications, by selection
frequency (`outputs/agreement/selection_frequency_network.csv`):

| Feature | Selection frequency in top 5 |
|---|---|
| CON ↔ SMN | 1.00 |
| FPN ↔ SMN | 0.81 |
| CER ↔ VIS | 0.58 |
| CER ↔ CON | 0.39 |
| CER ↔ FPN | 0.32 |
| DMN ↔ VIS | 0.26 |
| FPN ↔ VIS | 0.26 |
| DMN ↔ SMN | 0.23 |
| SMN ↔ SMN | 0.19 |

Remaining features fall below 0.17; the full table of all 21 is in the supplement. We report
frequency rather than a single ranking because 32 participants cannot support an assertion
about the order of correlated features, whereas the reproducibility of a feature's prominence
is exactly what the multiverse can measure.

The connectivity between the cingulo-opercular and sensorimotor networks (CON-SMN) was among
the top five features in **all 31** specifications, and the connectivity between the
fronto-parietal and sensorimotor networks (FPN-SMN) in 25 of 31. Note that
these are frequencies over *all* specifications, not over above-threshold ones: conditioning
on accuracy would both shrink the set to nothing and, where it does not, inflate agreement by
selecting specifications that succeeded for the same reason (section 3.4.3 reports the whole
threshold curve for this reason). The three top-performing models placed less importance on within-network
features in the individual networks associated with cognitive control, particularly the FPN
and CON, when compared with between-network features. Overall, to discriminate the
resting state of AVGPs from NVGPs the relationships *between* networks play a much larger
role than individual networks do: the importance of FPN alone is negligible compared with
the importance of FPN-SMN.

It might be argued that this analysis is not well suited to detecting within-network
differences, because aggregating region time courses discards within-network variation. That
is partly true. Diagonal values carry within-network variability for the tangent and precision
statistics, but not for correlation or partial correlation, whose diagonals are fixed at 1 by
construction and therefore contribute nothing. The feature-block analysis in section 3.2
addresses the concern directly, with a within-network block built from region pairs so that it
is defined for every statistic — and finds that block at chance, along with every other.

Attribution shows that a feature carries the discriminative signal such as it is; it does
not indicate whether a connectivity is stronger or weaker in AVGPs. To assess magnitude and
direction we compared partial correlation connectivity between groups for CON-SMN, FPN-SMN,
and CON-FPN. AVGPs showed increased CON-SMN connectivity (d = +1.17, 95% bootstrap CI
[+0.54, +2.03]) and directionally increased FPN-SMN connectivity whose interval includes zero
(d = +0.61 [-0.08, +1.61]), while CON-FPN connectivity did not differ (d = -0.44
[-1.26, +0.25]). The ordering matches the prediction of section 1.5 — the control-to-SMN
couplings differ where the control-to-control coupling does not — with the caveat, stated
above, that these intervals are descriptive.

**These comparisons are descriptive and are not independent tests.** The features were
selected because classifiers fitted to these same 32 participants ranked them highly, so
testing them on the same data is circular (Kriegeskorte et al. 2009). We report effect
sizes with bootstrap intervals and no p-values. The original draft reported t-tests here;
they have been removed. Note also that with 16 per group, 80% power requires d of about
1.03, so any effect detectable in this design is very large, which is itself informative.

### 3.2 No feature block separates the accounts, but the order of aggregation matters


The three accounts in section 2.1 make claims about different parts of the same connectivity
structure, and the analyses above cannot adjudicate between them because regional and
network-level specifications differ in dimensionality. We therefore built three feature blocks
from a single region-level partial-correlation matrix, so that no block gets an estimation
advantage, and compared them on identical folds (`scripts/run_multiscale.py`): **H1**, the
weighted degree of each region — its mean absolute connectivity to every other region —
restricted to the 86 regions of FPN, CON and SMN; **H2**, the mean connectivity among region
pairs *inside* each network, 6 features; **H3**, the mean connectivity among region pairs
*spanning* each network pair, 15 features.

**Every block is at chance, alone and together.** Across the linear SVM, XGBoost and the
perceptron: H1 gives 0.504, 0.496 and 0.571; H2 gives 0.455, 0.525 and 0.466; H3 gives 0.450,
0.541 and 0.449; all three together give 0.487, 0.512 and 0.517. Every Wilson interval at
n = 32 spans chance by a wide margin. Permutation drops in the combined model are all within
±0.05, which at this fold count is noise. The block comparison therefore does not adjudicate
between the three accounts. It cannot: at n = 32 none of the blocks is separable enough for the
comparison to have resolution, and reporting a winner among them would be reading rank order
out of noise.

**What the analysis does show is an effect of a choice nobody reports.** The same 15
between-network quantities can be computed two ways. Estimating connectivity between regions
and then averaging over network pairs — the H3 block above — gives 0.450, 0.541 and 0.449.
Averaging the region *time courses* into network time courses first and then estimating
connectivity between them gives **0.608, 0.667 and 0.636** for the same three classifiers, on
the same participants, folds and statistic. The features carry the same name and describe the
same network pairs; only the order of averaging and estimation differs, and it moves accuracy
by 0.13 to 0.19.

Averaging first raises the signal-to-noise ratio of each time course before the statistic is
estimated, whereas averaging afterwards averages 86-region estimates each of which is noisy at
124 timepoints. That direction is unsurprising in hindsight. Its size is not, and it is larger
than the effect of any feature block, of any classifier family, and of external pretraining. It
belongs in the multiverse as an axis in its own right, which it currently is not: every
network-scale specification in this paper aggregates first. We flag it as a limitation of the
declared grid rather than repairing it here, since adding the axis changes every specification
identifier and would require rerunning the family-wise null.

### 3.3 Relation to attentional performance

A connectivity difference between groups need not be the difference that supports better
performance. Features that discriminate AVGPs from NVGPs could in principle be unrelated to
ability, since group membership and individual performance are distinct targets and can be
carried by different signals (van Balkom et al. 2022). Because the transfer account in section
2.1 claims that the reconfigured coupling is what makes broad improvement possible, this
distinction matters here more than it would in a purely descriptive study.

The same participants completed an attention-demanding, auditory-cued visual discrimination
task in the same scanning session (Föcker et al. 2018), giving accuracy and response time for
each individual. **Behavioural records exist for all 32 participants**, and the two groups
differ behaviourally in the expected direction: AVGPs were more accurate (0.840 against 0.750),
faster (median RT 0.921 s against 0.995 s) and lower in inverse efficiency (1.111 against
1.409). Tested rather than merely described, accuracy gives d = +0.74 [+0.11, +1.49]
(t-test p = 0.046, Mann-Whitney p = 0.032) and IES d = -0.76 [-1.37, -0.31] (p = 0.044, 0.019);
response time alone does not separate the groups (d = -0.62 [-1.38, +0.02], p = 0.088, 0.221).
Across the three measures no result survives Holm correction, the smallest adjusted p being
0.056, so the honest description is a medium-to-large effect in the predicted direction that a
32-participant design detects nominally and cannot establish conclusively. We related the two
features identified in section 3.1 to accuracy, median correct RT, and IES
(`acnets.analysis.brain_behavior`), pooling across groups and, separately, within group.

> **Correction, and a caution about small-sample joins.** An earlier version of this section
> reported 25 participants and stated that seven later-recruited replacements had no task
> records. That was wrong. All 32 have records; the seven are keyed by a different identifier
> in the two record systems — the imaging side suffixes the original number with `NEW`, the
> behavioural side renumbers it by +20 — and an inner join silently dropped them. The
> investigators confirmed the pairing, which is now written out explicitly in
> `scripts/run_brain_behavior.py` and asserted, so the same loss fails loudly rather than
> quietly. The loss was not balanced: five of the seven are AVGP, so the analysis ran on
> 11 AVGP against 14 NVGP rather than 16 against 16. **Restoring them changes the result of
> this section**, which is recorded below rather than quietly absorbed.

**On all 32 participants, three of the eighteen intervals exclude zero, and one of them is a
within-group association.** CON-SMN coupling relates to inverse efficiency pooled
(r = -0.215 [-0.508, -0.043]) and, more informatively, **within AVGPs**
(r = -0.378 [-0.682, -0.070]); CON-SMN relates to response time pooled
(r = -0.263 [-0.537, -0.028]) though not within either group; and FPN-SMN relates to accuracy
within NVGPs (r = -0.435 [-0.883, -0.126]). The signs are the predicted ones for CON-SMN:
higher control-to-sensorimotor coupling goes with lower inverse efficiency, which is better
performance. CON-SMN with accuracy, the largest association at n = 25, is now the weakest
(r = +0.199 [-0.055, +0.485]).

**Three qualifications, and they matter more than the result.** These are eighteen bootstrap
intervals with no correction between them, so roughly one exclusion of zero is expected by
chance and three is not a striking excess; they are reported as effect sizes, not tests, and
nothing here is a corrected finding. The FPN-SMN association runs in the opposite direction to
the CON-SMN ones and sits on the coupling that fails the motion control of section 3.5, so we
do not interpret it. And the result reversed on the addition of seven participants — the same
analysis at n = 25 found nothing anywhere. An association that appears when a sample grows by
28% is exactly what this paper's own reliability argument (section 3.7) says to distrust, and
we report it at that strength: **suggestive, in the predicted direction, and not established.**

The distinction between pooled and within-group association is the substantive one, and we
report both because they answer different questions. A pooled correlation can be produced
entirely by the group difference itself: if AVGPs have both higher coupling and better
performance, the two will covary across the combined sample even when coupling explains nothing
about performance within either group. Only a within-group association speaks to whether the
coupling tracks ability. With 16 participants per group, the within-group estimates are
imprecise, and we report them as effect sizes with bootstrap intervals rather than as tests.

> **Note.** This analysis is not in the original manuscript, which listed the absence of a
> behavioural link as a limitation. Under the present framing it is load-bearing rather than
> optional, because a claim about transfer that never measures performance invites the obvious
> objection. On the corrected sample the analysis no longer returns a flat null: CON-SMN
> coupling tracks inverse efficiency within AVGPs. That is the one link the transfer account
> actually requires, and it is the weakest possible version of it — one uncorrected interval
> among eighteen, on a sample where adding seven people reversed the verdict. The couplings'
> separation of the groups is itself a description of what the models do rather than a tested
> effect (section 3.4.3), so what is reported here is a weak signal beside another weak signal,
> not a demonstrated mechanism.

### 3.4 The result does not depend on the analysis path

The three results above were obtained from one analysis path, and that path was one of many
defensible ones. This section establishes that the conclusion is not an artefact of those
choices. It is a warrant for sections 3.1 to 3.3 rather than a finding in its own right, and a
reader interested only in the transfer claim can take section 3.4.3 and skip the rest.

The logic is as follows. If the coupling result were produced by a fortunate combination of
parcellation, connectivity statistic, aggregation scale and classifier, then varying those
choices should dissolve it. We therefore enumerated all 495 defensible combinations
(section 2.3), fitted every one identically, and asked two questions: how much does decoding
accuracy move, and does the attribution move with it.

### 3.4.1 Decodability depends on the analysis path


The best-performing specification — Dosenbach 2010, partial correlation, network scale,
XGBoost — classified participants as AVGP or NVGP with a mean accuracy of 0.660 over 100
folds. The two uncertainties around that number are very different sizes, which is the point
of reporting both (section 6.6). Split stability, the spread across the 100 folds, has an SD of
0.163; that quantity shrinks with computational effort and says only that the estimate does not
depend on how the sample was partitioned. Sampling uncertainty, which is what bears on
generalisation to new participants, is the 95% Wilson interval at n = 32: **[0.49, 0.80]**.
It spans chance. No amount of resampling narrows it.

Across all 49 specifications accuracy ranged from 0.385 to 0.660, with a median of 0.519
(Figure 2, specification curve). The family-wise max-statistic permutation test placed the
95th-percentile critical value at **0.739**, meaning a specification must exceed that accuracy
to be significant once the search is accounted for. **No specification exceeded it — 0 of 49.**
The best specification had a family-wise p of **0.291** over 1000 permutations, with labels
permuted outside the entire pipeline including feature selection and hyperparameter search.

The shape of the null is worth reporting alongside the p-value, because it says something a
p-value alone does not. Refitting all 49 specifications under permuted labels and keeping the
best gives a maximum accuracy of 0.625 at the median, 0.709 at the 90th percentile and 0.848 at
its most extreme. In other words, **searching 49 analysis paths over 32 participants yields
about 0.63 from nothing at all**, and yields above 0.66 — the accuracy we actually observed —
29% of the time. The observed maximum is not in the tail of that distribution; it is next to
its centre. A reader shown only `0.660, partial correlation, network scale, XGBoost` would
reasonably read it as a positive finding, and the null is what shows it is not.

> **How the result held up as the grid was built.** Three runs were made, and the prediction
> that widening the grid would not rescue the result held each time, because a wider search
> widens the null it has to clear.
>
> | grid | best | critical | p | B | significant |
> |---|---|---|---|---|---|
> | 24 specs, 2 classifiers, stratified folds | 0.725 | 0.732 | 0.114 | 200 | 0 of 24 |
> | 36 specs, + LEMON transfer, fixed folds | 0.660 | 0.720 | 0.244 | 200 | 0 of 36 |
> | **49 specs, + neural classifier, fixed folds** | **0.660** | **0.739** | **0.291** | **1000** | **0 of 49** |
>
> Only the last row is reported in the text; the others are shown because the trajectory is
> itself informative and because two changes between them would otherwise be invisible. The
> switch to label-independent fixed folds (section 6.5) lowered the observed maxima, which is
> expected: stratification was giving the observed statistic balanced folds the null did not
> get. And the classic-ML transfer axis, while placing specifications near the top of the grid,
> moves accuracy variance by only 1.6% (section 3.4.2) — the external data helps at the top
> without lifting the family. The pretrained transfer model of section 2.4 was the remaining
> route and is reported below; it is also null. See `outputs/permutation/`.

This is a null result about decodability and it should be reported as one. It does not
retract section 3.1, which is about which features the models weight, and it does not touch
section 3.4.3, which measures agreement between attributions and has its own null.

> **The pretrained transfer model (section 2.4), executed.** The specification in
> `prereg_primary.md` was fixed, then run exactly once: the LEMON-pretrained
> latent-connectivity classifier reached a mean accuracy of **0.477** over the specified 100
> folds (Wilson CI at n = 32: [0.32, 0.64]), against a permutation null with mean 0.401 and
> 95th percentile 0.514; **p = 0.0999 at B = 1000.** It is null, and — per section 2.4 — it
> carries no confirmatory licence to inherit or to pass on. Two honest observations accompany
> it. The observed statistic sits in the null's
> upper decile — the transfer moved the model into the right tail without reaching the
> threshold, which reads as consistent with a weak real signal at n = 32 and with nothing.
> And the null's centre lies below 0.5, a known behaviour of fine-tuned models under permuted
> labels on small fixed folds; the test compares against that empirical null rather than
> against nominal chance, which is the point of running one. See `outputs/primary/`.

> **Note.** On p-value resolution: with B permutations the smallest attainable p is
> 1/(B+1), so the original draft's p = 0.015 required at least 66 permutations. The interim
> results above used B = 200, giving a floor of 0.005. The reported analysis uses B = 1000
> over the full **49**-specification grid at the full 4 x 25 cross-validation budget, a floor
> of 0.001. (The grid was 54 until five cells were found to be exact duplicates; see section
> 2.3.)

### 3.4.2 Which choices matter, and by how much


**The decomposition uses balanced subgrids, and the reason matters.** A specification grid is
generally not a full factorial, and attributing variance across an unbalanced design assigns
shared variance arbitrarily between the confounded dimensions. Two confounds arise here. The
five region-only atlases (three DiFuMo resolutions, AAL, Harvard-Oxford) never appear at
network scale, so atlas and aggregation are entangled; restricting to the five
network-capable atlases restores orthogonality (`SpecificationRegistry().balanced`). And
LEMON-pretrained transforms are defined at network scale only, and within network scale only
where they are not a no-op (section 2.3), so pretraining is entangled with both aggregation
and classifier. The decomposition is therefore reported on two subgrids: the no-pretraining
subset, where statistic x aggregation x classifier is fully crossed, and the network-scale
scale-sensitive-classifier subset, where statistic x classifier x pretraining is fully
crossed. The full grid is retained for the specification curve and the family-wise null,
where imbalance is harmless because no dimension is being attributed variance.

> **Result on the 49-specification grid** (one atlas; six statistics; network and
> random-network aggregation; linear SVM, gradient-boosted trees and a small neural network;
> with and without LEMON-pretrained transforms). No-pretraining subset, 36 specifications,
> R² = 0.54: **aggregation scale 44.4%**, statistic 6.6%, classifier 3.3%. Pretraining
> subset, 24 specifications, R² = 0.82: **statistic 75.1%**, classifier 5.2%, **pretraining
> 1.6%**. Atlas contributes nothing in either because only one atlas was run.
>
> Three things are worth stating. The dominant choice overall is whether real network
> structure is used at all — real-network specifications sit systematically above their
> shuffled controls, which anticipates the dimensionality-matched control below. Once that is
> held fixed, the connectivity statistic is what matters: three quarters of the remaining
> variance. And **the model family barely matters** — a linear decoder, a tree ensemble and a
> neural network span 3-5% between them. At this sample size, reaching for a more expressive
> model is not where the leverage is; varying the model family buys diversity of attribution
> method, not of accuracy. External pretraining moves less still, 1.6%, and does not appear in
> the best specification. See `outputs/agreement/variance.json`.

These figures replace the original manuscript's unquantified claim that methodological choices
matter with a statement about which ones and how much. They also carry a warning about their
own generality: a decomposition is a statement about the grid it was computed on, and a grid
containing one atlas cannot say that the atlas does not matter.

These components are descriptive. Every specification is fitted to the same 32 participants,
so accuracies are correlated across specifications and the components mix genuine sensitivity
to analytic choices with sampling noise common to all of them. No significance tests are
reported for the decomposition, and `acnets.multiverse.variance_decomposition` returns none.

We chose the Dosenbach parcellation because it is grounded in cognitive control research
and thus relevant in this context. One might ask whether similar results would be obtained
without grouping data into networks, or by grouping the same set of regions into different
networks of the same complexity, where the model input has identical dimensionality but the
mapping of regions to networks is shuffled.

Both questions are answered here, but they are not equally informative and the original
draft conflated them. Region-level and network-level specifications differ in
dimensionality by roughly a factor of 600 for the Dosenbach atlas: 160 regions give 12,880
edges against 24 training participants, whereas six networks give 21. The region-level
result therefore speaks to the curse of dimensionality at this sample size, not to whether
regional information exists. Only the **random-network control is dimensionality-matched**,
and only it licenses the inference that the atlas aggregates noisy time courses in a
meaningful way. Comparing the 18 exactly matched pairs — same statistic, same classifier, same
pretraining, same folds, differing only in whether the region-to-network assignment is the
atlas's or a shuffle — random-network specifications achieved a mean accuracy of 0.455 against
0.564 for their real-network counterparts, with the real assignment ahead in 15 of the 18
pairs. The shuffled controls also sit essentially at chance, so the gap is not an artefact of
the real networks having easier features to fit.

This is the paper's clearest positive result, and it is worth being precise about what it
licenses. It says the Dosenbach region-to-network mapping carries information that a random
partition of the same regions into the same number of groups does not. It does not say the
groups are separable: the same 31 real-network specifications fail the family-wise null
(section 3.4.1). What survives is the claim that the aggregation is meaningful, not the claim
that the contrast is decodable.

### 3.4.3 Attribution is more stable than accuracy, and the stability is not evidence


Every specification yields one attribution vector over its feature vocabulary, computed by
whichever method its model class admits (section 6.7): Haufe-transformed forward patterns for
the linear specifications, mean absolute SHAP values for the tree ensembles, and
gradient-times-input for the neural ones. Haufe et al. (2014) show that the weights of a
backward model such as an SVM cannot be interpreted in terms of brain processes, because the
extraction filter depends on every noise component in the data; the forward transform is
required before a weight can be read as a claim about the brain.

**Making the comparison fair.** The claim that explanation is more stable than prediction is
easy to state and easy to get wrong, so we set out the measurement before the result. Two
problems have to be handled.

First, raw agreement has a floor that depends on the number of features. Because every
specification ranks the same feature vocabulary, two entirely random rankings already agree
substantially under a top-weighted measure: for rank-biased overlap with p = 0.9, random
rankings of 6 features agree at 0.78, of 21 features at 0.43, and of 50 features at 0.20
(`acnets.multiverse.random_agreement_baseline`). An uncalibrated agreement statistic is
therefore uninterpretable, and its apparent size can be manufactured simply by choosing a
coarser parcellation. This is the attribution-space analogue of the argument for empirical
rather than theoretical chance levels in classification (Combrisson and Jerbi 2015).

Second, and less obviously, accuracy and attribution are not natively commensurable.
Comparing the coefficient of variation of a continuous accuracy against the agreement of a
coarse ranking invites the objection that rankings only appear stable because they carry
less information. We therefore express both as **pairwise agreement normalised against its
own empirical null**, giving two numbers in the same units where 0 is chance and 1 is
perfect (`acnets.multiverse.stability_comparison`):

| | agreement statistic | null |
|---|---|---|
| attribution | mean pairwise RBO | random rankings of the same vocabulary |
| accuracy | mean pairwise 1 - abs(a_i - a_j) | chance-level classifiers on n = 32 |

The accuracy null is not zero disagreement: two genuinely chance-level classifiers evaluated
on 32 participants differ by an amount set by binomial noise, and that is the reference the
observed accuracy agreement must beat.

**Result.** Over the 31 network-scale specifications and their shared vocabulary of 21
network edges, raw agreement was RBO 0.634 (bootstrap 95% CI [0.596, 0.673]) against a
random-ranking floor of 0.423 — normalised stability **+0.366** for attribution (p = 0.019)
against **+0.237** for accuracy, a difference of +0.128 in favour of explanation. Restricting
to the 26 specifications using symmetric statistics, which need no directed-to-undirected
collapse, gives the same picture slightly more strongly (+0.373 against +0.178). The
dimensionality-matched control behaves as it should: over the 18 random-network specifications
attribution stability falls to +0.154 (p = 0.135, not distinguishable from random rankings)
while accuracy stability *rises* to +0.484, reversing the ordering. Shuffling the
region-to-network mapping therefore destroys exactly the property being claimed, and leaves
the property that is not being claimed intact.

**Does the agreement have anything to do with the labels?** This is the objection the section
has to answer, and the random-ranking null above cannot answer it. Since no specification
decodes above a corrected null, a reader is entitled to suspect that the models are agreeing
about something other than the group contrast. The suspicion has a precise mechanism behind
it rather than being a vague worry about explainable AI. The Haufe pattern is
A = Σw / (w'Σw), so as the weight vector w degenerates toward noise the pattern degenerates
toward Σw, and Σ — the covariance of the 21 edges — is a label-free quantity that every
specification using the same statistic shares. On that account the multiverse would agree
because it is reading the same covariance structure every time, and would agree just as well
if the labels were nonsense. Two controls test this, and between them they yield three
statistics: agreement against label-free geometry, agreement against permuted labels, and the
identity of the converged-on feature against permuted labels (section 6.7).

**Label-free feature geometry does not reproduce the consensus**
(`scripts/run_attribution_geometry.py`). Ranking the same 21 edges by four properties that
never see a label — variance across participants, mean absolute value, loading on the leading
eigenvector of Σ, and the mean magnitude of Σw over 200 random draws of w, which is exactly
the pattern a chance-level linear decoder yields — gives rankings that agree with the observed
multiverse consensus at RBO 0.278 to 0.461 across all six statistics and all four criteria.
The random-ranking floor for the same comparison is 0.408 [0.282, 0.611], so **every
label-free ranking falls inside the interval two random rankings occupy**, against 0.634 for
the observed cross-specification agreement. The criteria also disagree with the consensus
about content, not merely about degree: they put DMN-SMN, DMN-FPN or the within-network
diagonals on top, and CON-SMN never ranks first under any of them — its best label-free rank
is 3rd of 21 and its median across the twenty-four cells is 7th. The edge the classifiers
converge on is therefore not the edge the covariance would hand a model that had learned
nothing.

**The stricter control passes, but only because the multiverse spans model families, and that
qualification is the finding.** (`scripts/run_attribution_null.py`.) Refitting the grid on
permuted labels — same folds, same cached features, same attribution methods, only the group
assignment destroyed — was run twice, over two nested grids.

| grid | specifications | observed | permuted-label null | excess | p | B |
|---|---|---|---|---|---|---|
| linear only, Haufe channel | 6 | 0.630 | 0.601 (SD 0.062) | +0.029 | **0.29** | 1000 |
| **all three classifiers** | **18** | **0.657** | **0.562 (SD 0.048)** | **+0.095** | **0.035** | **200** |

The second row is the one that corresponds to the analysis reported above, whose 31
network-scale specifications span the linear, tree and neural channels; the first is a subgrid
run because the Haufe channel has the sharpest theoretical rival. Read on the grid the paper
actually uses, **attribution agreement does survive its permutation null.** Read on a single
model family, it does not.

**The direction of that difference is not noise, and it says something about multiverse design
that we did not expect.** Adding the tree and neural specifications *raises* the observed
agreement, 0.630 to 0.657, and simultaneously *lowers* the permuted-label null, 0.601 to 0.562.
Both movements have the same explanation. Under permuted labels there is nothing to find, so
each specification fits noise; specifications from the same model family fit that noise in
nearly the same way, and their attributions agree strongly for reasons entirely internal to the
family. An L1 SVM, a gradient-boosted ensemble and a perceptron overfit the same 32
participants *differently*, so a grid spanning all three agrees much less under permutation.
Under real labels, a signal that is genuinely in the data is available to all three, so
agreement holds up. Classifier diversity therefore lowers the null without lowering the
observed statistic, which is exactly the property that makes a null informative.

The practical lesson is sharper than the general advice to "vary the classifier". A multiverse
built from one model family cannot distinguish a reproducible finding from a reproducible
fitting procedure, because its specifications share the inductive bias that produces the
agreement. Section 3.4.2 found that the model family contributes only 3-5% of accuracy variance
and concluded that varying it "buys diversity of attribution method, not of accuracy". That
now reads as an understatement: the diversity it buys is what makes the attribution claim
testable at all, and a grid that had economised on the classifier axis — as we were tempted to,
precisely because it moved accuracy so little — would have returned p = 0.29 and a withdrawn
finding.

Two limits on the surviving result. The three-classifier null was run at B = 200 against the
linear subgrid's B = 1000, so its resolution is coarser and p = 0.035 rests on 6 permutations of
200 reaching the observed value; and 18 specifications is not the full 31, which additionally
includes the LEMON-pretrained cells. Neither changes the sign, and both are cheap to extend.

**But the same null separates two claims we had been running together, and it partly clears
one of them.** How *strongly* a grid agrees and *what* it agrees on are different quantities with
different nulls, and only the first is measured by mean pairwise RBO. A permutation can produce
specifications that agree tightly with each other — they overfit the same participants along
the same folds — while landing on a different leading feature every time. Recording which
feature each permutation's grid puts first tests the substantive claim of section 3.1 rather
than the statistical one.

Under the real labels the consensus leader is **CON-SMN**. Across 1000 permutations the
consensus leader is CON-SMN in **62**, giving **p = 0.063**, against a uniform rate of 48 in
1000 over 21 edges. No other edge is common either — the most frequent permuted leaders are
SMN-VIS (90), DMN-FPN (86), CER-FPN (74), CER-VIS (74) and DMN-SMN (73) — so shuffled labels do
not merely fail to select CON-SMN, they select nothing consistently. **The identity of the
converged-on feature is therefore label-dependent on the very grid whose agreement magnitude
is not** — this test was run on the linear subgrid, the one that returns p = 0.29 for
magnitude. We report p = 0.063 as what it is: short of any conventional threshold and computed
after the fact rather than declared. Its value is that it points the same way as the
three-classifier magnitude result from a grid that fails the magnitude test, so the two do not
depend on the same property of the multiverse. [[TODO: the identity test on the
three-classifier grid was not run — the leader analysis was added to
`run_attribution_null.py` after that job had started. It is one rerun and should be reported.]]

**The methodological lesson generalises past this dataset, and we think it is the most useful
thing in this section.** A multiverse that reports attribution agreement against a
random-ranking baseline is testing against a null that no realistic alternative occupies: the
floor is 0.423 and a grid fitted to shuffled labels reaches 0.562 to 0.601 depending only on
how diverse the grid is. The comparison that carries information is against a grid refitted on
permuted labels, which holds the sample, the folds, the feature space and the attribution
machinery fixed and varies only the thing whose contribution is in question. That null is not
expensive — features are label-independent and cache across permutations, so only the
classifier refits are paid for. Had we reported the original +0.366 against random rankings
without it, we would have reported a number whose null was 40 points too low, and we would not
have known that the result depends on the classifier axis at all.

The second lesson is the one we would put in a methods paper. **Agreement magnitude and
consensus identity need separate nulls, and they fail independently.** Magnitude is dominated
by how much the specifications share — sample, folds, feature space and, decisively, inductive
bias — and within one model family at small n that is most of what it measures. Identity asks
whether the shared thing is the labels, and it is the quantity a reader actually cares about
when a paper says "every pipeline pointed at the same network". On the linear subgrid the two
diverge sharply, p = 0.29 against p = 0.063, so a paper reporting only magnitude would have
discarded a result that identity supports. Report both.

> **Scope of this null, stated exactly.** Both grids are network-scale, one atlas, six
> statistics, no pretraining; they differ only in the classifier axis. Neither is the full
> 31-specification set of the agreement analysis, which adds the LEMON-pretrained cells; the
> three-classifier grid's observed 0.657 brackets that set's 0.634 from above and the linear
> grid's 0.630 from below. [[TODO: extend the three-classifier null to B = 1000 and to all 31
> specifications before submission. At 32 s per permutation this is roughly nine hours and
> needs no new code — `run_attribution_null.py --all-classifiers -B 1000`. Nothing in the text
> anticipates a different sign, but p = 0.035 on 6 of 200 permutations deserves the
> resolution.]]

**The corroborating check fails, and that bounds the claim.** Kendall's tau, which weights
all ranks equally and is reported precisely as a guard against the conclusion being an
artefact of top-weighting, gives 0.083 with a normalised excess of +0.085 and p = 0.30 — not
distinguishable from chance. Read together with the selection frequencies in section 3.1, the
honest reading is narrow: **the top of the ranking is reproducible and the rest of it is
not**. CON-SMN appears in the top five of every specification and FPN-SMN in 81% of them,
after which frequencies fall away (CER-VIS 0.58, then a long tail). The claim this section
supports is that the multiverse agrees on *which two couplings matter most*, not that it
agrees on an ordering of the 21 edges.

**Agreement is not an artefact of conditioning on accuracy.** A specification performing at
chance has essentially arbitrary attributions, so restricting attention to well-performing
specifications will raise agreement mechanically, and specifications that decode well may do
so precisely *because* they recovered the same signal. Reporting agreement at a single
accuracy cutoff would hide this. Figure 3 therefore gives agreement as a function of the
cutoff, from zero (all specifications) upward
(`acnets.multiverse.agreement_vs_threshold`). Agreement at threshold zero was 0.634 across
all 31 network-scale specifications, moving to 0.646 at a 0.50 cutoff (27 specifications),
0.663 at 0.55 (20), 0.657 at 0.60 (12) and 0.681 at 0.65 (3). The curve is essentially flat
across the range where the specification count is large, and is not monotone, so **the
conclusion does not depend on the selection**: the modest rise at the far right involves only
three specifications and is what the mechanical argument above predicts.

**Attribution channels, and a caveat about independence.** Attribution is computed by the
method each model class admits: Haufe-transformed forward patterns for the linear
specifications, mean absolute SHAP values for the tree ensembles, and closed-form
gradient-times-input for the neural specifications. These are not three independent witnesses
to the same quantity, and it would overstate the evidence to treat them as such. SHAP and
gradient-times-input are both computed from the fitted model's own response surface and are
expected to correlate by construction. The Haufe pattern is derived differently, from the
model weights and the data covariance rather than from prediction perturbation, so agreement
between Haufe and the other two is the more informative comparison — and it is what the
across-classifier agreement reported above principally measures, since the 31 network-scale
specifications divide 12 Haufe / 7 SHAP / 12 gradient-times-input between the three channels,
so 67% of the 465 pairs in the agreement matrix span two different attribution methods and 63%
involve the Haufe channel. The agreement being measured is therefore mostly *across* methods,
not within one.

### 3.4.4 What would disconfirm the robustness claim


Stating the falsification condition in advance guards against the result being unfalsifiable
after the fact. The thesis fails if any of the following holds:

1. Normalised attribution stability does not exceed normalised accuracy stability. This is
   the direct test, implemented as `stability_comparison(...)['thesis_supported']`, and it
   returns False on data where attribution is noisy and accuracy is tight.
2. Attribution agreement does not exceed the random-ranking baseline at threshold zero,
   which would mean the apparent agreement is entirely an artefact of the shared feature
   vocabulary.
3. Agreement rises steeply and only with the accuracy cutoff, which would mean it is a
   by-product of selecting well-performing specifications rather than a property of the
   attributions.
4. Haufe patterns disagree with the perturbation-based channels, which would suggest the
   apparent stability is a property of one attribution algorithm rather than of the models.
5. A label-free ranking of the same features reproduces the consensus, which would mean the
   multiverse is agreeing about the covariance of the feature space rather than about the
   groups.
6. Refitting the grid on permuted labels reproduces the observed agreement, which would mean
   the same thing in the stricter form that lets the classifiers keep their fitting
   machinery.

We report all six, and **two of them fail**. On the data: (1) holds — +0.366 against +0.237;
(2) holds — 0.634 against a 0.423 floor, p = 0.019; (3) holds — the threshold curve is flat
and non-monotone, 0.634 to 0.663 across the range with usable specification counts; (4) holds
in the sense tested, since the across-classifier agreement spans three attribution algorithms
that include the covariance-based Haufe channel; (5) holds — label-free rankings agree with the
consensus no better than random rankings do; **(6) holds on the grid the paper uses and fails
on a single-family subgrid** — permuted-label refits reach 0.562 against an observed 0.657
across three classifier families (p = 0.035) but 0.601 against 0.630 within the linear family
alone (p = 0.29) — and the identity of the converged-on feature passes weakly, since permuted
linear grids put CON-SMN first in 62 of 1000 (p = 0.063).

A seventh condition was already known to strain the data and remains recorded here. The claim
as stated concerns agreement *at the top* of the ranking, which is what rank-biased overlap
measures. It would be a materially stronger claim that the whole ordering is stable, and that
claim fails too: Kendall's tau over all 21 edges is indistinguishable from chance
(section 3.4.3).

**What survives this list is narrower than the claim we set out to test, and we state the
revision rather than reinterpreting the conditions to fit.** Conditions (1) to (5) establish
that attribution agreement in this multiverse is high, is not manufactured by the shared
feature vocabulary, is not an artefact of selecting good specifications, is not confined to
one attribution algorithm, and is not predictable from the label-free covariance of the
features. Condition (6) establishes that it is **also distinguishable from what the same grid
produces on shuffled labels, provided the grid spans model families** — and that the
qualification is doing real work, since the single-family version of the same test returns
p = 0.29.

The conclusion is therefore that the attribution result stands, on a narrower base than the
original +0.366 suggested and with an explicit dependence nobody would have predicted. What
supports it is a 0.095 excess over permuted labels across three classifier families
(p = 0.035), corroborated by the converged-on feature being CON-SMN where permuted grids
select it 6% of the time (p = 0.063). What does not support it is the raw agreement against
random rankings, whose null is roughly 40 points too low to be informative. Stating the
falsification conditions in advance did its job in an unexpected way: condition (6) was added
because we expected it to fail, it failed on the first grid we ran it on, and running it
properly is what revealed that the classifier axis — which contributes almost nothing to
accuracy (section 3.4.2) — is what the attribution claim depends on.

### 3.4.5 The same model, priced by how hard we looked


The registry in section 2.3 fixes a family of 49 and section 3.4.1 prices it. But a reader is
entitled to ask what the correction is doing — whether it is a formality or whether it changes
conclusions. It changes them, and the clearest way to show that is to hold a result fixed and
vary only the declared family around it.

The result held fixed is precision connectivity restricted to the three networks of section
3.1 — FPN, CON and SMN — classified by shrinkage discriminant under leave-one-out. This is
*not* one of the 49: it is region-scale rather than network-scale and uses a different
estimator, so it is an auxiliary analysis, not a member of the primary family. It reaches
**0.812**. Four families were declared around it after the fact, each a defensible answer to
"what else could we have run":

| declared family | size | best in family | p |
|---|---|---|---|
| structure/function fusion combinations | 7 | 0.812 | 0.0010 |
| network triples, precision only | 20 | 0.812 | 0.0066 |
| statistic × network triple | 80 | 0.812 | 0.0399 |
| every modality, model and restriction tried | 833 | 0.880 | 0.2376 |

**None of these four p-values is ours to claim, and the reason is worth stating carefully,
because it is the single easiest way for a paper like this to mislead.** A family-wise
correction prices the search you declare. It cannot price a search you have already run and
whose answer you already know. FPN, CON and SMN were not chosen from prior literature — they
were chosen because section 3.1 found the classifiers leaning on them, in these same 32
participants (section 1.5). Restricting to them and then scoring on the same participants
selects and tests on one sample. Every row above therefore understates the true cost by an
amount nobody can compute, and the first row — a family of 7, p = 0.0010 — understates it
most, precisely because it looks best.

What the table does show, and shows well, is the *shape* of the dependence. The same analysis
is worth a factor of forty in p depending on a choice — how wide to draw the family — that no
fitting procedure sees and no cross-validation detects. In the widest family a *higher*
accuracy is *less* significant than a lower one in a narrow family: 0.880 at p = 0.24 against
0.812 at p = 0.0066. And most usefully for anyone reading a single reported number, searching
833 specifications over these participants produces 0.80 accuracy **at the median of the
permutation null** — not in its tail. Twenty-eight cells reach 0.80 and thirteen reach 0.84;
none survives, because none needs to be real to happen.

So the 0.812 enters this paper as a **hypothesis generated by it**, not as a finding it
establishes: the restriction that best separates these two groups involves control-to-
sensorimotor coupling, which is worth testing in a sample that did not generate it. Two
checks bound what it can already be claimed to mean, and both are reported below. Restricting
to those three networks is worth nothing over restricting anywhere else of the same size
(section 3.5), so the gain is dimensionality rather than localisation. And the networks do
not transfer: discovering them in one dataset and modality and testing them in the other
fails in both directions (section 3.6). A preregistered test of this hypothesis, in new data,
is the analysis this study licenses and does not perform — and nothing in this paper, section
2.4 included, is such a test.

### 3.5 Negative controls and noise ceiling


**Motion does distinguish the groups, and this is the paper's most serious confound.** An
earlier draft asserted the opposite. That assertion had never been tested; when we tested it,
it failed, and the correction changes how section 3.1 should be read.

A classifier receiving only per-participant summaries of fMRIPrep's motion confounds — mean
and SD of the six realignment parameters, framewise displacement, DVARS and RMS displacement,
20 features, on the same label-independent folds — reached **0.608** (linear SVM; 95% Wilson
CI [0.44, 0.76]; XGBoost and the perceptron both 0.606). That is higher than the median
connectivity specification (0.519) and close to the best one (0.660). Univariately, action
gamers moved more: framewise-displacement SD d = +0.89 (p = 0.019), RMS-displacement SD
d = +0.86 (p = 0.024), DVARS SD d = +0.80 (p = 0.030). Three of 20 channels are nominally
significant and none survives Holm correction, so the univariate evidence is suggestive rather
than conclusive — but "does not survive correction at n = 32" is not the same as "absent", and
the multivariate result is what matters here (`outputs/controls/motion.json`).

**The confound predicts the observed attribution, so we tested that directly.** Head motion
inflates apparent coupling most in sensorimotor cortex, and the feature the multiverse agrees
on is CON-SMN. Standard confound regression — nilearn's `simple` strategy, six realignment
parameters plus white-matter and CSF signals — is known not to remove the participant-level
relationship between motion and connectivity (Power et al. 2012; Satterthwaite et al. 2012),
so we measured that relationship rather than assuming it away
(`scripts/run_motion_coupling.py`).

It is substantial. Across statistics, the CON-SMN edge correlates with framewise-displacement
SD at r = +0.57 (correlation), +0.46 (partial correlation), +0.56 (covariance), +0.54
(tangent) and −0.50 (precision, where the sign flip is expected of an inverse-covariance
statistic), all with p ≤ 0.008. Between a third and two thirds of all 21 network edges
correlate with motion at p < 0.05.

The group effect on CON-SMN nevertheless survives partialling motion out, in four of the six
statistics: p = 0.031 (correlation), 0.025 (partial correlation), 0.002 (precision), 0.027
(tangent). It does not survive for covariance, which had no group effect to partial. FPN-SMN,
the second-ranked feature, does **not** survive in any statistic (p = 0.17, 0.22, 0.10, 0.14),
so its prominence in the attribution rankings is plausibly a motion artefact. None of these
p-values is corrected for 21 edges; at a Bonferroni threshold of 0.0024 only precision's
CON-SMN would remain.

The honest summary is therefore three-part. Motion is a real group difference in this sample
and cannot be dismissed. The leading attribution feature retains a nominal association with
group after motion is controlled, in most statistics. And the second feature does not. This is
recorded again in the limitations (section 4.6), and it is one more reason the paper's claim
is about the stability of an explanation rather than about a demonstrated group difference.

**The spatially-displaced control runs, and it constrains what restriction is worth.** A
classifier receiving only cerebellar connectivity is the negative control for any claim that
restricting to particular networks helps *because of which* networks they are. This was
previously recorded as blocked on the DiFuMo-256 parcellation; it is not, since Dosenbach2010
already assigns 18 of its 160 regions to the cerebellum. Nothing in section 1 predicts an
action-game effect there.

The comparison is held at 18 regions throughout, which is the point of it: an unmatched
contrast would confound *where* with *how many*. Within-cerebellum precision connectivity
(153 edges) gives 0.469 — below chance, as a displaced control should be. But 200 random
18-region draws from FPN+CON+SMN give a median of 0.469 as well, identical to 200 draws from
anywhere non-cerebellar (0.469), and the cerebellum sits at the 41st and 35th percentile of
those two distributions respectively. **At region scale, restricting to the three networks of
section 3.1 is worth nothing over restricting anywhere else.**

Two things follow. The restricted result of section 3.4.5 is about *aggregation to network
scale*, not about localisation: six network edges over three networks is a different object
from 153 region edges within the same territory, and only the former helps. This is the
narrower of the two readings available, and it is the one the data support — any sentence
claiming those three networks are where the signal lives is not supported by this control.
And the draws are themselves a lesson in search width: the best of 200 random 18-region
subsets reaches 0.750, from regions chosen for no reason whatever.

**The noise ceiling is not available, and we say so rather than substituting a weaker
number.** This is a different quantity from the *reliability* ceiling of section 3.7, which
is computed and is one of this paper's main results; what is missing here is the positive
control that would say what this pipeline family attains when a real effect is present. The
intended calibration was to run the same specification grid on MPI-LEMON
subsampled to n = 32, classifying a contrast with known signal — sex is the usual choice — so
that the observed accuracies could be read against what this pipeline family attains when a
real effect is present, rather than only against the 50% theoretical chance level. Our cached
LEMON parcellations carry timeseries and atlas coordinates but no participant metadata, so no
such contrast can be constructed from them; obtaining it means a fresh fetch from the MPI
archive. The analysis is specified and blocked, not abandoned.

Two weaker references are available in the meantime. The family-wise permutation null gives the
*floor* — the distribution of the maximum accuracy this grid produces when no signal exists
(section 3.4.1) — which is the more important of the two quantities for the claims made here.
And the behavioural data establish that n = 32 is not hopeless in principle for this sample:
the groups separate on task performance (accuracy 0.840 against 0.750, inverse efficiency
1.111 against 1.409; d = +0.74 and −0.76, nominally significant, section 3.3). What is missing is specifically the connectivity-pipeline
ceiling.

### 3.6 A structural arm: expert versus non-expert action gamers


An independent structural dataset — 24 expert and 23 non-expert action video game players,
T1-weighted MRI, different scanner and population from the resting-state sample — was run
through the same inferential machinery. Nothing is pooled with the resting-state arm.

> **This section was substantially rewritten after two defects were found in the pipeline
> that produced its original numbers.** The grey-matter volumes were skull-contaminated
> (`ants.get_mask` does not strip skull from whole-head images, giving a 2,127 cm³ "brain"
> and an inflated 785 cm³ "grey matter"), and they were read through the Dosenbach 2010
> sphere atlas, whose 10 mm spheres give near-arbitrary volumes under affine registration
> between acquisitions. The two datasets' regional profiles correlated at **−0.41** under
> that atlas and **+0.87 to +0.94** under volumetric ones. Everything below uses corrected
> volumes and volumetric atlases; `docs/paper/decodability_ceiling.md` §18–20 documents both
> defects and the fix.

**Method.** Native-space regional grey-matter volumes: N4 bias correction, brain extraction
by warping a conservative MNI brain mask through the registration, brain-to-brain affine
registration, Atropos three-tissue segmentation, atlas labels warped into each participant's
space, volumes normalised by total grey matter (`scripts/segment_t1.py`). Three volumetric
atlases — AAL, Harvard-Oxford, Schaefer-200. Per-participant brain-mask volume is recorded
and flagged outside 900–2,000 cm³; all 79 participants across both datasets passed. Affine
registration remains a declared limitation.

**Decoding: null, and the correction strengthened it.** Within gong2015, leave-one-out
accuracy is 0.574 (corrected) against 0.516 (contaminated) — both null. Within Julia2018's
structural data, 0.438. Running the arm's own specification grid — three volumetric atlases
by three classifiers, the same fixed folds and one family-wise max-statistic null over all
nine — gives a best of 0.582 (Harvard-Oxford, neural network) against a critical value of
0.664: **p = 0.30, 0 of 9 specifications clearing the threshold** (B = 500). The corrected
volumes moved the arm's best cell up by 0.066 and left its verdict where it was. Pooling the two datasets to 79 participants, with site harmonised
inside training folds, gives at best 0.544 (p = 0.27), and site itself is predictable at
0.96–0.99, which is the variance the harmonisation must remove.

**Pooling requires equating two contrasts that are not the same contrast, and we state the
assumption rather than burying it in a method.** gong2015's non-experts are *amateur action
gamers*; they play. Julia2018's NVGPs do not. Treating them as one class assumes action-game
exposure is a single dimension along which "amateur" sits closer to "non-player" than to
"expert", which is an empirical claim this study cannot check. If it is false, pooling adds
noise rather than participants, and any null it produces is ambiguous between no effect and a
wrongly merged class. What makes pooling defensible at all is that site is not confounded with
group — each dataset is internally balanced, 24/23 and 16/16 — so a classifier cannot use site
as a proxy for the label. Every pooled result in this paper, here and in section 3.8, carries
this assumption.

**External validation: six independent tests, all at chance.** Training on one dataset and
predicting every participant of the other requires no multiverse correction, because nothing
about the target informs the fit. Across three atlases and both directions, transfer accuracy
ranges 0.489 to 0.511 against null medians of 0.500 to 0.511. The feature spaces are
*verified* to correspond first, by a check that fails loudly below r = 0.5: the mean regional
volume profiles agree at r = +0.94 (Harvard-Oxford), +0.87 (AAL) and +0.58 (Schaefer-200),
the last lower because 200 small parcels are individually noisier than 48 large ones. So this
null cannot be dismissed as incomparable features.

**Attribution: the original claim is withdrawn, not re-run.** The earlier text reported
strong attribution agreement (+0.388, +0.406) and concluded that sensorimotor cortex leads
while the insula is absent — bearing directly on Gong et al.'s published insula finding. That
analysis was computed over 160 Dosenbach spheres from skull-contaminated volumes, and
sensorimotor cortex sits at the vertex where scalp contamination is largest. Attribution
ranks from a chance-level classifier describe noise in any case. **No claim about which
structures matter is made from this arm.**

A direct test confirms the ranking does not generalise. Discovering networks by Haufe
attribution in one dataset and modality, then testing them in the other, fails in both
directions: gong2015 structure → Julia2018 function gives 0.562, rank 10 of 20 network
triples (p = 0.71); Julia2018 function → gong2015 structure gives 0.511, rank 16 of 20
(p = 1.00). The functional side of that test does independently rank CON and SMN first and
second, reproducing section 3.1 — but the ranking carries no information across the gap.

**What the arm now shows, and what it is not.** Not that the dissociation replicates — that
claim rested on the withdrawn attribution result. What it shows is that action-game expertise
is not decodable from regional grey-matter volume in an independent sample, does not transfer
between samples, and does not become decodable when the samples are pooled.

**This is not a failed replication of Gong et al. (2015), and should not be cited as one.**
Their published finding is about resting-state *functional* connectivity, concentrated in the
left insula. What their authors generously shared, and what is analysed here, is the
*T1-weighted anatomical* series from the same session. Those are different measurements, and
a null for regional volume neither supports nor undermines a functional connectivity result;
the functional data that would test their claim directly were not part of this reanalysis. The
question answered here — is action-game expertise readable from structural volume — is one
their paper did not ask, and the answer at n = 47 is no. We state this plainly because the
alternative reading is both wrong and unkind to a group whose data-sharing made the only
external check in this study possible.

### 3.7 A reliability ceiling, and why it makes this design nearly unfalsifiable


Everything so far reports what the data do not show. This section asks the prior question —
how much *could* they show — and the answer explains the rest of the results better than any
of the substantive accounts do.

**Connectivity edges in this sample are about as reliable as the literature says.** Cutting
each participant's run in half, estimating connectivity on each half, correlating across
participants and applying the Spearman-Brown correction back to full length gives a median
edge reliability of **0.707** for correlation and 0.625 for partial correlation in Julia2018.
Expressed as an intraclass correlation the median is **0.283**, against the 0.29 mean that
Noble et al. (2019) report across a decade of test-retest studies. Two external corpora
bracket it: MPI-LEMON gives 0.367 (correlation) and 0.230 (partial correlation), ABIDE
controls give 0.732 and 0.570. The three corpora agree with each other only moderately
(r = 0.25 to 0.69 across edge profiles), which is itself worth stating — edge reliability is
not a fixed property of an edge but of an edge in a protocol.

**Unreliable features impose a hard cap on accuracy, and the cap is close to what everyone
reports.** Attenuation by measurement error means a group difference of a given true size
appears smaller in proportion to the square root of reliability, and the accuracy attainable
from attenuated features can be computed rather than guessed. At the measured reliability the
cap is approximately **0.72**. The literature's familiar 0.65-0.75 band for this kind of
contrast is what a ceiling looks like, not what an effect looks like.

**At n = 32 that ceiling coincides with the family-wise critical value, and the design cannot
distinguish a true effect from noise.** The max-statistic null over the 21-edge feature space
puts its 95th percentile at **0.739**. The reliability ceiling is 0.72. **A perfectly real
effect, measured perfectly given these instruments, would land below the threshold that
searching this many analysis paths already requires it to clear.** This is not a property of
our result; it is a property of the design, and it holds for any study of this size and scan
length regardless of what it finds. We think it is the single most useful thing this project
established, and section 5 returns to it.

**Weighting features by their reliability does not rescue the design, and nearly does.** A
12-cell grid crossing four statistics with three weighting schemes — plain, inverse-variance
weighted by reliability, and restricted to reliable edges only — peaks at **0.750**
(precision, reliability-weighted) against a critical value of 0.750: **p = 0.054, one cell
tied with the threshold**. Weighting helps, by about 0.03, and lands exactly on the bar. This
is the closest anything in this project comes to clearing a corrected null on a
pre-specifiable choice, and it is not close enough to report as a positive result.

**A parametric estimate agrees, and exposes what feature selection is worth.** Treating the
two groups as Gaussian and computing the bias-corrected Mahalanobis distance (Lachenbruch
1968) converts a group separation directly into the accuracy an optimal classifier could
reach:

| feature set | naive D² | bias-corrected D² | Bayes accuracy |
|---|---|---|---|
| all 21 network edges | 5.87 | **−1.06** | 0.50 |
| the 2 attributed edges | 2.58 | 2.07 | 0.764 [0.679, 0.889] |
| CON-SMN alone | 1.37 | 1.15 | 0.704 [0.578, 0.836] |

The first row is the important one. Over 21 edges at n = 32 the bias correction takes the
separation *negative* — the apparent separation in the full feature space is entirely
explained by estimating 21 dimensions from 32 participants, with nothing left over. The
second and third rows survive correction, but they are computed on edges chosen by looking at
these data, so they inherit the problem of section 3.4.5 in a form that can be measured
exactly: selecting the top two edges on all 32 participants and cross-validating gives 0.750,
while selecting them *inside* each training fold gives **0.585**. The 0.165 difference is the
price of circularity, measured. CON-SMN is recovered by in-fold selection in **94 of 100
folds**. This is the one stability statement in the paper that the permuted-label null of
section 3.4.3 does not undercut, because in-fold selection uses the labels on each training
fold and so cannot be reproduced by a grid fitted to shuffled ones. It is also much narrower
than the cross-specification agreement it replaces: it says one feature is consistently
identified within this sample, not that identifying it separates the groups. [[TODO: run the
in-fold selection frequency under permuted labels as well; at 21 edges the chance rate for any
one edge appearing in the top two is about 10%, so 94 of 100 is far above it, but the empirical
figure would be better than the analytic one.]]

### 3.8 Everything else that was tried


A reader is entitled to know whether the null survives more modelling effort. It does. This
section is a compressed record of the attempts, so that the negative result is read as the
outcome of a serious search rather than of a thin one. Accuracies below come from different
estimators and participant subsets and are **not comparable to each other**; each is stated
with what produced it. None clears a null it is entitled to claim.

**More classifiers, and tuning them, does nothing.** Thirteen classifier families — linear
and RBF SVM, logistic with and without elastic net, ridge, LDA, QDA, Gaussian naive Bayes,
k-NN, random forest, extra trees, XGBoost and a neural network — were run with and without
nested randomised hyperparameter search (20 draws, inner 3-fold), 74 cells in total. On the
honest 21-edge feature space accuracies span 0.56-0.67 and **tuning has a mean effect of
−0.003** (median −0.006, best +0.069, worst −0.034; 13 of 37 cells improved). Tuning inside
folds is the correct thing to do and it buys nothing, which is what one expects when the
limit is the measurement rather than the model. The only cells reaching 0.75 use the two
circularly-selected edges, and section 3.7 already prices those at 0.585 done honestly.

**Combining modalities does not help.** Connectivity, structural volumes, head motion and
behaviour were crossed in every combination. The best brain-only cell is connectivity alone
(0.666, XGBoost, n = 32); adding structural volumes never improves on connectivity. Under
leave-one-out over 32 cells the maximum is 0.840 for connectivity + motion + behaviour with
LDA, **p = 0.065** against its own null — and that cell combines a confound known to differ
by group (section 3.5) with the trained skill that defines the groups. It is reported for
completeness and is not evidence of anything.

**Structure-function fusion is worse than function alone.** Across seven cells, functional
connectivity restricted to the three networks gives 0.812, structural volumes give 0.28-0.44
depending on atlas, and fusing them gives 0.41-0.56 — a **loss of 0.25** relative to
functional alone. Concatenating a null modality with an informative one costs accuracy at
this sample size rather than adding to it. (This seven-cell family is the first row of
section 3.4.5's table, and its p = 0.0010 is subject to that section's objection.)

**Every participant and every modality in one model, and it is the worst of the three.** The
fusion above deletes anyone missing a modality, which at complete-case costs 47 of 79
participants and all of gong2015, since availability here is nested inside dataset: structural
volume exists for 79, and connectivity and behaviour for the same 32. We therefore also fitted the
version that keeps everyone (`scripts/run_stacked_all.py`). Each modality gets a base learner
trained on whoever has it, out-of-fold base scores become the features of a meta-learner, and
the meta-learner receives an availability indicator per modality, so a gong2015 participant
contributes a structural score and two missing flags while a Julia2018 participant may
contribute three scores. Site is harmonised inside training folds. Six cells — three
volumetric atlases by two declared families, brain-only and behaviour-including — with a
family-wise null over each family separately at B = 200.

**Brain-only reaches 0.528 against a critical value of 0.595 (p = 0.28); adding behaviour gives
0.549 against 0.589 (p = 0.17).** Wilson intervals at n = 79 are [0.42, 0.63] and [0.44, 0.65].
The control that licenses reading these at all behaves: a classifier given *only* the
availability mask reaches 0.448, so the stack cannot launder site into an accuracy, which is
what the balanced designs at each site (24/23 and 16/16) predict.

**The instructive part is that pooling made the headline worse, and the decomposition says
why.** Connectivity alone on 32 participants reaches 0.660 (section 3.4.1); everything on 79
reaches 0.528. Out-of-fold, the three paths differ enormously in what they carry: the
structural path correlates with group at r = +0.05 over all 79, the functional path at r = +0.50
over 32 and the behavioural at r = +0.26 over 32, and the meta-learner weights them accordingly
(+0.04, +0.51, +0.55). The paths are not redundant — pairwise correlations among their scores
are all below 0.19 in magnitude — so this is not a case of modalities duplicating each other.
The 47 participants that pooling adds carry **only the path that does not discriminate**, so
they raise n on an uninformative modality and dilute an informative one estimated from 32
people. More participants in the wrong modality is worse than fewer in the right one, and no
choice of combiner repairs it. This is the same lesson as the seven-cell fusion above,
measured now at 79 with missingness handled rather than deleted.

Two limits apply to this analysis specifically and are not repaired by it. The label mapping
treats gong2015's non-experts — amateur action gamers — as non-players, so a null is ambiguous
between no effect and classes wrongly merged (section 3.6, and the contrast noted in section
4.3). And the behaviour-including family is near-circular, since participants were
recruited for the performance difference those features encode, which is why it is corrected as
its own family and why the brain-only row is the one to read. We report no regional or
edge-level attribution from this model. Section 3.4.3's permuted-label null is passed only by a
grid spanning several model families, and this is a single stacked model at 0.528, so a feature
ranking taken from inside it has neither the accuracy nor the classifier diversity that would
make it testable. What the model supports interpreting is one level up — which paths carry signal, and
whether they are redundant — and that is what is reported above
(`scripts/run_stacked_weights.py`).

**Treating the run as a bag of windows does not help.** Multiple-instance learning over five
60-timepoint windows per participant — mean-pooling, max-pooling and per-window voting —
gives 0.750, 0.719 and 0.781 against a whole-run baseline of 0.812: a **gain of exactly
zero**, best case. If a group difference were concentrated in some intervals and absent in
others, max-pooling should find it. It does not.

**Normative modelling with a diffusion model makes things worse.** A denoising diffusion model
trained on 1,100 windows of MPI-LEMON, used to score each participant's connectivity as a
deviation from the normative distribution, gives 0.345-0.435 across three classifiers against
0.607-0.666 for the raw features and 0.533-0.636 for a Gaussian normative control. The
deviation scores are worse than the features they are computed from. A normative model needs
its reference population to be exchangeable with the target; LEMON differs from Julia2018 in
scanner, protocol and population, and the deviation score absorbs that difference along with
whatever else is there.

**The cost of doing it badly, measured on this dataset.** Six analysis variants, from the
defensible to the indefensible, over the same 12,880 candidate region-level features:

| rung | accuracy | what changed |
|---|---|---|
| 0 | **0.541** | selection and scaling fitted inside every training fold |
| 1 | 0.560 | maximum over 8 classifiers, no null reported |
| 2 | 0.763 | top 10 edges chosen on all 32 participants, then cross-validated |
| 3 | 0.825 | best fold partition of 20 seeds (median of those 20: 0.781) |
| 4 | 0.813 | selection and scaling both fitted on everything, leave-one-out |
| 5 | 1.000 | accuracy on the training set |

**Rungs 1-5 are demonstrations of invalid methodology and must never be cited as results.**
The honest number is 0.541. The gap to the top is **+0.459**, and every step in between is a
practice that appears in published work. Rung 3 is the one worth dwelling on: changing
nothing but the random seed of the fold partition, and keeping the best of twenty, moves
accuracy from 0.781 to 0.825 — a "result" produced by a decision most papers do not report
making.

## 4. Discussion


### 4.1 Breadth of transfer and the coordination of large-scale networks


Our second conclusion concerns which connectivity patterns do the classifying, and how they
bear on current theories of cognitive training and transfer through action video games. The
primary differences between AVGPs and NVGPs involve higher-level crosstalk between multiple
distributed brain networks rather than change confined to any one of them. Under the assumption
that the observed group differences are caused by action video game play, this contradicts
views attributing the effects solely to specific brain networks and hence to specific cognitive
processes. It instead offers an account of why playing action video games might produce broad
cognitive enhancements: what changes is the coordination among large-scale networks, which is
common to many tasks, rather than the efficiency of a single specialised system.

This finding also highlights the relevance of resting-state data in examining the impact of
action video game play, and possibly other media consumption, on brain functioning.
Resting-state data can be collected repeatedly, for example to assess dose-response curves
(Chopin et al. 2019). This contrasts with traditional behavioural measures, where repeated
exposure to the same test can confound the benefits of cognitive training with learning
effects on a specific cognitive test (Green et al. 2014; Green et al. 2019). Resting-state
data might also help assess differences between AVGPs and NVGPs untainted by prior task
experience or expectation effects, that is, action video game players performing better
because they believe they should perform better, addressing a controversy in the literature
(Parong et al. 2022; Tiraboschi et al. 2019). We note in section 4.6 that this advantage is
weakened when recruitment is overt.

### 4.2 Control and sensorimotor networks


Our findings highlight the importance of the interplay between the cingulo-opercular
network and sensorimotor network on the one hand, and the fronto-parietal network and SMN
on the other, in accurately classifying participants based on their resting-state data.

The FPN and CON have repeatedly been highlighted as key actors in adaptive cognitive
control and are hypothesised to work in tandem to provide both cognitive stability and
flexibility. While CON is associated with task-set maintenance that promotes stable control
across trials, FPN has been associated with moment-to-moment control demanded for flexible,
stimulus-driven control, for example processing task cues and feedback (Dosenbach et al.
2008). Interestingly, the direct relationship between CON and FPN was not discriminative of
AVGPs and NVGPs, but rather the relationships between these networks and the SMN.

The SMN has comparatively received much less attention in both the cognitive control and
the action video gaming literature. As noted in section 1.5, the SMN as defined in the
Dosenbach2010 parcellation comprises brain regions across the frontal, parietal and
temporal cortices and includes most notably the precentral gyrus, the supplementary motor
areas, the pre-SMA and the insula. The SMA and pre-SMA are part of the supplementary motor
complex, which comprises neurophysiologically heterogeneous structures involved in multiple
higher-level cognitive functions, including learning and executing action sequences,
switching between tasks, and the initiation and inhibition of actions; the supplementary
motor complex is crucial to link cognition to action (Nachev et al. 2008; Gordon et al.
2023). Neural recordings in the human brain have shown, for example, that the pre-SMA
encodes an integrated utility signal reflecting the value and uncertainty of possible
choices, which may drive value-based decision making and thus optimal goal-directed
behaviour (Aquino et al. 2023). The SMN also involves the insula, associated with a variety
of cognitive functions including socio-emotional processing, decision-making under
uncertainty, attention, and salience processing (Uddin et al. 2017).

Previous research suggests that the integration of information from multiple brain networks
may be crucial to successfully exert cognitive control over behaviour, as it allows for the
flexible use of various sources of information to make predictions, the prioritisation of
certain sources of information, and the allocation of computational resources to achieve
specific behavioural goals (Cohen 2021; Eisenreich et al. 2017; Jiang et al. 2018). This
dynamic interplay between control networks anchored in prefrontal cortex and the
sensorimotor complex may be a key factor contributing to the heightened cognitive
performance exhibited by action video game players. Action video games require players to
make quick and accurate decisions in a constantly changing environment; consequently AVGPs
may need to rely more on their CON to maintain a stable and coherent representation of the
environment. Action video games also require players to switch between different tasks and
strategies quickly and efficiently; as a result AVGPs may need to rely more on their FPN to
flexibly adapt to changing demands.

### 4.3 Relation to prior work


Gong et al. (2015) reported increased functional connectivity between attentional and
sensorimotor networks in action game experts, with the effect concentrated in the left
insula, a region contained within the Dosenbach SMN. Our attribution analysis points at the
same system, and that convergence is worth noting — an independent report using a different
design and a different method is what a specification-robust effect would predict.

It should not be leaned on, and one clarification has to come before the two reasons, because
the obvious misreading of section 3.6 would be unfair to that paper. **Our structural arm is
not a replication attempt of Gong et al.'s finding and cannot fail as one.** Their claim was
about resting-state functional connectivity; what they shared with us, and what we reanalysed,
is the T1-weighted anatomical series acquired in the same session. Regional grey-matter volume
and interregional functional coupling are different measurements, and a null for the former
says nothing about the latter. Anyone reading section 3.6 as "Gong et al. did not replicate"
has read it wrong: their functional result was never put to the test here, because the data
that would test it were not ours to analyse. What section 3.6 establishes is the narrower and
previously unknown fact that action-game expertise in that sample is not decodable from
structural volume — a question Gong et al. neither asked nor answered.

With that said, two things do bound how much the convergence can carry. Whatever structural
correlate might have underwritten a functional difference in that sample is not detectable at
this n (section 3.6). And when the three networks our functional attribution identifies are
carried over to that structural data — and vice versa — neither direction transfers above what
a random choice of three networks achieves, though a cross-modality transfer of a network
identity is a demanding test that a true effect could also fail. So the convergence is between
our functional attribution and their published functional claim, in two datasets that share neither
participants nor modality, and it has failed the one direct test we were able to construct.

Our contribution is distinct in three respects. Gong et al. contrasted experts with
amateurs, whereas we contrast players with non-players. They used seed-based insula
connectivity, whereas we use whole-network multivariate modelling, which allows the
relative importance of within- and between-network terms to be compared and therefore lets
the three families of accounts in section 2.1 be adjudicated rather than merely
illustrated. And their analysis followed a single path, whereas we report 49 of them and the
family-wise correction that searching them requires — which is also why our decoding claim is
weaker than theirs rather than stronger. The gong2015 arm (section 3.6) makes the comparison
direct: their own structural data, reanalysed under this discipline, is likewise null for
decoding.

Relatedly, Cahill, Jordan and Dhamala (2024) report enhanced dorsal visual stream connectivity
in AVGPs that tracks response time. That study achieves the brain-behaviour link we do not,
which is a limitation of ours rather than a disagreement — and it does so with task fMRI, where
the behavioural measure is collected in the scanner alongside the imaging, an advantage no
resting-state design has.

### 4.4 What the robustness analysis adds


We asked whether resting-state functional connectivity can classify participants as action
video game players or non-video game players, and whether the answer depends on how the
analysis is done. It depends substantially. Accuracy varied from 0.385 to 0.660 across the 49
analysis paths we ran, a spread of 0.275 in a measure whose whole range of interest is
0.5 to 1.0, so a reader given only the best path would form a materially different impression
from one given the distribution.

The explanation varied less. Attribution converged on between-network coupling of the control
networks with the sensorimotor network, at +0.366 normalised agreement across all 31
network-scale specifications, and the figure barely moves when the set is restricted by
accuracy: raw agreement runs 0.634, 0.646, 0.663, 0.657 as the cutoff rises from none to 0.60,
against a random-ranking floor of 0.423. Reporting both matters, because agreement
computed only among well-performing specifications would be inflated by the selection itself,
and here the fact that it is *not* inflated is the evidence.

The word "converged" should be read narrowly. What is reproducible is which couplings sit at
the top: CON-SMN in the top five of every specification, FPN-SMN in 81%. The full ordering of
the 21 edges is not reproducible — Kendall's tau over all ranks is indistinguishable from
chance — so this is a claim about the leading features, not about a ranking.

A multiverse also finds choices its designers did not think to declare. The largest single
effect we measured is not any of the four axes in the registry: it is whether region time
courses are averaged into networks *before* connectivity is estimated or *after*, worth 0.13 to
0.19 in accuracy on identical data, folds and statistic (section 3.2). That is larger than the
effect of the classifier family, of external pretraining, or of the choice among six
connectivity statistics. It is also a choice that papers in this literature routinely make
without stating, which is exactly the kind of degree of freedom a multiverse exists to expose —
and a reminder that the grid one declares is itself a choice with a garden of forking paths
behind it.

**What the multiverse does not fix.** That last point deserves more than a clause, because the
apparatus in this paper can be read as claiming more rigour than it delivers. A family-wise
correction disciplines a search *once the family is fixed*, and fixing the family is an
unpriced degree of freedom that sits upstream of every number reported here. Section 3.4.5
measures it on a single held-fixed result: four defensible families around the same 0.812
accuracy yield p = 0.0010, 0.0066, 0.0399 and 0.2376. Nothing in the fitting, the
cross-validation or the permutation distinguishes them; only a declaration does, and the
declaration came after the result was known. We would be entitled to report the first row if we
had written it down first, and we did not.

Two consequences follow for how this paper should be read, and for how we think multiverse
analyses in general should be. First, the honest use of the correction here is as a *floor* —
it tells the reader what a search of this width manufactures from nothing (about 0.63 at the
median over 49 cells, about 0.80 over 833) and therefore how much of any reported accuracy is
already accounted for. Read that way it is robust to the family being drawn slightly wrong.
Second, the sequence that would make the narrow family legitimate — declare, then look — is
available and cheap, and it is precisely the sequence this study did not follow. The
recommendation we draw is not "run a multiverse instead of preregistering". It is that a
multiverse without a prior declaration prices the search honestly but cannot restore the
licence that declaration would have conferred, and the two are complements rather than
substitutes.

Our first conclusion is therefore methodological, and it is not the one we expected to draw. In
this dataset explanation is descriptively more stable than prediction, and an earlier version
of this paper stopped there and proposed that the stable attribution might be the appropriate
target of inference in small samples. The permuted-label null in section 3.4.3 does not support
that proposal without qualification. Grids refitted on shuffled labels reach 0.562 against the
observed 0.657 when the grid spans three classifier families (p = 0.035), but 0.601 against
0.630 within the linear family alone (p = 0.29). Stability is a real property of this
multiverse, and it is a property of the *diversity* of the multiverse rather than of its size.

**The qualification is more useful than the unqualified claim would have been.** Stability
across a specification space is routinely offered — by us, in an earlier draft — as though it
were corroboration, on the intuition that a finding many pipelines agree about is less likely
to be an artefact of any one of them. That intuition is right only under a condition nobody
states. Pipelines fitted to the same participants are not independent witnesses: they share the
sample, and at n = 32 they share its noise. What determines whether their agreement is
informative is whether they *also* share an inductive bias. Six variants of a linear decoder
agree on shuffled labels at 0.601, because they overfit the same noise the same way. A grid
crossing a linear model, a tree ensemble and a neural network agrees on shuffled labels at
0.562, because they do not. The observed agreement barely moves between the two, so the entire
inferential content sits in the null.

We would make three recommendations to anyone running a comparable analysis, and we followed
none of them until late. Report attribution agreement against a permuted-label null, not a
random-ranking baseline; the cost is small, because features are label-independent and cache
across permutations, and the random-ranking floor here understates the real null by about 0.15.
Vary the model family even when it does not move accuracy — it contributes 3-5% of accuracy
variance in this grid (section 3.4.2), which made it look like the axis to economise on, and it
turns out to be the axis the attribution claim rests on. And treat the *identity* of the
leading feature and the *magnitude* of the agreement as separate claims with separate nulls,
since a grid can converge on a different feature in every permutation while agreeing strongly
within each one.

Two further caveats bound the descriptive stability result itself, both set out in section 4.6:
an attribution ranking carries less information than a continuous accuracy, and the result is
specific to a network-scale feature space.

### 4.5 What these nulls do not license

Because this paper reports many null results in a literature that has been contested, it is
worth being explicit about the inferences they do and do not support. The risk is not that a
careful reader will overread them; it is that "resting-state connectivity fails to classify
action gamers" travels further than the qualifications attached to it.

**Nothing here bears on whether action video game play improves cognition.** That question is
behavioural, it has been addressed by meta-analysis (Bediou et al. 2018; Bediou et al. 2023),
and this study contributes no evidence to either side of it. Our own participants show the
expected behavioural difference — AVGPs more accurate (0.840 against 0.750) and lower in
inverse efficiency (1.111 against 1.409), d = +0.74 and −0.76, nominally significant and short
of a corrected threshold — in the same session as the scans analysed here (section 3.3). Whatever else these data are, they are not a sample in which the
behavioural phenomenon is absent.

**A null is not evidence of absence when the design is provably unable to produce a
positive.** Section 3.7 is the load-bearing part of this paper for that reason. At the measured
edge reliability the attainable accuracy caps near 0.72, and the family-wise threshold this
grid must clear is 0.739. A real effect of ordinary size, measured perfectly by these
instruments, would have failed here. Reporting "no specification cleared the null" without that
number attached would invite exactly the inference the number forbids.

**The two datasets were shared, not scraped, and both behaved as their collectors described.**
The resting-state data were acquired alongside Föcker et al. (2018), whose behavioural and
task-fMRI findings we do not dispute and partly reproduce; the structural data were shared by
Gong et al. (2015), whose functional finding we did not test (section 4.3). Neither group
collected their data to answer the question we put to it, and neither should be read as having
failed a test they did not sit. Reanalysis under a discipline that did not exist when data were
collected is a service to a literature only if it is scored that way.

**What the study does support is narrower and, we think, more useful.** The Dosenbach
region-to-network mapping carries information a dimensionality-matched shuffle does not
(section 3.4.2); the leading attribution is reproducible across analysis paths and depends on
the group labels (section 3.4.3); a specific restriction — control-to-sensorimotor coupling at
network scale — is worth taking to a preregistered test in new data (section 3.4.5); and the
ceiling computation that explains the rest is available to any study of this kind before it
collects a participant (section 3.7).

### 4.6 Limitations


**Sample size and inference.** Averaging over 25 repetitions of 4-fold cross-validation gives
a well-determined point estimate, and the spread across the 100 folds shows the estimate is not
an artefact of any one partition. That is the most that resampling can achieve. It does not
reduce uncertainty about how the result would generalise to new participants, which is governed
by n: at 32 participants the best accuracy of 0.660 carries a 95% Wilson interval of
[0.49, 0.80], and the point estimate should not be read as precise. That interval spans
everything from chance to a strong effect.

The relevant benchmark is Marek et al. (2022), who show that brain-wide association studies
require thousands of participants for reproducibility, and note that the median neuroimaging
study is around 25 people. The usual reply on behalf of a design like ours is that the critique
targets mass-univariate correlations between one connectivity feature and one continuous
individual-difference measure across a heterogeneous population, where effect sizes are tiny
and sampling variability dominates, and that it does not directly govern a hypothesis-driven
contrast between two well-characterised groups on a pre-specified feature set. **That reply
is not fully available to us**, because our feature set was not pre-specified: the networks
that carry the result were identified in these same data (sections 1.5 and 3.4.5). The part
of it that does apply — that a two-group contrast is not a brain-wide association study — is
sound as far as it goes, and it is reinforced by evidence that multivariate predictive models
can be replicable at substantially smaller samples than univariate ones, in some cases from
n = 75 (Spisak, Bingel and Wager 2023, replying to Marek et al. 2022), and that contrasting
groups selected to be extreme on the covariate of interest raises standardised effect sizes
relative to population sampling (Kang et al. 2024) — the AVGP versus NVGP contrast being an
extreme-group design by construction.

**We nonetheless think the honest reading here runs the other way.** Our sample sits below even
the reduced thresholds those papers describe, and the family-wise permutation test returned no
specification clearing its null. Rather than deploy the above as a reason the null should be
discounted, we take it as what the power literature predicts: at this n, a real but modest
group difference in intrinsic connectivity is not reliably decodable, and a multiverse honestly
corrected for the search will say so. The value of the design is then not that it rescued a
positive result but that it prevented a false one — and that it identified a quantity, the
attribution ranking, which *is* stable at this sample size.

Reliability is a further real constraint rather than a rhetorical one. Mean edge-level
test-retest reliability of functional connectivity is poor (mean ICC = 0.29, 95% CI
[0.23, 0.36]; Noble, Scheinost and Constable 2019), though substantially higher for strong
within-network cortical edges and improving with more within-subject data; reliability rises
markedly with scan length, from five minutes toward thirteen or more (Birn et al. 2013); and
with sufficient within-individual data, individual network structure is highly stable (Gordon
et al. 2017). Our interpretation is pitched at the network rather than the single-edge level
for this reason, which is also why attribution is reported as selection frequency over networks
rather than as a ranking of individual edges. Our acquisition is 125 volumes at a repetition
time of 3 s, that is **6 min 15 s**, of which 124 volumes survive the initial-volume drop.
Against Birn et al.'s curve that is barely above the five-minute point where reliability is
poorest and less than half the thirteen minutes at which it begins to plateau, so the
achievable ceiling on both reliability and classification here is set by the acquisition before
any analytic choice is made. These results require replication in a larger independent sample,
and we would not expect the decoding result to become significant without one.

**Head motion.** Action gamers moved more in the scanner than non-gamers, motion alone decodes
group at 0.61, and motion correlates with sensorimotor coupling at r ≈ 0.5 (section 3.5). This
is the strongest alternative explanation the paper faces and we cannot fully exclude it. Three
things bound it. The confound regression applied to every time course includes the six
realignment parameters and white-matter and CSF signals. The CON-SMN group association survives
partialling motion out in four of six connectivity statistics, though not at a
multiple-comparison-corrected threshold. And the paper makes no decoding claim for motion to
explain away. What motion does bear on is the substantive reading of section 3.1: a motion
artefact would be expected to produce a stable attribution just as a real effect would, so the
stability of the CON-SMN attribution is not evidence against the motion account. Section
3.4.3's permuted-label null now says something stronger and less specific to motion — a stable
attribution is what correlated overfitting produces in general, whatever its source.

What would settle it is data this study does not have: motion-matched subsamples, which at
n = 32 leave too few participants, or an acquisition with the multi-echo or high-sampling-rate
denoising that removes participant-level motion effects rather than merely regressing
timecourses. We flag it as a limitation rather than resolving it.

**Two independent checks converge on the same distinction.** The motion analysis and the
early-versus-late carryover analysis below were run for different reasons and share no
machinery, and both separate the two leading features the same way: CON-SMN survives, FPN-SMN
does not. That agreement is worth more than either check alone, and it is why the paper's
substantive claim is stated in terms of control-to-sensorimotor coupling rather than in terms
of the pair of edges the rankings put on top.

**Cross-sectional design.** AVGPs and NVGPs are self-selected groups. Nothing here
establishes that action video game play caused the observed differences; pre-existing
differences that predispose individuals toward action gaming are equally consistent with
our data. Where we describe effects as consequences of gaming we do so under an explicit
causal assumption the design cannot test.

What can be said is that the gaming-to-network direction is at least plausible, because
training does reshape resting connectivity in this general way. Videogame practice
longitudinally altered resting parieto-frontal control-network connectivity (Martínez et al.
2013), and perceptual learning changed resting connectivity in proportion to the amount learned
(Lewis et al. 2009). Neither study is about action games specifically, so neither settles the
direction for our contrast; they establish that the effect a causal account requires is the
kind of effect training is known to produce. Longitudinal first-person-shooter training studies
([[VERIFY: Brilhault et al. 2020]]) are the design that would settle it directly.

**Rest followed task.** The resting-state scan was acquired after participants completed an
attention-demanding, auditory-cued visual discrimination task. Resting connectivity is
known to be altered by immediately preceding tasks (Hasson et al. 2009; Lor et al. 2022;
Tailby et al. 2015). Because the preceding task was itself sensorimotor and attentionally
demanding, differential carryover between groups is a live alternative explanation for the
CON-SMN and FPN-SMN effects, and one we cannot exclude. We tested whether the effect
attenuates between the first and second half of the run (62 timepoints each), on the
reasoning that carryover decays while a trait difference should not. The answer splits by
coupling: the CON-SMN group difference is stable across halves (early d = +0.94, late
d = +1.04), which is the pattern a trait difference predicts, whereas the FPN-SMN difference
attenuates by roughly 60% (early d = +0.86, late d = +0.34), which is what carryover
predicts. The primary coupling therefore behaves like a trait on this test; the FPN-SMN
result should be read with the carryover alternative live. If the carryover account is
correct for FPN-SMN, the present study would suggest that cognitive tasks have varying
short-term impacts on resting-state connectivity depending on participants' video gaming
habits, which is itself interesting.

**Acquisition.** Scans comprised 125 volumes at 4 mm isotropic resolution with an
eight-channel coil. This is short by current standards for reliable connectivity estimation
(Birn et al. 2013) and bounds the achievable ceiling on classification performance. With
TR = 3 s the Nyquist frequency is 0.167 Hz, so cardiac and respiratory signals alias into
band.

**Interpretation of attribution.** Connectivity features among spatially overlapping
networks are correlated, so every attribution method used here distributes credit among
collinear features, and rankings should be read as identifying a diagnostic subsystem rather
than an ordered list. Techniques such as multi-pass permutation importance, which keep
previously assessed features permuted while evaluating new ones, would partially address this
(McGovern et al. 2019); we did not use permutation importance in any channel, for a reason
worth stating, since it is the default choice in much of this literature. Permutation
importance is *defined* as the drop in accuracy when a feature is scrambled, so where accuracy
is indistinguishable from chance the quantity it measures is a difference between two noise
estimates. SHAP and gradient-times-input read the fitted model's response surface and the Haufe
pattern is built from the weights and the data covariance; none of the three inherits the
accuracy's resolution problem, which is what makes the attribution analysis of section 3.4.3
possible at all when the decoding analysis of section 3.4.1 is null. The group comparisons in section 3.1 are descriptive, not
independent tests.

**Attribution carries less information than accuracy, and this bounds the central claim.**
An attribution ranking over a few dozen features is a coarser object than a continuous
accuracy, so part of its apparent stability may reflect lower resolution rather than greater
robustness. Normalising both against their own empirical nulls (section 3.4.3) addresses the
differing floors of the two metrics but not the differing resolution. Two consequences
follow. The claim should be read as being about the reproducibility of a qualitative
conclusion, namely which subsystem is diagnostic, rather than as a claim that attribution
estimates are more precise than accuracy estimates. And the finding is specific to a
network-scale feature space; at region scale, where the ranking is over thousands of edges
and the random-ranking floor is correspondingly low, we would not expect the same result, and
section 3.4.3 reports the region-scale specifications separately for this reason.

**The attribution channels are not fully independent.** SHAP and gradient-times-input are both
computed from the same fitted model's response surface and correlate by construction, so their
agreement is a consistency check rather than corroboration. Only the Haufe pattern is derived by
a structurally different route, and only for the linear specifications. The paper should not be
read as reporting three independent witnesses. (An earlier draft named permutation importance as
the second channel; it was never used — see the note above on why it would be the wrong choice
here.) Section 3.4.3 shows the channels are nonetheless not interchangeable in a way that
matters more than their independence: the permuted-label null is cleared by a grid spanning all
three and not by the Haufe channel alone.

**Analytic and sampling variation are confounded in the accuracy spread.** Every
specification is fitted to the same 32 participants, so accuracies across specifications are
correlated, and the observed spread mixes genuine sensitivity to analytic choices with
sampling noise that all specifications share. The variance decomposition in section 3.4.2 is
therefore descriptive and its components should not be given inferential interpretation.
Separating the two sources would require either an independent sample or the LEMON
subsampling calibration extended to the full grid.

**Parcellations are not ground truth.** Different parcellation methods yield different
mappings, so a given region may be assigned to different networks, or to none, depending on
the parcellation used, and some networks exist in certain parcellations but not others.
There are inconsistencies across methods, hyperparameter choices remain subjective, and
most standard methods assign a region to only one network, implying networks do not
overlap. As methods and datasets improve, the validity of the identified networks will
improve with them. This is a reason to traverse the space rather than to pick from it.

**No behavioural link.** We did not establish that the connectivity features distinguishing
AVGPs from NVGPs also predict behavioural performance. Features useful for group
discrimination need not be those explaining individual differences in ability (van Balkom
et al. 2022).

**Overt recruitment.** Participants were recruited on the basis of their gaming habits, so
the expectation-effect advantage claimed for resting state in section 4.1 is weaker than it
would be under blind recruitment (Boot et al. 2011).

---

## 5. Conclusion


This study set out to classify participants as action video game players or non-players from
resting-state functional connectivity, and to use the multiverse to decide whether the answer
was robust. **No specification cleared the family-wise null**, and an earlier version of this
paragraph asserted the opposite. The correction matters less than the reason for it, which
section 3.7 supplies and which we would put ahead of the null itself: at the reliability these
data have, the attainable accuracy caps near 0.72, while a search of this width requires 0.739.
The design could not have returned a positive result for an effect of ordinary size. The
finding is therefore not that the effect is absent but that this instrument, at this n and this
scan length, cannot see it — and that this was computable in advance.

Across analysis paths that disagree about accuracy, the models converge on the same features:
coupling between the cingulo-opercular and fronto-parietal control networks and the
sensorimotor network — the precentral gyrus, supplementary motor area, pre-SMA, mid-insula and
superior parietal cortex. That convergence is reproducible at the top of the ranking, is not a
by-product of selecting well-performing specifications, survives a dimensionality-matched
control, and is not reproducible from the label-free covariance of the feature space.

**That convergence survives a permuted-label refit of the whole grid, and the condition under
which it survives is itself the result.** Across three classifier families the grid agrees at
0.657 where shuffled labels give 0.562 (p = 0.035); within the linear family alone it agrees at
0.630 where shuffled labels give 0.601 (p = 0.29). Models sharing an inductive bias overfit the
same 32 participants the same way and agree whether or not the labels mean anything, so a
multiverse establishes something about the data only to the extent that its specifications
disagree about how to fit them. The identity of the converged-on feature points the same way:
permuted grids select CON-SMN 6% of the time (p = 0.063).

Read at that strength the result is *consistent* with the idea that broad transfer would
express itself as reconfigured coordination between control and action systems rather than as
change within either. It is weaker support than the raw agreement figure suggested, since that
figure was calibrated against a null 0.15 too low, and the idea remains one this work sharpens
rather than establishes.

**The methodological finding is the more portable one, and we think it is the more
important.** Three results compose into a single argument. Attribution stability and
predictive accuracy are separate properties, and a study can have one without the other.
The worth of any accuracy depends on a declared family that no fitting procedure sees:
the same model here is worth p = 0.001 or p = 0.24 depending only on how wide one draws
it, and no width is legitimate once the features were chosen by looking. And the measurement
imposes a ceiling near 0.72 which, at this sample size, sits *below* the threshold a
corrected search already requires — so a real effect of ordinary size could not have been
detected here even if it were present. That last point does not depend on anything about
video games. It is a statement about small-sample resting-state connectivity studies
generally, and it is checkable in advance, before a sample is collected rather than after it
disappoints.

The generalisability of our conclusions is constrained by the dataset used: a limited number
of participants and a relatively short resting-state recording. Section 3.7 makes that
constraint quantitative rather than conventional — longer scans raise reliability, and
reliability is what sets the ceiling, so the remedy is more data per participant at least as
much as more participants. The methods developed here can readily be adapted to accommodate
additional, longer resting-state recordings (Poldrack et al. 2017), and we leave it to future
work to replicate and extend these findings. The specific hypothesis worth taking forward is
stated in section 3.4.5, and taking it forward means declaring it in advance.
Because the specification registry is published as a machine-readable artefact, such
replication can target the same 495 analysis paths rather than a single one.

---

## 6. Methods


### 6.1 Data

For the purpose of this study we used an unpublished resting-state fMRI dataset previously
collected alongside a task-fMRI study (Föcker et al. 2018). 32 participants (16 AVGPs and
16 NVGPs) underwent a resting-state fMRI session after completing several cognitive tasks
in the scanner. The primary objective of the original study was to investigate attentional
control in action video gamers during task performance.

All 32 participants were male (16/16 per group), so sex is constant by design rather than a
potential confound; the cost is that nothing here generalises to women, and the near-total
absence of female samples is a standing weakness of the action-gaming literature. The groups
were age-matched: AVGPs 20.8 +/- 2.1 years (range 19-27), NVGPs 21.1 +/- 2.0 (19-25)
(`outputs/descriptives.json`).

All participants were volunteers and gave informed consent. In accordance with the
Declaration of Helsinki, the Research Subject Review Board of the University of Rochester
approved the study.

> **Action required.** The original draft never states the operational
> definition of AVGP and NVGP, which is the independent variable of the entire paper:
> hours per week, over what period, which genres counted, and the screening instrument.
> A demographic table is also required, including **sex per group**; AVGP samples are
> typically male-skewed and if the groups differ, sex is a plausible confound for network
> connectivity that must be covaried. Age, handedness, and matching should be reported.

Resting-state fMRI data were acquired using a Siemens TrioTim 3T scanner with an
eight-channel head coil, 4 mm isotropic resolution, TE/TR = 30/3000 ms, flip angle 90
degrees. Anatomical T1w images were defaced prior to preprocessing to ensure participants'
privacy.

> **Action required.** The original draft states 125 volumes and a duration of
> 7 minutes 30 seconds. At TR = 3 s, 125 volumes is 375 s (6 min 15 s); 450 s would be 150
> volumes. If dummy scans were discarded, say so and give the number.

### 6.2 Preprocessing

Given that even minor changes to fMRI preprocessing can affect analysis results (Lindquist
2019), we used a reproducible pipeline for the entire preprocessing stage. We used MRIQC
for data quality checks and fMRIPrep (v20.2 LTS) for preprocessing, without modifying
default parameters. The only exception was that we skipped skull stripping because the
anatomical T1w images were already defaced for privacy reasons. For each participant,
preprocessing resulted in 125 images of 4 mm isotropic voxels in MNI152NLin2009cAsym space.
Additional preprocessing decisions followed the "simple" denoising strategy implemented in
Nilearn (v0.10.1), comprising high-pass filtering, 6-degree head motion correction, basic
CSF component removal, demeaning, no global signal removal, no scrubbing, no CompCor, and
no ICA-AROMA.

> **Action required.** (1) The original states high-pass filtering at 0.1 Hz. As written
> this removes essentially the entire resting-state band of interest (0.01 to 0.1 Hz).
> Nilearn's "simple" strategy uses the fMRIPrep cosine regressors, corresponding to a 128 s
> cutoff, about 0.008 Hz. Verify against the code and correct. (2) Defacing removes the
> face, not the skull, so the stated justification for skipping skull stripping does not
> follow as written. Specify the exact fMRIPrep flag used and what input it expects. Also
> note MRIQC v21.0.0rc2 is a release candidate; prefer a stable release or justify.

### 6.3 Parcellation and aggregation

Using parcellation atlases to group data from voxels into meaningful brain regions reduces
complexity and noise and allows semantics to be attributed to brain activations, in that
brain regions and networks are more interpretable than isolated voxels (Varoquaux and Craddock
2013). Voxels were aggregated into regions and then into large-scale networks according to
the selected parcellation, using maximum likelihood estimation of the time series of the
defined spatial maps.

By assigning semantics to networks rather than regions the model becomes simpler, albeit
less comprehensive, making it possible to interpret results in terms of general cognitive
functions commonly related to cognitive control rather than sparse activation in regions
(Dadi et al. 2019; Varoquaux and Craddock 2013). A smaller number of features is also
important for the computational and statistical tractability of the model, for example 6
networks instead of 160 regions in the Dosenbach2010 parcellation.

Where the original study used one atlas, the present study traverses ten (section 2.2), and
the aggregation scale is itself a specification dimension with three levels. The
random-network level shuffles the region-to-network mapping while preserving the number of
networks and regions per network, and serves as the dimensionality-matched negative control.

### 6.4 Connectivity statistics

Using the aggregated time series we calculated connectivity matrices measuring the
relationship between node pairs. Since connectivity matrices are symmetric we flattened the
upper triangular part, including diagonal values, and used the resulting vector as input to
the classification pipeline.

Seven statistics were computed: covariance, Pearson correlation, partial correlation, tangent
projection of covariance, precision (sparse inverse covariance), Chatterjee's coefficient, and
transfer entropy. The first five are the Nilearn kinds used in the original analysis; all
seven are implemented in `acnets.connectome.ExtraConnectivityMeasure`. Section 2.2 gives the
rationale for this particular set and for not widening it further.

Five of the seven statistics yield symmetric matrices, and for these we flatten the upper
triangular part including the diagonal, using the resulting vector as classifier input. Two do
not: transfer entropy is directed by construction, and Chatterjee's coefficient is asymmetric
in its arguments. For those the full matrix must be retained, since discarding the lower
triangle would remove exactly the directional information that motivated including them.
Feature counts therefore differ by statistic, n squared against n(n+1)/2 for n nodes, and are
recorded per specification (`Specification.n_features`) so that dimensionality is never
confounded with statistic in section 3.4.2.

> **Action required.** The current code does not implement this. `ExtraConnectivityMeasure`
> correctly refuses `vectorize=True` for transfer entropy, but the separate vectorisation step
> in `MultiScaleClassifier` (line 167) applies `np.triu_indices` unconditionally, which
> silently symmetrises both asymmetric statistics. Branch on
> `Specification.is_asymmetric` before running the grid. Until that is fixed, the claim that
> the signature survives a directed statistic is not supported by the pipeline, because no
> directed statistic is actually reaching the classifier.

### 6.5 Classification

We fitted binary classifiers receiving participants' vectorised connectivity matrices and
predicting their label. Three families are used. A support vector machine, frequently used as
a baseline in this literature (Dadi et al. 2019), specifically an L1-regularised linear SVM
after standardisation. XGBoost, as a non-linear tree ensemble. And a small multilayer
perceptron implemented directly in PyTorch
(`acnets.deep.TorchMLPClassifier`), which exposes the same fit/predict surface as the other
two so that a neural model can enter the same grid, the same permutation null and the same
attribution analysis rather than being reported separately.

**Hyperparameters, and a correction.** An earlier version of this section stated that
hyperparameters were tuned on training data. They were not: the three families ran at fixed
values chosen once. The claim has been made true rather than deleted.
`acnets.multiverse.classifiers` defines a search space per family and, when tuning is
enabled, wraps the estimator in a randomised search over an inner cross-validation *of the
training fold*, so the outer test fold is never seen by the search. That nesting is what
makes a tuned accuracy an honest estimate, and it is asserted by a test that tunes against
permuted labels and requires the result to stay at chance — a leak would show up there as
above-chance accuracy on data containing no signal.

Tuning turned out not to matter. Over 37 (feature set × family × tuning) cells the mean
change in accuracy was **+0.000**, the median +0.000, the range −0.050 to +0.075, and tuning
helped in 13 of 37 cells — a coin flip. We report the untuned grid as the primary analysis,
because it is what the family-wise null was computed over, and the tuned comparison as the
evidence that the choice does not matter.

The three families are included because they fail differently. The linear SVM can only
express additive effects and is the one for which a forward pattern is defined; the tree
ensemble can express interactions but is unstable at this sample size; the network is the
only family that can be pretrained on external data (section 6.8). If a feature is prominent
under all three, that is a fact about the data rather than about an inductive bias.

> **Change from the original.** The original pipeline performed model-based feature selection
> using L1-SVM coefficients and then classified with an L1-SVM. L1 already performs
> selection, so the extra stage adds variance without adding information, and with 21
> features it is very likely harmful; it has been removed. Standardisation is now fixed a
> priori rather than tuned, since tuning it enlarges the search space for no scientific
> reason.

Following recommended best practice for machine learning in neuroscience (Poldrack et al.
2020) we computed accuracy on held-out participants. The cross-validated pipeline was
trained on 75% of the data (24 participants, 12 AVGPs) and evaluated on the remaining 25%
(8 participants, 4 AVGPs).

Relying on a point estimate from a single split is insufficient at this sample size. With 8
participants in a test fold, accuracy can only take the nine values 0, 1/8, ... , 1, and which
of them is observed depends heavily on how the 32 participants happened to be partitioned.
Empirical work has documented how strongly such non-deterministic factors affect model
performance on small neuroimaging datasets (Chyzhyk et al. 2022; Thomas et al. 2022; Varoquaux
et al. 2018). Averaging over many partitions is the appropriate remedy, and it is the central
element of our approach to the small sample: we repeated 4-fold cross-validation 25 times and
take the mean over all 100 resulting fold accuracies as the estimate. This yields a
substantially better-determined estimate than any single split or any single statistical test on
one partition could provide, and it is why the design is built around repetition rather than
around a single fit.

The partition is generated by `acnets.multiverse.fixed_splits`, which draws the folds from the
participant indices alone and never from the labels. Label-stratified folds would be the
conventional choice, but they are not usable here: the permutation null (section 6.6) refits
the whole grid under permuted labels, and stratifying would give the observed statistic
balanced folds that the null does not get, biasing the comparison in favour of the observed
result. Every specification, and every permutation of every specification, therefore sees the
identical list of 100 train/test index pairs.

### 6.6 Two uncertainties, reported separately

Repetition solves one problem and leaves another untouched, and because the two are easily
conflated we report them as distinct quantities (`acnets.multiverse.accuracy_report`).

**Split stability** asks how much the estimate depends on the arbitrary partition. This is what
repetition addresses. The spread of accuracy across the 400 folds, reported as a standard
deviation and a 2.5 to 97.5 percentile range, documents that the point estimate is not an
artefact of one lucky split. It shrinks as repetitions are added, which is the intended effect.

**Sampling uncertainty** asks how much the estimate would move in a new sample of participants.
Repetition cannot reduce it, because every repetition reuses the same 32 people. It is governed
by n, and we quantify it with a Wilson interval on the number of participants. At n = 32 and an
accuracy of 0.72 that interval is approximately [0.55, 0.85], several times wider than the
split-stability range.

Both are reported for every specification. Quoting the first as though it were the second is the
error we are guarding against: it is narrow, it looks like a confidence interval, and it shrinks
with computational effort, which makes it tempting. It describes the stability of an estimator,
not the precision of a conclusion about action video game players in general. Conversely,
reporting only the Wilson interval would discard evidence the design deliberately generated, and
would leave unexplained why 100 repetitions were run at all.

Family-wise significance over the specification space uses a max-statistic permutation test
rather than a parametric test on the fold accuracies, for the same reason: the 400 fold
accuracies are not independent observations, so a t-test over them has no valid degrees of
freedom. Comparisons between two specifications evaluated on overlapping resampled splits use
the corrected resampled t-test of Nadeau and Bengio (2003), which inflates the variance to
account for exactly that dependence.

Every fitted step, including confound regression, parcellation, connectivity estimation,
scaling, feature selection, and hyperparameter search, is fitted inside training folds only.
This is asserted by a leakage test in the test suite that records the participant indices
seen by every transformer's `fit` and fails if any test-fold participant appears.

We report accuracy, AUC, sensitivity, specificity, and the confusion matrix. At n = 32
accuracy alone is a coarse and unstable summary.

The max-statistic test itself works as follows (`acnets.multiverse.max_stat_permutation`): for
each of B permutations of the group labels the entire grid is refitted and the maximum accuracy
across specifications is recorded, and the observed maximum is compared against that
distribution. Labels are permuted outside the whole pipeline, including feature selection and
hyperparameter search, since permuting inside leaks the true labels through the selection step
and produces an optimistic null. The p-value uses (1 + #{null >= observed}) / (1 + B), so the
smallest attainable value is 1/(B+1); we use B = 1000, giving a floor of 0.001.

The observed run and every permutation use the identical cross-validation budget of 4 folds
by 25 repeats. This is not a detail. A mean over 100 folds is less variable than a mean over
20, so comparing a smooth observed maximum against a noisier null maximum would inflate
significance by an amount that has nothing to do with the data. Refitting the full grid a
thousand times at that budget is affordable only because feature extraction never sees the
labels: features are content-addressed by specification-without-classifier, fold indices and a
hash of the input timeseries, and cached on disk (`acnets.multiverse.runner`), so each
permutation pays for classifier refits alone and reuses connectivity matrices computed once.

### 6.7 Explanation

Each specification yields one attribution vector over its feature vocabulary, computed by the
method its model class admits (`acnets.multiverse.runner._fold_attribution`), averaged over
folds. For the linear SVM this is the Haufe forward pattern
(`acnets.explain.haufe_transform`), reconstructed from the weights and the training-fold
covariance, because decoder weights conflate signal with noise suppression and cannot be read
as claims about the brain (Haufe et al. 2014). For XGBoost, which has no weights to transform,
it is the mean absolute SHAP value. For the multilayer perceptron it is the mean absolute
gradient of the logit with respect to the input, times the input, evaluated on the training
fold — the closed-form analogue of mean absolute SHAP for a differentiable model.

These are not interchangeable estimators of one quantity, and we do not treat them as three
independent witnesses. All three answer "how strongly does this feature carry the
discriminative signal" and are compared only by rank, never by magnitude, which is what makes
the comparison meaningful across such different units. But SHAP and gradient-times-input both
read the fitted model's own response surface, so their correlation is partly structural. The
Haufe pattern is the one channel built from the data covariance rather than from perturbing
the model, so agreement involving it carries more weight; it is available only for the linear
specifications, which is the reason attribution agreement is reported across the full
classifier axis rather than within one model family.

Cross-specification agreement is quantified by rank-biased overlap
(`acnets.multiverse.agreement_matrix`), which is top-weighted so that specifications
agreeing on the leading features count as agreeing, with Kendall tau reported as an
unweighted check. Because all specifications rank a shared feature vocabulary, raw agreement
has a floor that depends on the number of features, and we therefore report agreement
normalised against a random-ranking baseline computed over the same vocabulary
(`acnets.multiverse.random_agreement_baseline`), with 0 meaning chance and 1 perfect
agreement. Feature prominence is summarised by selection frequency in the top k across
specifications rather than by a single ranking.

**Two further nulls, because a random-ranking baseline is not the null the claim needs.**
The baseline above asks whether agreement exceeds what a shared vocabulary manufactures. It
cannot ask the question a sceptical reader asks first, which is whether the agreement has
anything to do with the group labels. That question has teeth here precisely because the
classifiers do not decode above a corrected null: the Haufe pattern is A = Σw / (w'Σw), so as
the weight vector w degenerates towards noise the pattern degenerates towards Σw, whose
ranking is governed by the label-free feature covariance that every specification shares. SHAP
and gradient-times-input have their own versions of the same degeneracy. If that is what
produces the agreement, the reported stability is a property of the explanation machinery
rather than a finding about these participants.

The first null removes the classifier and keeps the geometry
(`scripts/run_attribution_geometry.py`, reported in section 3.4.3): each of the 21 network edges is ranked by four
label-free criteria — its variance across participants, its mean absolute value, the magnitude
of its loading on the leading eigenvector of the feature covariance, and the mean magnitude of
Σw over 200 random draws of w, which is literally the pattern a chance-level linear decoder
would yield — and each resulting ranking is compared against the observed multiverse consensus
by the same rank-biased overlap. No labels enter at any point.

The second null keeps the classifiers and destroys the labels
(`scripts/run_attribution_null.py`). The whole network-scale grid is refitted on permuted
labels, with the same label-independent folds, the same cached features and the same
attribution methods, and the full pairwise agreement matrix is recomputed for each
permutation. Its mean is the statistic; its distribution over permutations is the null against
which the observed agreement is tested, with the upper-tail p bounded below by 1/(B+1) as
elsewhere. This is the stricter of the two, because a classifier fitted to permuted labels can
still latch onto sample-specific structure that a random weight vector cannot, and it is the
null the stability claim actually has to clear.

### 6.8 Pretraining on an external resting-state dataset

Thirty-two participants is too few to learn a representation from, and the multiverse makes
that worse rather than better by multiplying the ways a small sample can be overfitted. We
therefore added an axis, `pretraining`, taking values `none` and `lemon`, so that "was
anything learned from outside this sample" becomes a dimension of the multiverse whose
contribution to the variance can be read off alongside atlas, statistic and classifier
instead of being an uncontrolled implementation choice.

**External data.** MPI-LEMON (Babayan et al. 2019) provides 220 usable resting-state runs of
652 volumes, parcellated with the same atlases and the same pipeline as Julia2018. Runs are
cut into non-overlapping 124-timepoint windows to match Julia2018's run length, giving
roughly 1,100 windows (`acnets.multiverse.transfer.windowed`). Each window is z-scored over
time, channel by channel, which is also the mitigation for the scanner and TR mismatch: after
standardisation the connectivity statistics see fluctuation structure rather than units.

**What is transferred, for classical models.** A `pretraining='lemon'` specification differs
from its `none` twin in exactly two fitted objects, both estimated on LEMON and frozen before
Julia2018 is touched (`acnets.multiverse.transfer.fit_lemon_transforms`): the connectivity
measure and the feature scaler. For the tangent statistic the first is substantive, since
tangent-space projection requires a reference covariance and estimating it from 220 external
subjects is both better determined and structurally leakage-proof compared with estimating it
from 24 training participants. For the statistics whose `fit` is stateless the measure is
shared for uniformity and the twins differ only in scaling.

**What is transferred, for the neural model.** A bottleneck autoencoder is trained on LEMON to
reconstruct single timepoints of the network-scale signal
(`acnets.deep.TimepointAE`). Applied per timepoint, it can only reconstruct well by learning
how networks co-fluctuate, which is the same quantity every connectivity statistic estimates
but expressed in a learned basis. A subject's features are then the means and the upper
triangle of the covariance of their latent series. A sequence autoencoder was trained first and
rejected: held-out reconstruction plateaued at 1.004 relative to predicting the window mean, so
the temporal unrolling learned nothing, and the negative result is recorded here because it
constrains what resting windows can be said to carry.

**Why this cannot leak.** LEMON and Julia2018 share no participants; MPI-LEMON contains no
action-video-game variable and no Julia2018 group labels; and the fitting functions take
timeseries only and have no parameter through which a label could be supplied. The last point
is enforced by signature rather than by discipline, and asserted by a test that fits the
transforms, permutes the Julia2018 labels, refits, and requires bit-identical output
(`transfer_test.py`). Fine-tuning of the neural model happens inside training folds only, as
for every other fitted step.

### 6.9 Negative controls

Three analyses. The first two, on head motion, are on the same 32 participants and neither is
cross-validated for a performance claim, because a confound check is a question about
participant-level association rather than about generalisation.

The first asks whether motion carries the group label at all
(`scripts/run_motion_control.py`). Each participant is summarised by the mean and standard
deviation of the six realignment parameters, framewise displacement, DVARS, standardised DVARS
and RMS displacement — 20 features — and the same three classifiers are fitted on the same
label-independent folds used everywhere else. The features are deliberately chosen to be what a
motion-based account of the group difference would predict to be discriminative. Univariate
Welch tests per channel are reported alongside, with Holm correction over the 20 channels and
the uncorrected count given as well, since "nothing survives correction at n = 32" and "nothing
differs" are different claims.

The second asks whether motion explains the specific features the attribution analysis
identifies (`scripts/run_motion_coupling.py`). For each connectivity statistic and each
network edge we report the Pearson correlation between the edge and framewise-displacement
standard deviation, the Welch group comparison, and the group–edge partial correlation with
motion regressed out of both. Framewise-displacement SD is used as the single motion summary
because it showed the largest group difference in the first analysis; the choice is made once
and named in the script rather than selected per edge.

The third is the spatially-displaced control (`scripts/run_cerebellar_control.py`). Precision
connectivity is estimated **within** each retained set of regions rather than sliced out of
the whole-brain matrix, because an entry of the full inverse covariance conditions on all 160
regions while a restricted analysis would condition only on its own subset; the restricted
estimate is the one that matches what such an analysis would actually have. Every set is
k = 18 regions — the cerebellum's own count — so that the comparison varies location with
dimensionality held fixed. The two reference distributions are 200 draws each of 18 regions
from FPN+CON+SMN and from all non-cerebellar regions, classified by the same leave-one-out
shrinkage discriminant. Reading the cerebellum against chance rather than against these draws
would answer a question nobody asked; the informative comparison is against an arbitrary
territory of the same size.

### 6.10 Structural volumes, and the external validation between the two samples

Both datasets go through **one** pipeline, which is what makes the cross-dataset comparison
possible; an earlier version processed them differently and the two feature spaces did not
correspond (see the note below).

Per participant (`scripts/segment_t1.py`, run under an isolated antspyx environment because
antspyx pins an older scipy): N4 bias correction; brain extraction by warping a conservative
MNI brain mask through an initial affine registration and intersecting it with an intensity
mask; a second, **brain-to-brain** affine registration with that mask applied to both sides;
Atropos three-class tissue segmentation in native space; atlas labels warped into the
participant's space with the inverse transform. Grey-matter volume per region is the sum of
grey-matter probability inside the region mask times the voxel volume, divided by the
participant's total grey matter. Warping the atlas into native space rather than the brain
into template space means volumes are measured where the anatomy actually is, with no
modulation bookkeeping. Registration is affine only, a declared limitation.

Each participant's brain-mask volume is recorded and flagged outside 900-2,000 cm3. All 79
participants across both datasets passed. Julia2018's T1 arrives skull-stripped from fmriprep
and gong2015's does not, so their absolute mask volumes differ by roughly 40%; normalising by
total grey matter removes that scale difference, and the regional profiles then agree at
r = +0.87 to +0.94.

Three **volumetric** atlases are used -- AAL, Harvard-Oxford and Schaefer-200, staged from
nilearn by `scripts/prepare_atlases.py`. No FreeSurfer is required anywhere.

> **Two defects in the earlier pipeline, and why the atlas choice is not cosmetic.**
> `ants.get_mask` is an intensity heuristic, not a brain extractor, and left skull in
> gong2015's whole-head images: a 2,127 cm3 "brain" and an inflated 785 cm3 "grey matter",
> against 1,755 and 656 after the fix. Separately, the Dosenbach 2010 atlas is 160 spheres of
> about 10 mm; under affine registration between different acquisitions a sphere can miss its
> target or sit partly outside the brain, so its volume is close to arbitrary. Under that
> atlas the two datasets' regional profiles correlated at **-0.406**; under Harvard-Oxford,
> **+0.936**, with the ordering monotone in parcel size. Sphere atlases are appropriate for
> connectivity, where a time course is averaged over whatever the sphere contains, and
> unsuitable for regional volumetry across datasets. `prepare_atlases.py` reports median
> parcel volume and flags anything sphere-like so the mistake cannot recur silently.

**Analyses.** Within each dataset, leave-one-out accuracy. Between them, external validation:
a model fitted on one dataset predicts every participant of the other, with standardisation
fitted on the training dataset only -- refitting it on the target would let the target inform
its own prediction, and z-scoring the target within itself would quietly harmonise the sites.
A permutation null shuffles labels within the training dataset and refits
(`scripts/run_cross_dataset.py`). Pooled analyses additionally remove per-site means and
scales inside each training fold (`scripts/run_pooled_structural.py`); site is predictable at
0.96-0.99, so this is not optional. Nothing about the target dataset informs any fit, so the
transfer results need no multiverse correction.

### 6.11 Reliability, and the analyses of section 3.8

**Split-half reliability.** Each participant's run is cut in half, connectivity is estimated
on each half independently, and the two halves are correlated across participants per edge.
That is the reliability of a half-length run; Spearman-Brown, r_full = 2r / (1 + r), projects
it to full length. Edges that are constant by construction (a fixed diagonal) are skipped
rather than contributing a spurious zero. The same procedure is applied to MPI-LEMON and to
ABIDE controls, so that the Julia2018 estimate is read against two external corpora rather
than against an assumption.

**Reliability weighting.** Weights are applied **after** standardisation, which is the only
place they can survive: the scaler divides each column by its own standard deviation and so
exactly undoes any diagonal rescaling applied beforehand. The first implementation weighted
the raw features and produced weighted results identical to the unweighted ones to three
decimal places, which is what exposed the error.

**Bias-corrected Mahalanobis distance.** The naive sample D² is inflated by roughly p/n at
these ratios. We use the Lachenbruch (1968) correction, which subtracts that bias and can go
negative — a negative corrected D² is a statement that the observed separation is fully
accounted for by estimating p dimensions from n participants. Bayes accuracy is then Φ(D/2),
with a confidence interval propagated from the interval on D².

**Selection cost.** The circular estimate selects the top-k edges on all participants and
then cross-validates; the nested estimate repeats the selection inside each training fold.
The difference is the amount by which selecting on all the data inflates the result, and it
is reported rather than avoided because the inflated number is what an uncorrected analysis
would have published.

**Multiple-instance learning.** Windows of 60 timepoints at stride 16 give five windows per
participant. Bags are participants, instances are windows, and the participant label is
applied to the bag only. Mean-pooling and max-pooling aggregate instance scores; voting takes
the modal instance prediction. Windows from one participant never straddle a fold boundary.

**Normative modelling.** A denoising diffusion probabilistic model (cosine schedule,
epsilon-prediction, MLP denoiser with sinusoidal timestep embedding) is trained on 1,100
windows of MPI-LEMON connectivity. Each Julia2018 participant is scored as a deviation from
the normative distribution over 16 draws. A Gaussian normative model over the same reference
serves as the control that separates "normative modelling helps" from "the diffusion model
helps".

**The overfitting ladder.** Every rung uses the same 12,880 candidate region-level features
and the same participants; only the protocol changes. Rung 0 fits selection and scaling
inside each training fold. Rungs 1-5 progressively remove that discipline. The ladder is a
methodological demonstration computed on real data, and the store carries a `WARNING` field
saying so, because a number in a results file is a number someone will eventually quote.

### 6.12 Reproducibility

The specification registry is serialised to JSON and published as a supplementary file, so
the set of analysis paths considered is auditable rather than asserted. Data, model
checkpoints, and outputs are versioned with DVC. Each (specification, run, fold) result is
stored as an immutable record, and every figure is a projection over that store, so
questions of the form "what if you had excluded transfer entropy" are answered by
re-projecting rather than re-running. Code is available at
`https://github.com/morteza/ACNets`; a COBIDAS-MRI checklist is provided as supplementary
material.

---

## Acknowledgements and data availability

Neither dataset analysed here was collected by us, and neither was collected to answer the
question we put to it. Both exist in this paper because the groups who collected them chose to
share them.

The resting-state series were acquired alongside the task-fMRI study of Föcker et al. (2018),
whose design, recruitment and behavioural protocol are theirs; the resting-state run was
secondary to that study's purpose and had not been analysed. The T1-weighted structural series
of section 3.6 were shared by the authors of Gong et al. (2015), with provenance confirmed by
them, and are the only external check anywhere in this work. Data sharing of this kind carries
an asymmetric risk for the sharer — a reanalysis can only ever qualify the original — and the
field depends on people accepting it anyway. We record our thanks, and note that the negative
results in sections 3.4 and 3.6 are properties of sample size, scan length and measurement
reliability (section 3.7), not of how these data were collected. Neither result contradicts a
published claim by either group: section 3.6 tests a modality Gong et al. did not analyse
(section 4.3), and the behavioural difference reported by Föcker et al. is present in these
same participants (section 3.3).

[[TODO: author contributions, funding, competing interests. Confirm the terms under which the
gong2015 T1 series may be redistributed before stating a data-availability position for it;
the Julia2018 resting-state data are unpublished and their release is the original
investigators' decision, not ours.]]

---

## References

Entries below are grouped by **how they were checked**, which is not the same as how likely
they are to be right, and the distinction is the point of the grouping. Group 1 was confirmed
against the sources, with quoted text, by the literature review. Group 2 was written from
domain knowledge to close the gaps Group 1 left: the bibliographic details are given in full
so that checking each is a lookup rather than a search, but none was confirmed against the
source and any of them may carry a wrong volume, page range or DOI. Group 3 could not be
confirmed at all and must be verified or dropped.

Every citation in the body now has an entry. **Nothing in Group 2 or 3 should be submitted
without a spot-check** — an incorrect citation caught by a reviewer costs more than a dropped
one, and it costs most when the rest of the list is accurate enough to be trusted.

### Group 1 — confirmed against the sources

Bavelier, D., and Green, C. S. (2019). Enhancing attentional control: Lessons from action video
games. *Neuron*, 104(1), 147-163. doi:10.1016/j.neuron.2019.09.031

Bavelier, D., Green, C. S., Pouget, A., and Schrater, P. (2012). Brain plasticity through the
life span: Learning to learn and action video games. *Annual Review of Neuroscience*, 35,
391-416. doi:10.1146/annurev-neuro-060909-152832

Bediou, B., Adams, D. M., Mayer, R. E., Tipton, E., Green, C. S., and Bavelier, D. (2018).
Meta-analysis of action video game impact on perceptual, attentional, and cognitive skills.
*Psychological Bulletin*, 144(1), 77-110. doi:10.1037/bul0000130 (Correction: *Psychological
Bulletin*, 144(9), 978-979. doi:10.1037/bul0000168)

Bediou, B., Rodgers, M. A., Tipton, E., Mayer, R. E., Green, C. S., and Bavelier, D. (2023).
Effects of action video game play on cognitive skills: A meta-analysis. *Technology, Mind, and
Behavior*, 4(1). doi:10.1037/tmb0000102

Birn, R. M., Molloy, E. K., Patriat, R., Parker, T., Meier, T. B., Kirk, G. R., et al. (2013).
The effect of scan length on the reliability of resting-state fMRI connectivity estimates.
*NeuroImage*, 83, 550-558. doi:10.1016/j.neuroimage.2013.05.099

Cahill, K., Jordan, T., and Dhamala, M. (2024). Connectivity in the dorsal visual stream is
enhanced in action video game players. *Brain Sciences*, 14(12), 1206.
doi:10.3390/brainsci14121206 — **task fMRI and diffusion, not resting-state**

Cardoso-Leite, P., Ansarinia, M., Schmuck, E., and Bavelier, D. (2021). Training cognition with
video games. In K. Cohen Kadosh (Ed.), *The Oxford Handbook of Developmental Cognitive
Neuroscience*. Oxford University Press.

Cole, M. W., Bassett, D. S., Power, J. D., Braver, T. S., and Petersen, S. E. (2014). Intrinsic
and task-evoked network architectures of the human brain. *Neuron*, 83(1), 238-251.
doi:10.1016/j.neuron.2014.05.014

Dale, G., Joessel, A., Bavelier, D., and Green, C. S. (2020). A new look at the cognitive
neuroscience of video game play. *Annals of the New York Academy of Sciences*, 1464(1),
192-203. doi:10.1111/nyas.14295

Dosenbach, N. U. F., Fair, D. A., Cohen, A. L., Schlaggar, B. L., and Petersen, S. E. (2008). A
dual-networks architecture of top-down control. *Trends in Cognitive Sciences*, 12(3), 99-105.
doi:10.1016/j.tics.2008.01.001

Dosenbach, N. U. F., Fair, D. A., Miezin, F. M., Cohen, A. L., Wenger, K. K., Dosenbach, R. A.
T., et al. (2007). Distinct brain networks for adaptive and stable task control in humans.
*PNAS*, 104(26), 11073-11078. doi:10.1073/pnas.0704320104

Dosenbach, N. U. F., Nardos, B., Cohen, A. L., Fair, D. A., Power, J. D., Church, J. A., et al.
(2010). Prediction of individual brain maturity using fMRI. *Science*, 329(5997), 1358-1361.
doi:10.1126/science.1194144

Finn, E. S., Shen, X., Scheinost, D., Rosenberg, M. D., Huang, J., Chun, M. M., Papademetris,
X., and Constable, R. T. (2015). Functional connectome fingerprinting: identifying individuals
using patterns of brain connectivity. *Nature Neuroscience*, 18(11), 1664-1671.
doi:10.1038/nn.4135

Gong, D., He, H., Liu, D., Ma, W., Dong, L., Luo, C., and Yao, D. (2015). Enhanced functional
connectivity and increased gray matter volume of insula related to action video game playing.
*Scientific Reports*, 5, 9763. doi:10.1038/srep09763

Gong, D., Ma, W., Gong, J., He, H., Dong, L., Zhang, D., et al. (2017). Action video game
experience related to altered large-scale white matter networks. *Neural Plasticity*, 2017,
7543686. doi:10.1155/2017/7543686

Gong, D., Ma, W., Liu, T., Yan, Y., and Yao, D. (2019). Electronic-sports experience related to
functional enhancement in central executive and default mode areas. *Neural Plasticity*, 2019,
1940123. doi:10.1155/2019/1940123

Gordon, E. M., Laumann, T. O., Gilmore, A. W., Newbold, D. J., Greene, D. J., Berg, J. J., et
al. (2017). Precision functional mapping of individual human brains. *Neuron*, 95(4),
791-807.e7. doi:10.1016/j.neuron.2017.07.011

Gratton, C., Laumann, T. O., Nielsen, A. N., Greene, D. J., Gordon, E. M., Gilmore, A. W., et
al. (2018). Functional brain networks are dominated by stable group and individual factors, not
cognitive or daily variation. *Neuron*, 98(2), 439-452.e5. doi:10.1016/j.neuron.2018.03.035

Green, C. S., and Bavelier, D. (2003). Action video game modifies visual selective attention.
*Nature*, 423(6939), 534-537. doi:10.1038/nature01647

Green, C. S., Li, R., and Bavelier, D. (2010). Perceptual learning during action video game
playing. *Topics in Cognitive Science*, 2(2), 202-216. doi:10.1111/j.1756-8765.2009.01054.x

Guerra-Carrillo, B., Mackey, A. P., and Bunge, S. A. (2014). Resting-state fMRI: A window into
human brain plasticity. *The Neuroscientist*, 20(5), 522-533. doi:10.1177/1073858414524442

Harmelech, T., and Malach, R. (2013). Neurocognitive biases and the patterns of spontaneous
correlations in the human cortex. *Trends in Cognitive Sciences*, 17(12), 606-615.
doi:10.1016/j.tics.2013.09.014

Kühn, S., and Gallinat, J. (2014). Amount of lifetime video gaming is positively associated
with entorhinal, hippocampal and occipital volume. *Molecular Psychiatry*, 19(7), 842-847.
doi:10.1038/mp.2013.100 — first published online 2013; **use one convention throughout**

Kühn, S., Gleich, T., Lorenz, R. C., Lindenberger, U., and Gallinat, J. (2014). Playing Super
Mario induces structural brain plasticity: gray matter changes resulting from training with a
commercial video game. *Molecular Psychiatry*, 19(2), 265-271. doi:10.1038/mp.2013.120

Lewis, C. M., Baldassarre, A., Committeri, G., Romani, G. L., and Corbetta, M. (2009). Learning
sculpts the spontaneous activity of the resting human brain. *PNAS*, 106(41), 17558-17563.
doi:10.1073/pnas.0902455106

Marek, S., Tervo-Clemmens, B., Calabro, F. J., Montez, D. F., Kay, B. P., Hatoum, A. S., et al.
(2022). Reproducible brain-wide association studies require thousands of individuals. *Nature*,
603(7902), 654-660. doi:10.1038/s41586-022-04492-9 (Correction: *Nature*, 605, E11.
doi:10.1038/s41586-022-04692-3)

Martínez, K., Solana, A. B., Burgaleta, M., Hernández-Tamames, J. A., Álvarez-Linera, J.,
Román, F. J., et al. (2013). Changes in resting-state functionally connected parietofrontal
networks after videogame practice. *Human Brain Mapping*, 34(12), 3143-3157.
doi:10.1002/hbm.22129

Noble, S., Scheinost, D., and Constable, R. T. (2019). A decade of test-retest reliability of
functional connectivity: A systematic review and meta-analysis. *NeuroImage*, 203, 116157.
doi:10.1016/j.neuroimage.2019.116157

Papadaki, E., Koustakas, T., Werner, A., Lindenberger, U., Kühn, S., and Wenger, E. (2023).
Resting-state functional connectivity in an auditory network differs between aspiring
professional and amateur musicians and correlates with performance. *Brain Structure and
Function*, 228(9), 2147-2163. doi:10.1007/s00429-023-02711-1

Reineberg, A. E., Andrews-Hanna, J. R., Depue, B. E., Friedman, N. P., and Banich, M. T.
(2015). Resting-state networks predict individual differences in common and specific aspects of
executive function. *NeuroImage*, 104, 69-78. doi:10.1016/j.neuroimage.2014.09.045

Rosenberg, M. D., Finn, E. S., Scheinost, D., Papademetris, X., Shen, X., Constable, R. T., and
Chun, M. M. (2016). A neuromarker of sustained attention from whole-brain functional
connectivity. *Nature Neuroscience*, 19(1), 165-171. doi:10.1038/nn.4179

Sadaghiani, S., and D'Esposito, M. (2015). Functional characterization of the cingulo-opercular
network in the maintenance of tonic alertness. *Cerebral Cortex*, 25(9), 2763-2773.
doi:10.1093/cercor/bhu072

Sala, G., and Gobet, F. (2017). Does far transfer exist? Negative evidence from chess, music,
and working memory training. *Current Directions in Psychological Science*, 26(6), 515-520.
doi:10.1177/0963721417712760

Sala, G., Tatlidil, K. S., and Gobet, F. (2018). Video game training does not enhance cognitive
ability: A comprehensive meta-analytic investigation. *Psychological Bulletin*, 144(2),
111-139. doi:10.1037/bul0000139

Spadone, S., Della Penna, S., Sestieri, C., Betti, V., Tosoni, A., Perrucci, M. G., Romani, G.
L., and Corbetta, M. (2015). Dynamic reorganization of human resting-state networks during
visuospatial attention. *PNAS*, 112(26), 8112-8117. doi:10.1073/pnas.1415439112

Tavor, I., Parker Jones, O., Mars, R. B., Smith, S. M., Behrens, T. E., and Jbabdi, S. (2016).
Task-free MRI predicts individual differences in brain activity during task performance.
*Science*, 352(6282), 216-220. doi:10.1126/science.aad8127

West, G. L., Konishi, K., Diarra, M., Benady-Chorney, J., Drisdelle, B. L., Dahmani, L.,
Sodums, D. J., Lepore, F., Jolicoeur, P., and Bohbot, V. D. (2018). Impact of video games on
plasticity of the hippocampus. *Molecular Psychiatry*, 23(7), 1566-1574. doi:10.1038/mp.2017.155

### Group 2 — completed from domain knowledge; details to spot-check

These fill the gaps the literature review did not reach. They are standard methodological and
background citations whose details are given in full below; each needs a DOI lookup, not a
literature search.

Combrisson, E., and Jerbi, K. (2015). Exceeding chance level by chance: The caveat of
theoretical chance levels in brain signal classification and statistical assessment of decoding
accuracy. *Journal of Neuroscience Methods*, 250, 126-136. doi:10.1016/j.jneumeth.2015.01.010

Corbetta, M., and Shulman, G. L. (2002). Control of goal-directed and stimulus-driven attention
in the brain. *Nature Reviews Neuroscience*, 3(3), 201-215. doi:10.1038/nrn755

Dadi, K., Rahim, M., Abraham, A., Chyzhyk, D., Milham, M., Thirion, B., and Varoquaux, G.
(2019). Benchmarking functional connectome-based predictive models for resting-state fMRI.
*NeuroImage*, 192, 115-134. doi:10.1016/j.neuroimage.2019.02.062

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., and Bießmann, F.
(2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
*NeuroImage*, 87, 96-110. doi:10.1016/j.neuroimage.2013.10.067

Hutzler, F. (2014). Reverse inference is not a fallacy per se: Cognitive processes can be
inferred from functional imaging data. *NeuroImage*, 84, 1061-1069.
doi:10.1016/j.neuroimage.2012.12.075

Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*,
52(3), 239-281. doi:10.1023/A:1024068626366

Nahum, M., and Bavelier, D. (2020). Video games as rich environments to foster brain
plasticity. *Handbook of Clinical Neurology*, 168, 117-136.
doi:10.1016/B978-0-444-63934-9.00010-X

Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? *Trends in
Cognitive Sciences*, 10(2), 59-63. doi:10.1016/j.tics.2005.12.004

Spisák, T., Bingel, U., and Wager, T. D. (2023). Multivariate BWAS can be replicable with
moderate sample sizes. *Nature*, 615(7951), E4-E7. doi:10.1038/s41586-023-05745-x

Sporns, O. (2013a). Network attributes for segregation and integration in the human brain.
*Current Opinion in Neurobiology*, 23(2), 162-171. doi:10.1016/j.conb.2012.11.015

Sporns, O. (2013b). The human connectome: Origins and challenges. *NeuroImage*, 80, 53-61.
doi:10.1016/j.neuroimage.2013.03.023 — **the a/b assignment is ours, not the source's**;
section 1.3 cites both together, so confirm which supports which claim, or merge them

Varoquaux, G., and Craddock, R. C. (2013). Learning and comparing functional connectomes
across subjects. *NeuroImage*, 80, 405-415. doi:10.1016/j.neuroimage.2013.04.007
Anticevic, A., Repovs, G., Shulman, G. L., and Barch, D. M. (2010). When less is more: TPJ and
default network deactivation during encoding predicts working memory performance.
*NeuroImage*, 49(3), 2638-2648. doi:10.1016/j.neuroimage.2009.11.008

Babayan, A., Erbey, M., Kumral, D., Reinelt, J. D., Reiter, A. M. F., Röbbig, J., et al.
(2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology
in young and old adults. *Scientific Data*, 6, 180308. doi:10.1038/sdata.2018.308

Boot, W. R., Blakely, D. P., and Simons, D. J. (2011). Do action video games improve
perception and cognition? *Frontiers in Psychology*, 2, 226. doi:10.3389/fpsyg.2011.00226

Corbetta, M., Patel, G., and Shulman, G. L. (2008). The reorienting system of the human brain:
from environment to theory of mind. *Neuron*, 58(3), 306-324. doi:10.1016/j.neuron.2008.04.017

Hubert-Wallander, B., Green, C. S., and Bavelier, D. (2011). Stretching the limits of visual
attention: the case of action video games. *WIREs Cognitive Science*, 2(2), 222-230.
doi:10.1002/wcs.116

Koziol, L. F., Budding, D., Andreasen, N., D'Arrigo, S., Bulgheroni, S., Imamizu, H., et al.
(2014). Consensus paper: the cerebellum's role in movement and cognition. *Cerebellum*, 13(1),
151-177. doi:10.1007/s12311-013-0511-x

Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., and Petersen, S. E. (2012).
Spurious but systematic correlations in functional connectivity MRI networks arise from
subject motion. *NeuroImage*, 59(3), 2142-2154. doi:10.1016/j.neuroimage.2011.10.018

Satterthwaite, T. D., Wolf, D. H., Loughead, J., Ruparel, K., Elliott, M. A., Hakonarson, H.,
et al. (2012). Impact of in-scanner head motion on multiple measures of functional
connectivity. *NeuroImage*, 60(1), 623-632. doi:10.1016/j.neuroimage.2011.12.063

Uddin, L. Q., Nomi, J. S., Hébert-Seropian, B., Ghaziri, J., and Boucher, O. (2017). Structure
and function of the human insula. *Journal of Clinical Neurophysiology*, 34(4), 300-306.
doi:10.1097/WNP.0000000000000377

### Group 3 — could not be confirmed; verify or drop before submission

Marked `[[VERIFY: ...]]` at their first use in the text. The literature review could not
confirm full bibliographic details for these within its search budget. They are plausible and
widely cited, but an incorrect citation caught by a reviewer costs more than a dropped one.

| Citation | Status | Priority |
|---|---|---|
| Benady-Chorney et al. 2020 | title and author list confirmed via ResearchGate; journal, volume and DOI not verified (likely *Brain and Behavior*) | **highest** — most directly relevant to the RSFC claim |
| Föcker et al. 2018; Föcker et al. 2019 | not verified | high — coauthor's own work, should be quick |
| Bavelier et al. 2012 (task fMRI, reduced frontoparietal recruitment) | the 2012 *Annual Review* is verified above; the specific task-fMRI claim attributed to it is not | high |
| Brilhault et al. 2020 (FPS longitudinal) | not verified | high — carries the causal-direction argument in §4.5 |
| West et al. 2015 | not verified (distinct from the verified West et al. 2018) | medium |
| Gleich et al. 2017; Watanabe et al. 2015 | not verified | medium |
| Menon and D'Esposito 2022 | if the intended item is *Neuropsychopharmacology*, 47(1), 90-103, doi:10.1038/s41386-021-01152-w, confirm before citing | medium |
| Gordon et al. 2023 / 2024 "action-mode network" | only a lab preprint verified; confirm the peer-reviewed venue | medium — a strong fit for the §2 pivot if it has appeared |
| Lindquist 2019 (§6.2, "minor preprocessing changes affect results") | year and author uncertain; the claim is standard and well supported — Botvinik-Nezer et al. 2020 (*Nature*, 582, 84-88, doi:10.1038/s41586-020-2314-3) makes it directly and is already cited elsewhere in the draft | **highest** — consider substituting Botvinik-Nezer and dropping this |
| Thomson and Poldrack 2022 (§1.3, networks over modules) | not confirmed; Poldrack has several candidates in this vein | high |
| Liu et al. 2025 (§2.3, benchmark of 239 pairwise interaction measures on HCP) | not confirmed; the claim is specific and load-bearing for the statistic axis | **highest** — it justifies why covariance/precision/distance were chosen |
| Aquino et al. 2023; Cohen 2021; Eisenreich et al. 2017; Jiang et al. 2018; Uddin-adjacent insula claims in §4.2 | not confirmed | high |
| Bryce et al. 2021; Chopin et al. 2019; Thomas et al. 2022; McGovern et al. 2019 | not confirmed | high |
| Cole et al. 2013 (§2.2, global connectivity and behavioural performance) | distinct from the verified Cole et al. 2014; not confirmed | medium |
| Dugué et al. 2018; Lynch et al. 2014 | not confirmed | medium |
| Parong et al. 2022; Tiraboschi et al. 2019 | not confirmed | medium |
| Poldrack et al. 2017 (§4.5, longer recordings); Poldrack et al. 2020 (§6.5, ML best practice) | not confirmed; both are plausibly the precision-fMRI and the *Nature Reviews Neuroscience* ML-best-practice papers respectively | medium |
| Tseng et al. 2007 | not confirmed | medium |
| Badre 2021; Poldrack 2015; Seguin et al. 2020; Nachev et al. 2008; Gozli et al. 2014; Kraus et al. 2021; Varoquaux 2020; Cohen et al. 2008; Anguera et al. 2013; West et al. 2020; Nahum and Bavelier 2020; Mishra et al. 2011; Richlan et al. 2018; Cui et al. 2021; Hasson et al. 2009; Lor et al. 2022; Tailby et al. 2015; Spisak et al. 2023; Kang et al. 2024; Kriegeskorte et al. 2009; Combrisson and Jerbi 2015; Varoquaux 2018; Chyzhyk et al. 2022; Nadeau and Bengio 2003; Webber et al. 2010; Haufe et al. 2014 | carried over from the original draft; outside the literature review's scope | audit before submission |


---

## Appendix B: Figures


| # | Content | Source |
|---|---|---|
| 1 | Analysis pipeline schematic. Preprocessing, the four specification axes, cross-validation, attribution. Must be redrawn to match section 6.5 and to answer Cardoso-Leite's comment about independent runs. | redraw of the original Figure 1 |
| 2 | **Specification curve.** Accuracy for all 49 executed specifications, sorted, with a dot matrix below showing which atlas, statistic, aggregation and classifier each uses; family-wise critical value from the max-statistic null shown as a horizontal band. The paper's signature figure. | `acnets.plotting.specification_curve_figure` |
| 3 | **Agreement versus accuracy cutoff.** Attribution agreement as a function of the accuracy threshold used to select specifications, starting at zero. Shows the agreement is not an artefact of conditioning on well-performing specifications. | `acnets.plotting.agreement_figures.agreement_vs_threshold_figure` |
| 3b | **The two nulls for attribution agreement**, on one axis, and now the paper's second signature figure. The observed agreement (0.630) plotted against the random-ranking floor (0.423), the permuted-label null distribution (mean 0.601, 95th 0.708) and, as a third reference, the spread of label-free rankings of the same features (0.278–0.461). The figure's point is the gap between the two nulls: the claim clears the one that is conventionally reported and not the one that matters. | `scripts/run_attribution_null.py`, `scripts/run_attribution_geometry.py` (**renderer to be written**) |
| 4 | **Stability on a common scale.** Attribution and accuracy agreement, each normalised against its own empirical null, with bootstrap intervals. The direct test of the thesis. | `acnets.plotting.agreement_figures.stability_comparison_figure` |
| 5 | **Selection frequency.** How often each connectivity feature appears in the top five across specifications, replacing the original single feature-importance ranking. | `acnets.plotting.agreement_figures.selection_frequency_figure` |
| 6 | Group comparison for CON-SMN, FPN-SMN and CON-FPN: effect sizes with bootstrap intervals, no p-values. Replaces the original Figure 4. | revision of the original Figure 4 |
| 7 | Feature-block competition: accuracy for H1 regional, H2 within-network and H3 between-network alone and combined, with Wilson intervals; plus the aggregate-then-connect reference that outperforms all of them. | `scripts/run_multiscale.py` (**not** `pipeline.MultiScaleClassifier`, whose H1 block is zero by construction — Appendix E item 13) |
| S1 | Variance decomposition of accuracy, on the two balanced subgrids of section 3.4.2. | `multiverse.variance_decomposition` |
| S2 | Motion negative controls: motion-only decoding accuracy per classifier, and the motion/CON-SMN association per statistic before and after partialling. | `scripts/run_motion_control.py`, `scripts/run_motion_coupling.py` |
| S3 | Spatially-displaced control: within-cerebellum accuracy against the two size-matched draw distributions (18 regions from FPN+CON+SMN, 18 from anywhere non-cerebellar), so that location varies with dimensionality fixed. | `scripts/run_cerebellar_control.py` |
| S4 | Reliability: per-edge split-half reliability in Julia2018, MPI-LEMON and ABIDE controls, with the Spearman-Brown correction and the implied accuracy ceiling drawn against the family-wise critical value. | `scripts/run_reliability_transfer.py`, `scripts/run_abide_reliability.py`, `scripts/estimate_ceiling.py` |
| S5 | The overfitting ladder: accuracy at each of the six rungs, with rung 0 marked as the only valid one. | `scripts/run_overfitting_ladder.py` |
| S6 | Dosenbach2010 networks, retained from the original supplement. | original Figure 6 |
| S7 | Specification curve for the gong2015 structural arm, on the same axes as Figure 2. Nine specifications — three volumetric atlases (AAL, Harvard-Oxford, Schaefer-200) by three classifiers — decoding only. Regenerated from the corrected volumes; the earlier six-cell version, over skull-contaminated volumes and Dosenbach spheres, is withdrawn along with its attribution panel. | `acnets.plotting.specification_curve_figure` |
| S8 | External validation between the two samples: transfer accuracy in both directions and all three atlases against its permutation null, with the feature-profile correlation that licenses the comparison. | `scripts/run_cross_dataset.py` |

`uv run python scripts/make_figures.py` renders six of these — 2, 3, 4, 5, S7 and S8 — by
reading `outputs/` and fitting nothing. Figures 1 and 6 are drawn by hand. The remaining
six, S1 through S6, have their numbers in the store but no renderer yet: the source column
names the script that produced the values, not a function that draws them. Whoever prepares
the submission either adds them to `make_figures.py` or draws them by hand from the JSON;
what must not happen is a figure whose numbers are typed in rather than read.

Note the original Figure 3 is deliberately absent. Its caption listed eight networks with
Gordon labels against six Dosenbach networks in the Methods (Appendix E, item 1); under the
reframe that analysis becomes one labelled specification among 495 rather than a figure in
its own right.

## Appendix E: Open items


1. **Figure 3 atlas contradiction.** The original Figure 3 caption lists eight networks
   with Gordon labels (DAN, VAN, CPN, RSTC) while the Methods describe six Dosenbach
   networks. The *original* submission's supplementary S4 — not the S4 of the table above,
   which is this draft's numbering — confirms a Gordon analysis existed. Under the reframe this is
   no longer a contradiction to hide but a specification to label correctly. Resolve against
   the code before finalising any figure.
2. **Reference list.** ~~The original stops in 2023.~~ **Partly done.** There is now a
   References section, built from `literature_review.md`, split into verified entries (safe as
   written, with DOIs) and a table of entries still to confirm. Three concrete errors were
   fixed: "Reineberg et al. 2018, *Nature Communications*" **does not exist** and is now
   Reineberg et al. 2015, *NeuroImage* (use the 2018 *Frontiers in Psychology* paper instead if
   the intended emphasis is the executive-function relationship specifically); the *Brain
   Sciences* 2024 paper is Cahill, Jordan and Dhamala and is **task fMRI and diffusion, not
   resting-state**, so it no longer sits among the resting-state precedents; and Kühn and
   Gallinat needs one year convention throughout (2013 online, 2014 issue). What remains:
   (a) confirm or drop everything in Group 3, Benady-Chorney et al. 2020 first since
   it is the most directly relevant; ~~(b) complete the list~~ **done** — every body citation
   now has an entry, with Group 2 written from domain knowledge and flagged for a DOI
   spot-check; (c) one targeted search for any 2024-2026 large action-game
   intervention trial or registered report, which would supersede Bediou et al. 2023 as the
   headline behavioural citation and would materially strengthen the causal-direction argument
   in §4.5, currently resting on Martínez et al. 2013 and Lewis et al. 2009.
3. **Figure 1** (pipeline schematic) must be redrawn to match section 6.5 and to answer
   Cardoso-Leite's comment about independent runs.
4. **Author contributions, funding, competing interests, data availability.** None present
   in the original.
5. ~~Decide whether to state explicitly that the study was not preregistered and is
   exploratory.~~ **Done, and it turned out to matter more than a disclosure line.** Section
   1.5 states it plainly. The consequence is worked through in 3.4.5: because FPN, CON and SMN
   were identified in these data rather than declared, none of the four family-wise p-values
   around the 0.812 restricted result is ours to claim, and the result enters the paper as a
   hypothesis for new data. Section 2.4 goes further and withdraws the claim that *anything*
   here was preregistered: the one specification document was written partway through the
   analysis of these same participants, so it constrained nothing and licenses nothing. Two
   framings are withdrawn with it — section 1.5's *a priori* prediction of
   control-to-sensorimotor coupling, written after the result was known, and the description
   of the transfer model as a confirmatory primary.
6. ~~**Run the grid before finalising section 3.4.**~~ **Done.** The 49-specification grid has
   been run with both nulls at B = 1000. The attribution-stability claim of 3.4.3 *holds* under
   rank-biased overlap (excess +0.366, p = 0.019; the dimensionality-matched control correctly
   fails at +0.154, p = 0.135) but *fails* under Kendall's tau (+0.085, p = 0.30), so the claim
   is now stated as being about the top of the ranking only. The decodability claim of 3.4.1
   *fails*: 0 of 49 specifications clear the family-wise null, best 0.660 against a critical
   value of 0.739, p = 0.291. The prediction recorded here — that widening the grid widens the
   null and so would not reverse 3.4.1 — held across all three runs (p = 0.114 → 0.244 →
   0.291). ~~**The paper's centre of gravity has therefore moved from the group difference to
   the attribution/accuracy dissociation**~~ — **superseded by item 6d.**
6d. **Run the null the attribution claim actually needs.** **Done, and it fails.** The
   agreement result above was calibrated against random rankings of the shared vocabulary,
   which no rival hypothesis occupies. Two stricter controls were added
   (`scripts/run_attribution_geometry.py`, `scripts/run_attribution_null.py`). Label-free
   feature geometry does *not* reproduce the multiverse consensus — rankings by variance, mean
   magnitude, leading covariance eigenvector and random-decoder Haufe pattern agree with it at
   RBO 0.278–0.461 against a 0.408 [0.282, 0.611] random-ranking interval, and CON-SMN never
   ranks first under any of them. Refitting the grid on **permuted labels** gives two answers
   depending on the grid: **0.562 against an observed 0.657 across the three classifier
   families (p = 0.035, B = 200)** and **0.601 against 0.630 within the linear family alone
   (p = 0.29, B = 1000)**. The claim therefore stands on the grid the agreement analysis
   actually uses, and the dependence on classifier diversity is the reportable finding — models
   sharing an inductive bias overfit one small sample identically and agree on shuffled labels.
   **The centre of gravity moves to** (a) the reliability ceiling of section 3.7, which sits
   below the correction threshold and explains the decoding null without appeal to biology, and
   (b) the methodological result that attribution agreement needs a permuted-label null *and* a
   classifier-diverse grid. Title, abstract, 3.1, 3.4.3, 3.4.4, 4.4 and 5 rewritten; conditions
   5 and 6 added to 3.4.4. **An intermediate draft, circulated only within this session,
   reported the linear subgrid alone and wrongly concluded the claim had failed.**
6c. **Decide how to respond to the null decoding result.** **Decided, then partly undone.**
   The response was to keep reporting the corrected null and the dissociation (option a),
   *and* to designate a LEMON-pretrained transfer model as a confirmatory primary whose
   specification was fixed before its accuracy was computed (option b), *and* to bring in
   external data, which is option (c) by transfer rather than by recruitment. Options (a) and
   (c) stand and carry the paper. **Option (b) is withdrawn**: the specification document was
   written partway through the analysis, so calling it confirmatory was not defensible, and
   section 2.4 now says so. The model is reported as one more exploratory cell (0.477,
   p = 0.0999) inside the family-wise null. The original three-way trade-off text is
   preserved in `plan.md` §1.
   (Original options, for the record: (a) report the null and lead on the dissociation;
   (b) narrow to a pre-registered primary plus sensitivity analyses; (c) increase n.)
6b. ~~**Run the brain-behaviour analysis**~~ **Done** (`scripts/run_brain_behavior.py`,
   `outputs/brain_behavior.json`). Result: a clean null — no association pooled or within
   group for CON-SMN or FPN-SMN against accuracy, RT or IES; 25 of 32 participants had
   behavioural records (the seven later-recruited replacements do not). §3.3 and the abstract
   are updated. The groups do differ behaviourally (AVGP accuracy 0.840 vs 0.750, IES 1.111 vs
   1.43), which sharpens the dissociation reading rather than undermining it.
7. ~~**Decide the primary specification's fate if it lands at chance.**~~ **Moot.** There is
   no primary. Section 2.4 withdraws the confirmatory designation, so there is no uncorrected
   p-value to honour and no licence to pass to a substitute. Every path, including the one
   that was briefly called primary, is reported inside the family-wise null. The item is kept
   because the question it asks — what would you have done if the designated analysis had
   failed and another had succeeded — is the question that exposed the designation as empty:
   nothing about the answer would have changed.
8. ~~**State the number of permutations actually used.**~~ **Done.** The reported analysis
   uses B = 1000 over the 49-specification grid at the full 4 x 25 cross-validation budget,
   floor 0.001, and B is now stated in sections 3.4.1 and 6.6.
9. ~~**Sex distribution per group**~~ **Done.** `participants.tsv` carries sex and age for
   all 32: **every participant is male** (16/16 per group), so there is no sex confound to
   answer — the objection becomes a generalisability limitation instead, now stated in §6.1.
   Ages matched (20.8 +/- 2.1 vs 21.1 +/- 2.0). See `outputs/descriptives.json`.
10. ~~**Asymmetric statistics are silently symmetrised**~~ (section 6.4). **Fixed.**
    `acnets.multiverse.runner` branches on `Specification.is_asymmetric` and keeps the full
    matrix for directed statistics; `MultiScaleClassifier` is no longer on the path that runs
    the grid. Chatterjee's xi therefore contributes what it was chosen for. Transfer entropy
    still does not run at all, for an unrelated reason: `pyinform` vendors an x86_64-only
    native library and fails on arm64, so the grid covers six of the seven statistics on this
    machine. Run it on the cluster, or precompute and read it back through `Specification`'s
    `ext:` prefix.
11. **Directed and undirected statistics do not share a feature vocabulary.** Comparing them
    naively gives an agreement of exactly zero, which reads as disagreement but is a naming
    artefact. `runner.collapse_to_undirected` maps a directed attribution onto the undirected
    edge set for the cross-statistic comparison only; section 6 needs a sentence on this, and
    3.4.3 reports the symmetric-only subset as a sensitivity check.
12. **Degenerate cells were being emitted into the grid.** LEMON pretraining alters the
    connectivity measure only for `tangent` and the feature scaler only for a scale-sensitive
    classifier, so five combinations (`xgboost + lemon` on the five stateless statistics)
    computed bit-identically to their `none` twins. The grid was 54 cells containing 49
    distinct computations. Fixed in `Specification.is_valid`, with a test asserting no two
    specifications denote the same computation. The family-wise p-value is unaffected — the
    maximum over a set with duplicates equals the maximum over the distinct set, for the
    observed statistic and every permutation alike — but the reported grid size, the mean
    attribution agreement and the pretraining variance component all were. All three are
    recomputed.
13. **`MultiScaleClassifier`'s regional block is zero by construction.** `ExtractH1Features`
    returns `dataset['timeseries'].mean('timepoint')`, and the parcellated time courses are
    z-scored, so those means are numerical residue: max |mean| = 0.033 against a per-region SD
    of 0.996. Any section-3.2-style comparison built on it would find regional features
    uninformative whatever the data said. Section 3.2 now uses `scripts/run_multiscale.py`
    instead, with all three blocks derived from one region-level matrix. The class itself is
    unused on the grid path but is still exported; decide whether to fix or remove it.
14. **Aggregation order is an undeclared degree of freedom, and a large one.** Averaging
    region time courses into networks before estimating connectivity, versus estimating
    region-level connectivity and averaging afterwards, moves accuracy by 0.13 to 0.19 on
    identical data (section 3.2) — larger than any axis in the registry. Every network-scale
    specification here aggregates first. Adding it as a fifth axis would change every
    specification identifier and require rerunning the family-wise null, so it is currently
    declared as a limitation. Decide before submission whether to run it.
15. **The motion confound needs a decision, not just a disclosure.** Section 3.5 now reports
    that gamers moved more, that motion alone decodes at 0.61, and that FPN-SMN does not
    survive controlling for it. Options: report as-is with the limitation (current), add a
    motion-matched subsample analysis (loses too many participants at n = 32), or add
    scrubbing as a preprocessing axis in the multiverse (defensible, and the multiverse is the
    natural home for it, but it doubles the grid).

16. **The gong2015 attribution result is withdrawn, and coauthors may hold the old draft.**
    Two defects invalidated it: `ants.get_mask` left skull in the volumes, and the Dosenbach
    sphere atlas gives near-arbitrary cross-dataset volumes. The decoding null survives and
    is slightly stronger; the attribution and the "dissociation replicates across modality"
    claim do not. Anyone circulated an earlier §3.6 needs telling.
17. **Attribution does not transfer across dataset or modality.** Discovering networks by
    attribution in one and testing in the other fails in both directions (p = 0.71, p = 1.00),
    and the structural analogue of the three networks puts the *control* networks above them.
    Section 3.4.3's stability claim is therefore about a measurement, not about biology, and
    the Discussion should say so rather than leaving the scope implicit. **Refined by item
    6d:** the permuted-label null shows the stability is distinguishable from what shuffled
    labels produce on a classifier-diverse grid (p = 0.035), so it is established as a claim
    about these participants — but it does not thereby become a claim about biology, and the
    transfer failures recorded here still bound it.
18. **A label off-by-one in the Schaefer atlas tables.** nilearn returns 201 names for 200
    parcels with `Background` first; indexing with `label - 1` gave every parcel its
    neighbour's network. Volumes were keyed by label id and stayed correct, so only the
    *interpretation* was wrong -- the failure mode that produces confident false conclusions.
    Fixed in `prepare_atlases.py`; the cached functional timeseries still carry the old names
    and should be regenerated or read by position.
19. **Decide whether to add the ceiling analysis as a standalone contribution.** Section 11 of
    `decodability_ceiling.md` -- reliability caps accuracy near 0.72, and at n = 32 that
    ceiling coincides with the null's 95th percentile -- is general, novel, and independent of
    the AVGP question. It may be worth its own short methods paper rather than a subsection.

20. **The three networks were never a hypothesis, and three documents said they were.**
    Section 1.5 was written as an *a priori* prediction of control-to-sensorimotor coupling;
    `run_inductive_bias.py` argued its direction "comes from other people's data";
    `run_network_replication.py` labelled the restriction "the pre-registered restriction".
    All three are corrected. FPN, CON and SMN came from section 3.1, on these participants.
    The consequence is that **no family-wise p-value in this paper prices the 0.812 result**
    — not 0.0010, not 0.0066, not 0.0399 — and section 3.4.5 now says so. Two checks bound
    what it can mean: it survives no size-matched comparison (3.5) and does not transfer
    (3.6). Anyone who saw an earlier draft claiming a corrected-significant restricted result
    needs telling. Watch for this pattern rather than this instance: a reviewer asking a
    finding to be *motivated* is not licensing it to be *redated*.

21. **Graph metrics were never tried, and the registry has a natural place for them.**
    `src/acnets/connectome/swn.py` computes small-world features with a permutation test and
    is unused. Graph-level features — modularity, participation coefficient, global
    efficiency — are an alternative feature representation, not another statistic, and they
    are the representation the multi-network account of section 2.1 is most naturally stated
    in: integration and segregation are graph properties. Adding them widens the family, so
    section 3.4.5 applies; but unlike most widenings this one is motivated by a competing
    account rather than by search. Carried over from the analysis roadmap, which is otherwise
    superseded and has been removed.
