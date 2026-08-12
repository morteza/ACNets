# Supplementary material

Companion to `manuscript.md`. This file holds the editorial record — where each section came
from, how coauthor comments were dispositioned, and what was cut from the original draft. None
of it is intended for a reader of the paper; it exists so that the provenance of the prose stays
recoverable while the manuscript itself reads as continuous text.

Figures (Appendix B) and the pre-submission open-items list (Appendix E) stay in the
manuscript, since the first belongs to the paper and the second is the submission checklist.

---

## Appendix A: Provenance and editorial notes


### A.1 Section provenance

Where each section's prose came from. "reused" means text from the original manuscript with
em-dashes converted to ASCII punctuation and track-changes residue removed.

| Section | Provenance |
|---|---|
| Abstract | new, though the closing theoretical claim is the original abstract's. 205 words. |
| 1.1 Action video games and cognition | reused, near-verbatim |
| 1.2 Neural correlates of action video game play | reused, with additions |
| 1.2 Neural correlates of action video game play | new, filling the original's citation to-do list |
| 1.3 The brain is organised into interacting large-scale networks | reused; this text was in the original's "OUT TEXT" block and is restored |
| 1.4 Why resting state | reused, near-verbatim |
| 1.5 On-line control motivates looking at control-to-sensorimotor coupling | new; this is Bavelier's comment about the sensorimotor loop. **First written as an a priori prediction; that framing is withdrawn** — see the entry below |
| 1.6 Hidden degrees of freedom | new |
| 1.7 The present study | new |
| 2. The specification space | new section, replacing the original "Hypotheses"; the three families are the original's, reworded |
| 2.2 The registry | new |
| 2.3 The pre-specified primary analysis | new; this is what makes the primary p-value legitimate |
| 2.4 Negative controls | reused in substance from the original's control analyses, promoted out of the supplement |
| 3. Results | restructured; result prose reused where it survives, with numbers as slots |
| 3.2 Which choices matter, and by how much | reused, corrected |
| 3.3 Attribution is more stable than accuracy | new; the paper's central result |
| 3.3.1 What would disconfirm the thesis | new |
| 3.4 The stable signature is control-to-sensorimotor coupling | reused, restructured around selection frequency instead of a single ranking |
| 3.5 Feature blocks in competition | new analysis, but it implements the plan in the original draft's 19/09/2023 working notes |
| 3.6 Negative controls and noise ceiling | reused from the original's supplementary S3, promoted to main text |
| 3.6 Negative controls and noise ceiling | new |
| 4.2 Action video gaming and the crosstalk between large-scale networks | reused, near-verbatim |
| 4.3 Control and sensorimotor networks | reused, near-verbatim; this is among the strongest writing in the original draft |
| 4.4 Relation to prior work | new; the novelty delimitation the paper needs |
| 4.5 Limitations | new subsection, gathering material previously scattered across the conclusion and supplement |
| 5. Conclusion | reused with edits |
| 6. Methods | reused; changes are flagged inline |
| 6.3 Parcellation and aggregation | new |
| 6.5 Classification | new |
| 6.6 Inference | new; replaces the original's interval and p-value construction |
| 6.7 Explanation | new; the original used permutation importance alone |
| 6.8 Reproducibility | new |

### A.2 Notes lifted from the body

These were inline annotations in earlier drafts. They are collected here so the body reads as
continuous prose; the coauthor-comment disposition table in Appendix C is the authoritative
index.

**1.1 Action video games and cognition.** Coauthor comments addressed here: Cardoso-Leite on the Green, Li and Bavelier (2010) and Cardoso-Leite et al. (2021) citekeys; Bavelier and the anonymous reviewer on distinguishing action games from video games in general; Kühn cited for genre variety rather than as action-game evidence, per Cardoso-Leite's note that she used Super Mario. The incomplete sentence ending "by stressing motor, perceptual, and cognitive;" is completed. Framing should be checked against Bavelier and Green (2025, *Current Directions in Psychological Science*), which postdates the original draft.

**1.5 On-line control motivates looking at control-to-sensorimotor coupling.** This section addresses Bavelier's comments "I would rather link the AVGP work to on-line control and the importance of sensori-motor loop during such on-line monitoring which goes with the FPN really" and "this comes out of nowhere, needs to be motivated". In the original the SMN result was described as "somewhat novel or unexpected" and justified after the fact.

**The first revision overcorrected, and this entry is the record of it.** That draft rewrote the section as a prediction made in advance, and this file said so approvingly: *"Here it is predicted in advance, which is worth considerably more."* It was not predicted in advance. The section was written after section 3.1 had already identified CON-SMN and FPN-SMN, in response to a reviewer comment asking for motivation — which is a request to *motivate* a finding, not a licence to redate it. Presenting a post hoc rationale in the grammar of a prediction is the mechanism by which honest exploratory work turns into an unreproducible confirmatory claim, and it is worth considerably *less* than saying what happened.

The section now motivates the region of interest from the gaming literature, states that the specific contrast came from these data, and hands the consequences to 3.4.5. Bavelier's comment is still addressed: the SMN result is no longer unmotivated. It is motivated as a finding rather than dated as a hypothesis.

**2.4 The pre-specified primary analysis.** This addresses Bavelier's comment "I sincerely think we will be dinged here, you cannot test a methodo without ground truth". She is right that a methodology cannot be validated without ground truth, and the reframed design does not attempt it. We measure how far a conclusion moves when the methodology moves, which needs no ground truth. Her proposed yardstick of better-than-chance classification does not by itself resolve the problem, because better-than-chance is precisely the test being run 280 times; the max-statistic null in section 3.4.1 is what resolves it.

**3.5 Feature blocks in competition.** This addresses Bavelier's comment "will we then have a best model?" and the anonymous comment expecting four levels of parcellation. The four levels in that comment (random, regions, within-network, between-network) map onto the random-network control plus the three feature blocks here.

## Appendix C: Coauthor comment disposition


| Commenter | Comment | Where addressed |
|---|---|---|
| Cardoso-Leite | Kühn used Super Mario, not action games | 1.1, cited for genre variety only |
| Cardoso-Leite | Green, Li and Bavelier (2010) citation | 1.1 |
| Cardoso-Leite | Cardoso-Leite et al. (2021) citation | 1.1 |
| Cardoso-Leite | Fill in details on the tangent metric | 6.4 |
| Cardoso-Leite | Step 2 uses independent runs, step 3 does not? | 6.5, pipeline made explicit; Figure 1 to be redrawn to match |
| Anonymous | Boost the AVG neural bases review; distinguish AVG from VG | 1.1, 1.2 |
| Anonymous | Introduce the methodological variables in the intro | 1.6, 2.2 |
| Anonymous | Present the 3 parcellation x 5 metric = 15 models | 2.2, now 280 specifications |
| Anonymous / Bavelier | Expected FOUR levels of parcellation | 2.4 plus 3.5: random-network control plus three feature blocks |
| Bavelier | Use "parcellation" consistently | throughout |
| Bavelier | Where is this shown? Text needs structure | 3.1 to 3.6 restructured |
| Bavelier | Metrics need introducing in the text and intro | 1.6, 2.2 |
| Bavelier | Need more VG lit; distinguish VG from AVG | 1.1, 1.2 |
| Bavelier | Could move as a discussion point, remove from intro | 4.1 |
| Bavelier | Link AVGP work to on-line control and the sensorimotor loop | **1.5, the pivot of the reframe** |
| Bavelier | This comes out of nowhere, needs motivating | 1.5 |
| Bavelier | Will we then have a best model? | 2.3 primary specification; 3.5 feature blocks |
| Bavelier | Does this still apply? | resolved during the rewrite |
| Bavelier | Cannot test a methodology without ground truth | 2.3, and see the note there on why the yardstick alone is insufficient |

## Appendix D: Text removed from the original


- The "TEXT DB REMOVED", "OUT TEXT", and "Notes 19/09/2023" working blocks. The substantive
  content of the OUT TEXT block is restored in 1.3; the 19/09/2023 analysis plan is
  implemented in 3.5.
- The claim that the region-level control shows the Dosenbach parcellation is effective.
  Region- and network-level specifications differ in dimensionality by a factor of about
  600, so only the random-network control supports that inference (3.2).
- The post-hoc t-tests in the original section on group differences. Replaced by descriptive
  effect sizes with bootstrap intervals and an explicit circularity note (3.4).
- The standalone Discussion subsection on methodological choices increasing confidence. That
  content is now the paper's spine and needs no separate home.
- Track-changes residue: "associationconnectivity", "most accuratebest", "the the mappings",
  "In all the casestheother cases", "The second stantent", the duplicated Bediou 2023
  citation, and the incomplete sentence ending "by stressing motor, perceptual, and
  cognitive;".
