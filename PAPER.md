# AF2-PocketCross-DTA — Does Real Protein Structure Help Predict Drug Binding?

**A Research Summary**

*Author: Turjo Rahman*
*Contact: us.khan.2002@gmail.com*
*Status: work in progress — not peer reviewed. Figures below are 2-seed means; a third seed is running.*

---

## Abstract

Predicting how strongly a drug binds a protein (drug–target affinity, **DTA**) lets
researchers narrow thousands of candidate compounds down to a handful worth testing in a
lab. The hardest and most useful version of the problem is predicting binding for a
protein the model has **never seen before** — because that is the situation in real drug
discovery.

This work asked a specific question: **does grounding the protein representation in real
3D structure from AlphaFold2, instead of contacts guessed from sequence, make this
easier?**

Measured under a controlled comparison, the answer is **no**. Substituting real structure
for sequence-predicted contacts did not improve accuracy, and independent work published
since reports the same effect on other benchmarks.

The model itself is competitive: on the unseen-protein split of the DAVIS kinase
benchmark it **ranks candidate drugs as accurately as the best published method**. But
its remaining error is not something better modelling can fix — and part of it turns out
to be caused by a flaw in the benchmark itself that, as far as I can determine, has not
previously been reported.

---

## 1. The question

Most DTA models are evaluated on a *random* split, where every drug and protein in the
test set has already appeared in some other pairing during training. Strong scores there
can come partly from memorising which entities are involved, which says little about
performance on a genuinely new target.

This work therefore evaluates on the **unseen-protein** split: all test proteins are held
out of training entirely. Memorisation is impossible; only transferable knowledge helps.

The hypothesis was that **real 3D structure** would transfer better than
sequence-predicted contacts, because it reflects actual physical proximity between
residues rather than a statistical guess.

---

## 2. What was built

A model combining:

- **Protein** — its amino-acid sequence encoded by a protein language model, plus a graph
  of which residues sit close together in the AlphaFold2 structure
- **Drug** — its chemical string and its molecular graph
- **Interaction** — drug atoms and protein residues attend to each other directly, rather
  than each side being summarised separately and then concatenated
- **Predictor** — a Kolmogorov–Arnold Network, which learns its own activation functions
  instead of using fixed ones

13.55 M parameters, trained with plain mean-squared error.

---

## 3. How it was tested

- **Benchmark** — DAVIS kinase binding data (68 drugs × 442 kinases)
- **Split** — unseen-protein: the 44 test proteins never appear in training
- **Metrics**
  - **MSE** — how far off the predicted binding strength is *(lower is better)*
  - **CI** — how often two compounds are placed in the correct order *(higher is better)*
  - **r²ₘ** — whether predictions span the true range instead of hedging toward the average
- **Protocol** — independent runs on different data splits, matching the evaluation
  protocol of the closest prior work so the numbers are comparable

---

## 4. Key findings, in plain terms

### Finding 1 — Excellent at *ordering* candidates, weaker at *scoring* them

| | This work | Best published | Standing |
|---|---|---|---|
| Ordering compounds correctly (CI) | **0.857** | 0.857 | **tied 1st of 9** |
| Range fidelity (r²ₘ) | 0.519 | 0.556 | 2nd of 9 |
| Absolute accuracy (MSE) | 0.356 | 0.314 | 4th of 9 |

**What it means:** hand it a target and 3,000 compounds and it puts them in close to the
right order — as well as anything published. Ask *how tightly* a particular compound
binds and it is less reliable. For screening, the ordering is what matters.

### Finding 2 — Real 3D structure did **not** help

This was the central hypothesis, and it failed. The model using **real AlphaFold2
structure** scores 0.356; the closest prior method, using contacts **guessed from
sequence**, scores 0.314. Swapping in real structure made results slightly *worse*.

Independent work published since (arXiv:2606.04228) measured the same effect on other
benchmarks: for binding affinity specifically, structural features add *negative* value
over protein-language-model features.

**What it means:** the sequence already carries most of what a model needs. Knowing the
fold adds surprisingly little on top.

### Finding 3 — The benchmark itself is partly broken *(previously unreported)*

DAVIS lists **442 protein entries but only 379 distinct sequences.** The mutations were
never written into the sequence strings — so `BRAF` and `BRAF(V600E)` are **identical
inputs** to any model, yet carry different correct answers.

- **19% of training examples contradict each other** — same question, different answer key
- **11% of test questions are unanswerable** by construction
- Together this accounts for roughly **7% of the total test error**, and no architecture
  can fix it

**What it means:** part of the exam has duplicate questions with conflicting answers.
Every published result on this benchmark is affected.

### Finding 4 — The remaining error is missing knowledge, not miscalibration

A natural assumption is that predictions are merely mis-scaled and could be corrected
afterwards. This was tested directly: the **best possible** rescaling — one fitted using
the test answers themselves, which is cheating and unachievable in practice — reached
only 0.339, still short of the leading method's 0.314.

**What it means:** **94% of the error survives even a perfect correction.** The model is
not miscalibrated; it is missing information.

### Finding 5 — A bigger model does not help

Increasing the predictor's capacity **144-fold** made test accuracy *worse*, not better.
The model already fits its training data to within 0.025 of the theoretical noise floor.

**What it means:** the bottleneck is the data, not the model. With only 354 training
proteins there is little left to extract.

---

## 5. What this means

**For this line of research.** The intuition that better protein structure should yield
better affinity prediction is appealing but, on this benchmark, unsupported. Two
independent lines of evidence now point the same way. Effort is better spent on **more
diverse training proteins** than on richer structural features.

**For the field.** Published unseen-protein numbers are **not mutually comparable** —
different papers report the same nominal task with baselines ranging from 0.31 to 0.88,
depending on how the split was built. And the benchmark flaw above caps what any model
can achieve on DAVIS, which may partly explain why published results have plateaued in a
narrow band.

---

## 6. Limitations, stated plainly

- **Two seeds** of the final model, with a third in progress. The r²ₘ spread (±0.038) is
  wider than the gap to the leading method, so no r²ₘ claim is currently decisive.
- **One benchmark, one split setting.** Unseen-drug and unseen-pair are untested.
- **The structure comparison is not yet fully controlled.** The reference method also
  weights its graph edges while this model does not, so contact *source* and edge
  *weighting* differ together. A three-arm experiment separating them is planned.
- **An interpretability claim was tested and withdrawn.** An internal component was
  expected to highlight binding pockets; measured against known catalytic motifs across
  60 kinases, it does not (p = 0.37). No interpretability claim is made.
- **AlphaFold2 for mutants.** Point mutants use the wild-type structure, which is
  defensible structurally but interacts badly with the sequence flaw in Finding 3.

---

## 7. Availability

Code, trained weights, the full experimental log — including every failed variant — and
the analysis scripts behind each finding are maintained in the project repository. Every
number in this summary is reproducible from it.

Requests regarding collaboration, evaluation, or access may be directed to the contact
address above.

Released under the MIT License (see `LICENSE`). The license covers the code; the DAVIS
benchmark data and the pretrained ChemBERTa / ESM-C / AlphaFold2 models each carry their own
terms.

**© 2026 Turjo Rahman.**

---

*Work in progress. Figures are 2-seed means; a third seed is running and headline numbers
may shift slightly. Not peer reviewed.*
