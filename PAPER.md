# AF2-CrossKAN-DTA — Structure-Aware Drug–Target Affinity Prediction

**A Research Summary**

*Author: Turjo Rahman*
*Contact: us.khan.2002@gmail.com*
*© 2026 — All rights reserved. See "Availability & Intellectual Property" below.*

---

## Abstract

Predicting the binding affinity between a drug molecule and a protein target
(drug–target affinity, **DTA**) is a central problem in computational drug
discovery: a model that can rank how strongly candidate compounds bind a target
can dramatically narrow the search space before any wet-lab experiment.

This work introduces **AF2-CrossKAN-DTA**, a DTA model that combines three
ingredients: (1) protein representations grounded in **real 3D structure** from
the AlphaFold2 database rather than sequence-predicted proxies, (2) an
**interaction-aware attention mechanism** that lets the drug and protein
representations condition on one another before prediction, and (3) a
**learnable-activation predictor** in place of a conventional multilayer
perceptron.

Evaluated on the standard **Davis** benchmark, the model is **competitive with
recent state-of-the-art structure-aware methods on the conventional
(random/"warm") split**, and — more notably — retains strong **ranking ability
on the much harder "unseen-protein" split**, where the test proteins are never
observed during training. This second result is the more scientifically
interesting finding: it indicates that the model's structure-grounded
representations *generalise* to novel proteins rather than merely memorising
seen ones.

The full architecture, implementation, and training recipe are **not disclosed
in this document** (see below).

---

## 1. Motivation

Most published DTA models are evaluated on a *random* split of the data, where
every drug and every protein in the test set has already been seen (in other
pairings) during training. On such splits, strong numbers can be achieved partly
by **memorising the identities** of the drugs and proteins involved. This
inflates apparent performance and says little about how a model will behave on a
genuinely new target — which is exactly the case that matters in real drug
discovery.

Two questions motivate this work:

1. Does grounding the protein representation in **real 3D structure** (instead of
   sequence-predicted contacts) improve affinity prediction?
2. Does that structural grounding help the model **generalise to proteins it has
   never seen** — the regime where memorisation is impossible?

---

## 2. Approach (High-Level)

> **Note.** This section describes the approach only at a conceptual level.
> Architectural specifics, layer configurations, the fusion mechanism, the
> training procedure, and the source code are intentionally omitted.

At a high level, AF2-CrossKAN-DTA:

- Represents each **protein** using both its amino-acid sequence *and* a graph
  derived from its **experimentally-plausible 3D fold** (AlphaFold2), so that the
  model sees true spatial proximity between residues rather than a statistical
  guess from sequence alone.
- Represents each **drug** from both its chemical string and its molecular graph.
- Allows the drug and protein representations to **interact and condition on one
  another** before a final prediction is made, rather than encoding each side in
  isolation and concatenating.
- Produces the final affinity estimate with a **learnable-activation predictor**.

The specific way these components are combined — which is the novel contribution
of this work — is proprietary and not described here.

---

## 3. Experimental Setup

- **Benchmark:** Davis kinase binding dataset (log-scale pKd).
- **Splits:**
  - *Warm / random split* — the conventional protocol; all drugs and proteins
    appear (in other pairings) in training.
  - *Unseen-protein (cold) split* — every test protein is entirely held out of
    training; the model must generalise to novel targets.
- **Metrics:** Mean Squared Error (**MSE**, lower is better), Concordance Index
  (**CI**, ranking quality, higher is better), and **r²ₘ** (range-faithfulness).
- **Protocol:** multiple independent runs across different data-split seeds,
  matching the evaluation protocol of the closest prior work.

---

## 4. Results

### 4.1 Conventional (warm) split — competitive with the field

On the standard Davis random split, AF2-CrossKAN-DTA sits **within the
competitive cluster of recent methods**, matching the direct sequence-structure
predecessor and approaching the strongest published structure-based model.

| Method (Davis, warm split) | MSE ↓ | CI ↑ |
|----------------------------|-------|------|
| Strongest published structure-based baseline | 0.184 | 0.917 |
| Recent sequence-structure predecessor         | 0.204 | 0.898 |
| Other recent graph baselines                   | 0.20–0.23 | 0.89–0.90 |
| **AF2-CrossKAN-DTA (this work)**               | **≈ 0.20** | **≈ 0.88** |

*Baseline values are as reported in the respective publications; evaluation
protocols differ slightly between papers, so this table reflects the competitive
landscape rather than a perfectly controlled head-to-head.*

The takeaway on the warm split is that the benchmark is **near-saturated** — the
top methods are separated by small margins, and additional capacity yields
diminishing returns. This motivated evaluation on a harder regime.

### 4.2 Unseen-protein (cold) split — the key finding

On the far harder unseen-protein split, absolute error rises substantially (as
it does for *all* methods, since the model can no longer rely on having seen the
target). Crucially, the model's **ranking ability is largely preserved**:

| Setting | MSE ↓ | CI ↑ |
|---------|-------|------|
| Warm split (proteins seen)        | ≈ 0.20 | ≈ 0.88 |
| **Unseen-protein (never seen)**   | **≈ 0.41** | **≈ 0.84** |

The concordance index falls only modestly (≈ 0.88 → ≈ 0.84) even though the test
proteins were **never seen during training**. In other words, the model
continues to rank drug–protein affinities correctly for *novel* targets — the
signature of genuine generalisation from structure/sequence features rather than
memorisation of seen entities.

This is the result the work is built around: a structure-grounded model that
**degrades gracefully** on unseen proteins, which is the setting that matters for
prospective drug discovery.

---

## 5. Discussion

- **Warm-split benchmarks are close to saturated.** Differences between leading
  methods largely reflect split/seed noise. Reporting *only* warm-split MSE
  overstates how much any single architecture "wins."
- **The interesting axis is generalisation.** Evaluating on unseen proteins
  exposes whether a model has learned transferable structure–affinity
  relationships or simply memorised the training entities. AF2-CrossKAN-DTA's
  retained ranking ability on unseen proteins is evidence for the former.
- **Structure grounding is a plausible driver.** Because the protein
  representation is anchored in real 3D geometry, the features it learns are tied
  to properties that transfer across proteins, not to protein identity.

---

## 6. Availability & Intellectual Property

This document is a **results summary intended for public dissemination**. It
deliberately excludes the information required to reproduce the work:

- The **model architecture** (component configurations, dimensions, the specific
  interaction and fusion design) is **not disclosed**.
- The **source code**, trained weights, preprocessing pipeline, and training
  configuration are **not released** with this document.
- The described method and its novel combination of components are the
  **intellectual property of the author**.

Requests regarding collaboration, evaluation, or licensing may be directed to the
contact address above. Reproduction of the *method* or *implementation* is not
authorised; this summary is provided for informational purposes only.

**© 2026 Turjo Rahman. All rights reserved.**

---

*This summary reports work in progress; some multi-seed results are still being
finalised and headline numbers are stated approximately. It is not a
peer-reviewed publication.*
