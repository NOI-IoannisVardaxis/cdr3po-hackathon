# Terminology, tools and thresholds

Everything a judge might ask about, with the number and where it came from.
Written for the team to prepare on, not for the pitch.

**The one-line rule for every number here: if someone asks "why that value",
there is an answer, and it is never "it seemed reasonable".**

---

## The molecule

| Term | What it means here |
| --- | --- |
| **Trastuzumab** | The clinical anti-HER2 antibody we are improving. Measured KD **0.152 nM** (row 1 of the SPR table). Everything is compared against it. |
| **HER2** | The target. We use **domain IV**, the part trastuzumab binds, from PDB **1N8Z**. |
| **Fv** | The variable fragment: heavy chain variable domain (**VH**, 120 aa) + light chain variable domain (**VL**, 107 aa) = **227 aa**. |
| **`paired`** | The format we submit. abguard accepts VHH, scFv or Fab; `paired` is VH+VL with no linker. |
| **Why not scFv** | An scFv joins VH and VL with a linker: 120 + 107 + 15 = 242 aa against a 250 cap, and the linker adds aggregation risk for nothing. `paired` leaves 23 aa of headroom. |
| **CDR** | Complementarity-determining region — the six loops that form the binding site. Three on the heavy chain (H1, H2, H3), three on the light. |
| **CDRH3** | The third heavy-chain loop, the main contact with HER2 and the only region the challenge's novelty rules constrain. |
| **Paratope** | The antibody's binding surface. **Epitope** is the matching surface on HER2. |

### CDR3 numbering — three definitions, and they nest

| Definition | Sequence | Length | Who uses it |
| --- | --- | --- | --- |
| Mutated window | `WGGDGFYAMD` | 10 | the FACS and SPR tables |
| Kabat-style | `WGGDGFYAMDY` | 11 | common in the literature |
| **IMGT CDR3** | `SRWGGDGFYAMDY` | **13** | **abguard — this is the one that counts** |

They nest: every provided variant is `SR` + the 10-mer + `Y`, with the flanks
constant. So Hamming distance is identical whichever you use; only the identity
*percentage* differs. We work in IMGT throughout.

**IMGT positions 3–12 are the ten that vary.** Positions 1, 2 and 13 are
constant in every provided sequence and in every design we make.

---

## The two tracks

**We do not choose a track. abguard computes it from the sequence, and a design
is never reclassified — miss your track's rule and it is rejected outright.**

| | Optimisation | De novo |
| --- | --- | --- |
| Identity to trastuzumab | ≥ 70% | **< 70%** |
| CDRH3 mutations required | **> 2** (i.e. ≥ 3) | **≥ 2** |
| How we reach it | native framework + designed CDR3 → ~98% identity | ProteinMPNN redesigns the framework → < 70% |
| Step | `06a_optimisation` | `06b_denovo` |

Measured, by probing abguard with constructs carrying increasing numbers of
framework mutations:

| Framework mutations | Identity | abguard's verdict |
| --- | --- | --- |
| 10 | 0.956 | optimisation |
| 60 | 0.740 | optimisation |
| **80** | **0.648** | **de novo** |
| 100 | 0.559 | de novo |

Prizes are split evenly between the tracks and each is validated
independently, so they are ranked separately and then interleaved into the one
submitted list.

---

## The judged axes

Adaptyv measures three properties against the measured trastuzumab baseline.

| Axis | How Adaptyv measures it | Our evidence |
| --- | --- | --- |
| **Affinity** | SPR (dissociation constant, KD — lower is tighter) | FACS surrogate, AUC 0.920 |
| **Epitope preservation** | competitive binding against trastuzumab | contact recovery vs 1N8Z, threshold derived |
| **Thermostability** | direct measurement (melting temperature) | **our weakest axis — see below** |

**Tiers.** S-tier improves all three, A-tier two, B-tier one. Teams rank by
S-tier count first. **Ten designs that each improve one property lose to two
that improve all three** — which is why the ranking sorts by predicted tier and
not by a weighted score.

---

## The tools

| Tool | What it is | What we use it for |
| --- | --- | --- |
| **abguard** | The organisers' CLI. The only authority on whether a design is submittable. | Every gate. It decides the track, the novelty verdict and the schema. |
| **Boltz-2** | Apheris Foundry co-folding model. Predicts the design bound to HER2. | Step 7, primary. ipTM, pLDDT, the folded complex. |
| **OpenFold3** | Second Foundry co-folding model. **Required for the monetary prize.** | Step 7, secondary. Its disagreement with Boltz-2 is the stopping rule's noise floor. |
| **ProteinMPNN** | Inverse folding — given a backbone, proposes sequences that fold to it. Served by Nebius as `v_48_020`, the **vanilla** checkpoint. | Step 6b. Redesigns the framework to reach the de novo track. |
| **RFD4-Proteina** | Apheris generative design model. | Not in the full path: its weights only redesign CDRs on a fixed framework, so they cannot reach under-70% identity. |
| **gemmi** | Structure library. | Contact maps, disulfide geometry, epitope recovery. |
| **scikit-learn** | `HistGradientBoostingClassifier`. | The FACS affinity surrogate. |

### Terms you will hear

| Term | Meaning | Our number |
| --- | --- | --- |
| **ipTM** | Interface predicted TM-score. The model's confidence in the *interface*, 0–1. | **AUC 0.645 — barely above chance. We do not filter on it.** |
| **pTM** | Predicted TM-score for the whole complex. | recorded, not filtered on |
| **pLDDT** | Per-residue confidence, 0–100. A foldedness proxy. | baseline 96.2 |
| **PAE / ipAE** | Predicted aligned error. Lower is more confident. | recorded |
| **Co-folding** | Predicting the complex from sequences jointly. Not docking. | step 7 |
| **Docking** | Sampling rigid-body poses of two known structures. | **not used** — only small-molecule docking (DiffDock) is available, and co-folding beats classical docking on antibody–antigen |
| **Inverse folding** | Given a backbone, design a sequence that folds to it. | ProteinMPNN, step 6b |
| **ΔΔG** | Predicted change in folding free energy. Negative is stabilising. | **not used** — see thermostability below |
| **Hamming distance** | Number of positions at which two equal-length sequences differ. | the novelty metric |

---

## Every threshold, and where it came from

| Threshold | Value | Derivation |
| --- | --- | --- |
| **CDR3 mutations** | ≥ 3 | **Measured.** abguard reports `required_cdrh3_mutations = 3`; four of six three-mutation probes passed. The task PDF's "under 70% identity" wording implies 4 on a 13-mer, and building to that would have excluded the best region. |
| **Prior-art distance** | ≥ 3 from all 525,063 | abguard's rule. Both provided tables, not just FACS — probes were rejected against the SPR table specifically. |
| **Edit distance designed at** | 3 | **Measured.** Distance 3 is 90.0% high binders; distance 4 is 74.6%; distance 7 — where the provided HER2 weights generate — is 33.6%. |
| **Affinity shortlist** | top 400 | A ranking cut, not a rejection. Nothing is disqualified by it. |
| **Epitope recovery** | **0.900** | **Derived in-run.** Trastuzumab folded by the same model recovers 7 of 7 native HER2 contacts, so the bar is 90% of that. The constant we had invented was 0.75 — fifteen points too lenient. |
| **Contact cutoff** | 5.0 Å | Parameter, with the sensitivity recorded: 4.0 Å gives 2 contact positions, 4.5–5.0 Å gives 4, 6.0 Å gives 6. 5.0 is the stable side of that cliff. |
| **ipTM cutoff** | 0.9514 **recorded, not applied** | Calibrated at 5% FPR on 8,190 measured non-binders. Not used as a filter because the metric's AUC is 0.645. |
| **Stopping rule** | median two-model spread | The disagreement between Boltz-2 and OpenFold3 on our own designs. Below that gap, a ranking difference is noise. |
| **Submission split** | ranks 1–12 optimisation, 13–20 de novo | Only the top N is tested and N depends on pitch placement, so de novo costs none of the guaranteed four. |

---

## The findings that inverted our priors

Each one is a place where the obvious move was wrong and a measurement said so.
**This is the 40% judgment score.**

**1. The mutation floor is 3, not 4.** The brief implies four; abguard enforces
three. Distance 3 is 90.0% high binders against 74.6% at distance 4 — the
strict reading would have thrown away the richest region on the board.

**2. Contact residues should be mutated, not frozen.** We computed the
CDR3–HER2 contact map and froze the four touching residues. Then we checked:
contact-intact designs are **67.5%** high binders, designs that mutate a contact
are **81.9%**. The novelty rule forces a mutation budget; refusing to spend it on
contacts means spending it on positions that carry more function. D12 sits
9.3 Å from HER2, touches nothing, and halves binding when changed.

**3. ipTM cannot be used as a filter.** Boltz-2 separates 91 cognate pairs from
8,190 measured non-cognate ones at **AUC 0.645**. 94.3% of measured non-binders
score ≥ 0.70; 81.7% clear 0.80. Our own designs score 0.946–0.965, inside the
non-binder range. The metric every team will filter on is barely better than
chance here.

**4. abguard screens patents.** Designs were rejected against
`uspto:uspto_WIO11640`. Local filtering against the provided tables is necessary
and **provably insufficient** — and the highest-affinity designs collide most,
because they sit closest to trastuzumab.

**5. The engineered disulfide cannot be submitted.** Both chains already carry
two cysteines; adding one makes three, and abguard's per-chain parity check
reads an inter-chain disulfide as two unpaired cysteines. 48 of 48 rejected
against 0 of 48 native.

---

## Thermostability — say this honestly

**It is our weakest axis and we should not pretend otherwise.**

- Trastuzumab is an already-optimised clinical antibody; moving CDRH3 barely
  touches melting temperature.
- The engineered disulfide, which was our intervention, is rejected by abguard.
- No tool available predicts Tm. ΔΔG predictors carry ~1 kcal/mol error and are
  built for single mutations, while we move three at once.

**What to say if asked:** "Neither tool predicts melting temperature. We treat
any stability output as a rank, not a temperature. Our stability argument on the
de novo branch is inverse folding — ProteinMPNN optimises sequence-structure
compatibility, which is a genuine if indirect stability argument. On the
optimisation branch we do not claim a thermostability improvement."

Naming the limitation scores better under a judgment rubric than overclaiming,
and a judge who finds it themselves scores it far worse.

---

## Likely questions

**"Your surrogate was validated at distance 4 but applied at distance 3."**
True, and conservative — the production model does train on distance-3 data. We
held out a whole edit-distance stratum rather than splitting randomly, because
variants one mutation apart are near-duplicates and a random split would have
flattered us.

**"How do you know the epitope is preserved?"** We fold each design with HER2
and recompute which HER2 residues the CDR3 touches, against the native 1N8Z set.
**Without a template** — so the model is not told where trastuzumab binds.
Folded that way, trastuzumab itself recovers 7 of 7 contacts, which is what sets
the threshold.

**"Why not dock?"** Only small-molecule docking is available, and co-folding
outperforms classical rigid-body docking on antibody–antigen.

**"Why only N designs?"** The stopping rule. Past the point where predicted gain
falls below the disagreement between two folding models, the ranking is reading
its own noise.

**"Why should we believe any of this?"** `shortlist.csv` names, for every design
that reached scoring, the step that rejected it and why. `funnel.json` records
each threshold beside the counts it produced.
