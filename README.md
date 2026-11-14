# her2 stopping rule

Multi-objective antibody design pipeline for trastuzumab/HER2: affinity, epitope preservation and thermostability, with calibrated filters and an explicit stopping rule.

Built at the **AI × Antibodies Hackathon, Oslo, 14 November 2026**.

---

## The brief, in one page

Improve trastuzumab, a clinical antibody against HER2. Adaptyv expresses our designs and
measures three properties against a measured trastuzumab baseline:

| Property | Measured by | Beat the baseline? |
| --- | --- | --- |
| Binding to HER2 | SPR (KD) | comparable or better |
| Epitope preservation | competitive binding vs trastuzumab | preserved |
| Thermostability | direct measurement | improved |

**A design is tiered by how many of the three it improves.** S-tier is 3 of 3, A-tier 2, B-tier 1.
Teams rank by S-tier count first, then A-tier. *Ten designs that each improve one property lose to
two that improve all three.* This single sentence should drive every cut we make.

### Two tracks, decided for us

`abguard` classifies each design; we do not choose the track. It is never reclassified — miss your
track's rule and the design is rejected.

The track follows from one number, **identity of the whole Fv to trastuzumab**:

- **≥ 70% identity → `novel`** (the optimisation track): an engineered trastuzumab variant.
- **< 70% identity → `de_novo`**: a molecule of its own.

Prizes are split evenly across the two tracks, so a track with fewer entrants is worth more per design.

### Hard filters, all designs

Format VHH, scFv or Fab · no unpaired cysteines · immunum numbering confidence ≥ 0.1 ·
total length ≤ 250 aa.

### The novelty rule, settled

Read from `abguard`'s source (`novelty.py`, `config.py`), not from the prose. The task PDF and the
Apheris playbook state this inconsistently and **both are misleading**; the code is the arbiter:

| | |
| --- | --- |
| What is compared | the **IMGT CDRH3**, Levenshtein distance |
| Trastuzumab's IMGT CDRH3 | **`SRWGGDGFYAMDY`** (13-mer) |
| `novel` needs | **≥ 3** mutations |
| `de_novo` needs | **≥ 2** mutations |
| Measured against | trastuzumab **and every CDRH3 in the reference** — 610,109 loops |

Two traps worth stating plainly:

- **The 70% is on the whole Fv and only picks the track.** It is not a pass/fail, and it is not
  CDRH3 identity, whatever the PDF says.
- **Use the IMGT 13-mer, never the 10-mer core `WGGDGFYAMD`** the assay tables use. abguard
  re-derives every reference loop to IMGT precisely because comparing against the cores "would hand
  every design three free mutations from the flanks alone".

The binding constraint is prior art, not trastuzumab. A loop can sit 3 mutations from trastuzumab and
still be rejected for landing 2 from something in the FACS screen — which is exactly what happened to
one of our first two designs.

### Submission and scoring

- A ranked list of **up to 20 designs**. Everyone gets 4 tested; pitch placement and judge
  nominations add up to 16.
- Pitch: **40% judgment** (why these molecules), 20% execution (reproducible repo), 20% user fit,
  20% path to value.

Judgment is the largest single weight, and it is exactly what this repo is for.

---

## Our thesis

Everyone will generate hundreds of candidates. Almost nobody can defend the cut from 400 down to 20.
**Our product is the funnel, not the generator** — and the agent stops when the next predicted gain
falls below the spread between our scoring sources, and says so.

Two observations that shape the plan:

1. **Nothing on offer optimises thermostability.** The fine-tuned HER2 weights were trained on a
   binding screen. Affinity is where every team will crowd; thermostability is a scored axis with no
   provided model, and epitope preservation is nearly free if we keep the trastuzumab framework and
   only move CDRH3. An optimisation-track design that holds affinity and wins on thermostability is a
   cheaper S-tier than a de novo binder that has to win all three from scratch.
2. **Folding confidence is known to be unreliable here, and we were handed the proof.** The
   Smorodina data has 11,342 non-binding pairs that models still score confidently. Calibrating our
   ipTM/ipSAE cutoff on measured non-binders — instead of picking 0.8 because it is a round number —
   is both a better filter and the most defensible slide in the pitch.

---

## Plan for the day

Six steps. Parallelise: design on the VM's H100, score on Nebius, keep the slow jobs moving.

| # | Step | Where | Output |
| --- | --- | --- | --- |
| 1 | Baseline: fold trastuzumab + HER2 (1N8Z), record its scores under every metric we will use | VM | the number every design is compared to |
| 2 | Calibrate the confidence filter on measured non-binders | VM, CPU | a cutoff with a stated false-positive rate |
| 3 | Generate CDRH3 variants: HER2 novel + exploratory (optimisation), antibody weight (de novo) | VM, GPU | a few hundred designs |
| 4 | Filter: `abguard`, then novelty, then liabilities and expression proxies | VM | the survivors |
| 5 | Score: refold against HER2 with two independent models; predict affinity and thermostability | VM + Nebius | one row per design |
| 6 | Rank, stop, and write down why | here | 20 designs with reasons |

### TODO

**Blockers — settle these first**

- [x] `abguard` — installed and verified, see [abguard](#abguard). Novelty thresholds read from its
      source, so there is nothing to ask the organisers.
- [ ] **Confirm we have the current `abguard` build.** Two different wheels on this VM both call
      themselves `0.1.0` and one has no novelty gate at all (see below). Ask Apheris whether 11 Nov
      is the final one before trusting any verdict.
- [ ] `foundry-toolkit login` — no stored credential, so `module list` and `module describe` fail.
      Probably not blocking: every image is already pulled, and `docker run` needs no login.
- [ ] Decide the track split. Recommendation: weight optimisation heavily, keep a few de novo VHHs as
      a lottery ticket, since the prize pots are equal and de novo will likely be thinner.

**Setup on the VM** (`~/ioannis/her2-stopping-rule`)

Most of the stack is already installed — see [What is on the VM](#what-is-on-the-vm).

- [x] `aws s3 sync s3://apheris-oslo-hackathon/rfd4-proteina/docs/ ~/rfd4` — docs, `Data/`, `Settings/`
      and `Scripts/run-her2.sh`, the 4-design smoke test.
- [x] HER2 tables (6 MB) in `data/raw/her2-trastuzumab/`. **Note:** `data/raw/` is the git-ignored path,
      not `data/` as the Apheris docs suggest. Keep organiser data out of this public repo.
- [x] Smoke test green — see [the smoke test](#the-smoke-test-14-nov-0955).
- [ ] `uv pip install torch` if we train our own affinity or thermostability model (step 5).
      `torch.cuda.is_available()` must print `True` before anything else.
- [ ] Nebius: `bash agent/install.sh`, then `export NEBIUS_CREDS=...`. Keys live outside the repo.
      The VM account is shared with the other teams' folders, so decide before copying keys there.
- [ ] Redeem the JapanFold code ($1,000, one account, the one that runs the jobs) and claim AMASS ($100).

**Pipeline**

- [ ] Step 1 baseline. Everything is relative to trastuzumab; without this number no claim is checkable.
- [ ] Step 2 calibration — the differentiator. Use `sigma2-ab-ag-champloo/` ipTM/PAE per replica and the
      SPR supplementary tables. Note: "no reportable fit" is **not** a true negative; only the 454
      measured non-binders are.
- [ ] Step 3 generation. Start RFD4 jobs early, they are the long pole. `her2-novel-2026-11-09` passes
      novelty 86% of the time with the lowest interface error (6.8 Å) — start there, add
      `her2-2026-11-07 --sequence-tsr-k 1` for variety.
- [ ] Step 4 filters, including the expression proxies Adaptyv recommends: Aggrescan 3D, SoluProt,
      NetSolP, ProteinMPNN soluble weights. A design that does not express scores zero on all three axes.
- [ ] Step 5a affinity: train on the SPR table (190 KDs) and the FACS classes; rank, do not trust absolutely.
- [ ] Step 5b thermostability: no provided model — decide the approach (ddG predictor, or framework
      choice plus disulfide/liability inspection). **The most under-served axis; do not leave it to the end.**
- [ ] Step 5c epitope: co-fold each design with HER2, check the CDRH3 contacts the 1N8Z epitope.
- [ ] Step 6 the stopping rule itself, written as code with its threshold derived from step 2.
- [ ] Ranked list of 20, each with a one-line reason and its run id.

**Pitch** (40% of the score)

- [ ] The judgment story: the funnel, the calibration, the stopping rule.
- [ ] User fit and path to value — 40% combined, and the easiest marks to drop by forgetting them.
- [ ] Check the repo reproduces from a clean clone.

### Deadlines

Designs freeze at **14:00**. The VMs are **shut down at the end of Saturday 14 November** — anything
not pushed to GitHub by then is gone.

---

## Resources

| What | Where | Notes |
| --- | --- | --- |
| Team VM | 1× H100, 16 vCPU, 196 GB RAM, 484 GB disk | all Foundry models as Docker images, already pulled |
| Apheris Foundry | `docker run` on the VM, no Hub | design, refold-prepare, refold-score, metrics, finetune |
| Nebius Scientific AI | `forge.py` skill or browser | ~60 hosted models, no GPU needed |
| JapanFold | japanfold.aiand.com | $1,000 credit, RoseTTAFold3, OpenDDE, AF2-IG |
| AMASS | platform.amass.tech | $100, literature and patents with citations |
| Data bucket | `s3://apheris-oslo-hackathon` | read-only; `data/README.md` is the index |

### The smoke test, 14 Nov 09:55

The organisers' example, `bash Scripts/run-her2.sh all` in `~/rfd4` — **their pipeline, not ours.**
4 designs → prepare top 2 → Boltz-2 refold → score, 245 s cold. Per-step times are in
[notes/RUNTIMES.md](notes/RUNTIMES.md). It proves generate → fold → score works on this GPU. It proves
nothing about our funnel, which does not exist yet.

| Design | IMGT CDRH3 | vs trastuzumab | vs prior art | ipTM | min_ipae | binder pLDDT | abguard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `her2_cdrh3_2` | `SRWGPPYFYAMDY` | 3 | **2** | 0.946 | 0.331 | 95.6 | **rejected** |
| `her2_cdrh3_3` | `SRWASSSEYYFDY` | 7 | ≥ 3 | 0.965 | 0.310 | 98.5 | passes |

Trastuzumab is `SRWGGDGFYAMDY`; both designs classify as `novel` (Fv identity 0.99 and 0.97).
Four things this already tells us:

1. **Half our first designs were rejected, and not by the obvious rule.** Design 2 cleared
   trastuzumab at distance 3, then landed 2 mutations from `SRWGPYGFYAMDY` in the FACS screen.
   **Prior art is the binding constraint, not trastuzumab** — and with 610,109 reference loops it
   will stay that way. Any ranking we build has to run the gate *first*, or we will rank designs
   that cannot be submitted.
2. **`pass` is empty and that is expected** — the scorer cannot measure absolute movement on this
   input. Rank on `min_ipae`, `iptm`, `binder_plddt`.
3. **ipTM 0.95+ is not evidence of binding.** The Smorodina non-binders score this well too. These
   numbers are the input to step 2's calibration, not a result.
4. **The design model does not know the novelty rule.** It optimises predicted fit, and the HER2
   weights were trained on the very screen we must stay away from. Expect a real rejection rate and
   generate with margin.

### What is on the VM

Checked 14 Nov, 09:45. Almost nothing needs installing — the expensive parts are already there.

| Installed | Detail |
| --- | --- |
| All Foundry models | 20 Docker images pulled, ~200 GB: RFD4-Proteina ×3 weights, Boltz-1/2, OpenFold3 ×2, Protenix v1/v2, Mol-1, ESMFold2, MSA |
| Docker + GPU | **verified**: `--gpus all` reaches the H100 from inside a container |
| `foundry-toolkit`, `aws`, `nextflow`, `zstd`, `jq`, `git` | on `PATH` |
| Claude Code 2.1.322, `uv` 0.13.7, `rcsb-mcp`, `codex` | in `~/.local/bin` |
| CUDA toolkit 13.0 | `/usr/local/cuda-13.0` |
| foundry-assistant | local Foundry docs Q&A, models downloaded, `~/foundry-assistant` |

| Missing | Cost to fix |
| --- | --- |
| PyTorch | `uv pip install torch`, minutes, only for our own models |
| Nebius skills | `bash agent/install.sh`, ~1 min |

### abguard

Installed in `.venv-abguard/` (git-ignored) from the wheel already in the shared `uv` cache —
a teammate installed it on 11 Nov, so nothing was downloaded:

```bash
uv venv --python 3.12 .venv-abguard
VIRTUAL_ENV=$PWD/.venv-abguard uv pip install \
  ~/.cache/uv/sdists-v10/path/bd466a53b5e23834/nSBDF8oKc5yFGa5i/abguard-0.1.0-py3-none-any.whl
```

> **Two wheels on this VM both call themselves `abguard 0.1.0`, and they are not the same tool.**
> The 10 Nov build has no `novelty.py`, no reference data and no novelty gate: it validates the hard
> filters, reports every design compliant, and never mentions that it skipped the check that matters.
> We installed it first and got a clean "2/2 valid" on a submission that is **not** valid.
> Always verify before trusting a verdict:
>
> ```bash
> ./.venv-abguard/bin/abguard --help | grep novelty-report   # must print
> ```

Usage, no network and no 4.2 GB DuckDB needed — the wheel bundles the reference index
(610,109 CDRH3s, 2 MB Parquet, IMGT, min_confidence 0.1):

```bash
./.venv-abguard/bin/abguard team-validate submission.csv          # the real gate
./.venv-abguard/bin/abguard team-validate submission.csv --drop   # keep the good rows
./.venv-abguard/bin/abguard novelty-report submission.csv         # trastuzumab only, no prior art
./.venv-abguard/bin/abguard liability-report submission.csv       # LAP liabilities
```

Columns: `sequence_id, kind, heavy_sequence, light_sequence, design_description`, with `kind` one of
`vhh` / `scfv` / `paired`. Two designs validate in 0.5 s, so run it early and often.

**`novelty-report` does not check prior art.** It compares against trastuzumab alone and leaves
`is_novel_enough` null — it says so in a warning that is easy to miss. Only `team-validate` runs the
full gate. Do not use the report as a pre-filter and assume the designs are submittable.

`~/.local/bin` is on `PATH` only in a login shell. Over SSH use `ssh ... bash -lc '...'`, or
`claude`, `uv` and `rcsb-mcp` will look missing when they are not.

**Folding models.** Prefer one that takes a template for binders, which sees the known HER2 structure
and gives a fairer check: Boltz-2, OpenFold3, Protenix v2, Mol-1. Use two independent models and treat
their spread as the noise floor — that spread *is* the stopping rule's threshold.

Two traps already documented: **OpenFold2 never comes up** (fails after ~55 min, both loaded and idle —
use OpenFold3 or Boltz-2), and **do not rank on CDRH3 refold agreement**, because Protenix does not
reproduce trastuzumab's own CDRH3 loop.

### Keys

Credentials stay out of the repo. `*.private.json` and `data/raw/` are git-ignored; point tools at the
file with `NEBIUS_CREDS`, never paste a key into a prompt. **The repo is public.**

---

## Development

Install `uv` if you don't have it already:

```bash
curl --location --silent --show-error --fail https://astral.sh/uv/install.sh | sh
```

See also [installing uv].

```bash
git clone git@github.com:NOI-IoannisVardaxis/her2-stopping-rule.git
cd her2-stopping-rule
make dev     # python virtualenv & dependencies
make test
make lint
make wheel   # check the wheel packages
```

We edit locally, push, and pull on the VM; runs happen on the VM and results are pushed back.
`results/` and `nebius-runs/` hold outputs and run ids, no keys, and are committed — showing where
each score came from is part of the 20% execution score.

### Dependency management

[uv] manages dependencies. Run `uv add package`, or `uv add --dev package` for dev dependencies; the
change lands in `pyproject.toml` and is locked in `uv.lock`. Commit both.

`uv run` ensures the dependencies in `pyproject.toml` are installed before running, so python
environments never have to be managed by hand.

[uv]: https://github.com/astral-sh/uv
[scripts]: https://docs.astral.sh/uv/guides/scripts/
[installing uv]: https://docs.astral.sh/uv/getting-started/installation/
