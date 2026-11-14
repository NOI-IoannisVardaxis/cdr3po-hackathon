# Tooling notes

Reference notes on the external tools this pipeline depends on, the traps we hit
using them, and what we learned about each one. Written during the AI × Antibodies
Hackathon (Oslo, 14 November 2026) and kept because the lessons outlive the event.

**On paths and hosts:** the team VM was shut down at the end of the hackathon, so
absolute paths like `~/ioannis/` and `~/rfd4` are historical. The commands, model
names, costs and caveats still apply wherever the tools are installed.

For what the pipeline is → [README.md](../README.md).
For how to install and run it → [PIPELINE.md](../PIPELINE.md).

---

## Contents

- [abguard — the submission gate](#abguard--the-submission-gate)
- [Nebius Scientific AI](#nebius-scientific-ai)
- [The thermostability axis](#the-thermostability-axis)
- [JapanFold](#japanfold)
- [Folding models, and which metric to rank on](#folding-models-and-which-metric-to-rank-on)
- [Known sharp edges in the code](#known-sharp-edges-in-the-code)
- [The environment we ran on](#the-environment-we-ran-on)

---

## abguard — the submission gate

`abguard` is the organisers' CLI. It has the last word on whether a design is
submittable, so the pipeline runs it on real constructs throughout rather than only
at the end.

Install from the wheel into its own virtualenv, kept out of the project environment:

```bash
uv venv --python 3.12 .venv-abguard
VIRTUAL_ENV=$PWD/.venv-abguard uv pip install <path-to>/abguard-0.1.0-py3-none-any.whl
```

> ### ⚠️ Verify the build before trusting any verdict
>
> **Two wheels in circulation both call themselves `abguard 0.1.0`, and they are not
> the same tool.** The 11 Nov build has the novelty gate. The 10 Nov build has no
> `novelty.py`, no reference data and no novelty gate: it validates the hard filters,
> reports every design compliant, and never mentions that it skipped the check that
> matters. We installed it first and got a clean "2/2 valid" on a submission that was
> **not** valid.
>
> ```bash
> ./.venv-abguard/bin/abguard --help | grep novelty-report   # must print
> ```
>
> `s02_guard` refuses to run against the wrong build.

### Usage

No network and no 4.2 GB DuckDB needed — the wheel bundles the reference index
(610,109 CDRH3s, 2 MB Parquet, IMGT numbering, `min_confidence` 0.1).

```bash
abguard team-validate submission.csv          # the real gate
abguard team-validate submission.csv --drop   # keep only the good rows
abguard novelty-report submission.csv         # trastuzumab only — NOT a prior-art check
abguard liability-report submission.csv       # LAP developability liabilities
```

Columns: `sequence_id, kind, heavy_sequence, light_sequence, design_description`,
with `kind` one of `vhh` / `scfv` / `paired`. Two designs validate in 0.5 s, so run
it early and often.

### Two traps

- **`novelty-report` does not check prior art.** It compares against trastuzumab
  alone and leaves `is_novel_enough` null, in a warning that is easy to miss. Only
  **`team-validate`** runs the full gate. Do not use the report as a pre-filter and
  assume the designs are submittable.
- **It screens prior art we do not hold.** A design was rejected against
  `uspto:uspto_WIO11640` — a **patent** sequence. Our local blacklist knows only the
  CDR3s in the two supplied tables, and the comparison differs too (local: Hamming on
  a 10-residue window; abguard: Levenshtein on the 13-mer, across lengths). No local
  work replicates it, which is why `13_submit` drops and refills rather than raising.

---

## Nebius Scientific AI

~60 hosted models, no GPU needed. Credentials live outside the repo; point the client
at them with `NEBIUS_CREDS`.

```bash
export NEBIUS_CREDS=/path/to/credentials.private.json
F=~/.claude/skills/nebius-scientific-ai/forge.py
python3 "$F" models                 # 34 of 59 have a tested input format
python3 "$F" schema <model>         # never ask for the full catalog, it is ~930 KB
python3 "$F" runs | status op-xxx | wait op-xxx | fetch op-xxx --dest results/run1
```

**Why it earns its place** even when a local GPU is available:

- **Parallelism.** Design generation is the long pole on one GPU. Anything scored here
  runs while the GPU keeps generating.
- **Models a Foundry VM does not have:** `antifold` (antibody-aware redesign that can
  hold the CDRs fixed — this is what `s06c_antifold.py` uses), `nanobodybuilder2` /
  `abodybuilder3`, `esm-if1`, the ESM-2 / ESM-C and SaProt families, `proteinmpnn`,
  `stab-ddg`.

> **`stab-ddg` is a *binding* ddG predictor, not a thermostability one.** It scores ΔΔG
> of binding for named point mutations across two partner chains, so it serves the
> **affinity** axis — usefully, since it is independent of the FACS screen everything
> else is trained on. For thermostability use `esm-if1`, not this.

**Operational rules:** never submit the same job twice; after a storage failure use
`resume`, not a new submit. First calls are slow because the model loads on demand —
30 s to 26 min depending on the model.

**If several people share one account,** agree on one working directory. `forge.py`
writes `nebius-runs/` into the *current* directory, and that is how `resume`, `status`
and `fetch` find a run. Launch from different folders and you cannot see or resume each
other's jobs — which, with first calls up to 26 minutes, is how an expensive job gets
duplicated.

---

## The thermostability axis

This is the weakest axis in the pipeline, and it is worth being precise about why.

**The task specification says almost nothing about how it is measured.** The whole of it:

- "Design should improve an antibody's thermostability, as a proxy of real-life
  developability."
- A table row: Thermostability · **"Direct measurement"** · Adaptyv.

No method, no units, no threshold. The participant guide is more useful and names the
tools the organisers expect:

| | |
| --- | --- |
| Readout | **T<sub>m</sub>**, melting temperature — "does it stay folded", via T<sub>m</sub> and accelerated stress |
| General developability target | T<sub>m</sub> above ~65 °C (industry guidance, *not* the challenge bar) |
| Named predictors | **ThermoMPNN** ("predicts how a mutation changes a protein's stability") and **ESM-IF** ("its scores are also used to judge stability") |
| Antibody-specific flags | **TNP** (Therapeutic Nanobody Profiler) and **TAP**, developability risk vs clinical-stage therapeutics |
| Honest rating | the guide rates computational substitutes for melting temperature as **"Moderate"** reliability |

What we used, and what it buys:

- **ESM-IF1, hosted on Nebius** (`esm-if1`, ~4 min). Nothing to install. `10b_stability`
  scores each design's mean log-likelihood given the backbone with the antigen stripped,
  and compares it to trastuzumab's own score of **−1.0571**, measured by the same model
  in the same run.
- **ThermoMPNN** is open source and would need installing.
- A sanity bound from the guide: most single mutations fall between **−1 and +2
  kcal/mol**, so a ΔΔG far outside that is the model reporting its own failure, not a
  discovery.

> **Be honest about what this is.** Both tools are proxies — ThermoMPNN predicts folding
> ΔΔG, ESM-IF gives a sequence likelihood. **Neither predicts T<sub>m</sub>**, which is
> what the wet lab measures. ThermoMPNN is also built for *single* mutations, while these
> designs change up to 10 CDRH3 positions at once, which is exactly where it is least
> trustworthy. Treat the output as a **rank, never a predicted temperature** — and say so
> before a reviewer finds it.

---

## JapanFold

Hosted folding, billed at $0.26 per processor-hour with queue time free. At the
per-target costs below, a $1,000 credit is effectively unlimited — **do not ration jobs.**

```bash
set -a; . /path/to/japanfold.env; set +a
curl -s -H "Authorization: Bearer $JAPANFOLD_API_KEY" $JAPANFOLD_BASE_URL/v1/models
curl -s -H "Authorization: Bearer $JAPANFOLD_API_KEY" $JAPANFOLD_BASE_URL/v1/account
```

| Model | Why it matters | $ / target |
| --- | --- | ---: |
| `af2ig` | **AF2-IG.** Scores a binder you already designed, reports ipTM and interface PAE. No MSA, no sampling, so the same input always gives the same answer | 0.0004 |
| `opendde-abag` | **OpenDDE antibody-antigen checkpoint**, Protenix-v2 stack. Purpose-built for Fab + antigen | 0.006 |
| `rf3` | **RoseTTAFold3**, a third independent lineage for measuring spread | 0.007 |
| `boltz2`, `openfold3`, `esmfold2` | The same models a Foundry VM has, useful for parallelising off a single GPU | 0.002–0.003 |

**AF2-IG's determinism is the useful property.** Every other scorer samples, so repeat
runs disagree with themselves. A scorer with no sampling noise removes one term from the
stopping rule's spread and makes the remaining disagreement mean something.

---

## Folding models, and which metric to rank on

Prefer a model that takes a template for binders — it sees the known HER2 structure and
gives a fairer check: Boltz-2, OpenFold3, Protenix v2, Mol-1. Use two independent models
and treat their spread as the noise floor; **that spread is the stopping rule's threshold.**

Apheris OpenFold3 is one of the two in `fold_models`. It was a stated condition of the
hackathon's monetary prize, and we wanted a second independent model anyway, so it cost
nothing to satisfy.

> **Prefer ipSAE over bare ipTM.** JapanFold's `write_pae` option returns the full PAE
> matrix and documents it as *"the one Adaptyv's ipSAE pipeline reads"*. The Smorodina
> calibration set uses ipSAE as well. Pass `write_pae: true` on every scoring run or the
> matrix is not returned.
>
> In the delivered run, `08_calibrate` works from bare ipTM and finds **AUC 0.645** — so
> weak that it computes the cutoff and refuses to filter on it. Moving that calibration
> to ipSAE is the single highest-value open item in the pipeline.

**Three traps, all paid for:**

- **OpenFold2 never comes up** — it fails after ~55 min whether loaded or idle. Use
  OpenFold3 or Boltz-2.
- **Do not rank on CDRH3 refold agreement.** Protenix does not reproduce trastuzumab's
  own CDRH3 loop, so disagreement measures the model, not the design.
- **ipTM above 0.95 is not evidence of binding.** The Smorodina non-binders score that
  well too. See [finding 2](../README.md#2--iptm-cannot-separate-binders-from-non-binders-here).

---

## Known sharp edges in the code

Read these before changing the corresponding knob.

- **`novelty.scan` uses three pigeonhole blocks,** which makes it exact only up to a
  threshold of 3. Raising `min_dist_to_provided` to 4 would **silently pass designs at
  true distance 3**. Add an assertion before anyone tunes that knob.
- **The affinity surrogate is validated at distance 4 but applied at distance 3.** The
  production model does train on distance-3 data, so this is a conservative estimate
  rather than a leak — but it is worth one sentence before anyone asks.
- **`10_express` is thinner than it looks.** It runs abguard's liability report only.
  Aggrescan 3D, SoluProt and NetSolP are external web services that were not enabled, and
  ProteinMPNN's soluble weights were unavailable. A design that does not express scores
  zero on all three axes, so this is a real gap rather than a cosmetic one.
- **The de novo graft costs something measurable.** `06b_denovo` redesigns the framework
  with ProteinMPNN and grafts trastuzumab's CDRs back in afterwards, but ProteinMPNN chose
  those framework residues *conditioned on the CDRs it also chose*. Delivered de novo
  designs scored a mean of −0.052 against the baseline on inverse-folding likelihood,
  against −0.021 for the optimisation designs. `06c_antifold` is the fix: AntiFold takes a
  `regions` argument, so the framework is redesigned with the CDRs **held fixed during
  design** rather than restored after it.

---

## The environment we ran on

Recorded for reproducibility; the VM itself no longer exists.

| What | Detail |
| --- | --- |
| Hardware | 1× H100, 16 vCPU, 196 GB RAM, 484 GB disk |
| Apheris Foundry | 20 Docker images, ~200 GB: RFD4-Proteina ×3 weights, Boltz-1/2, OpenFold3 ×2, Protenix v1/v2, Mol-1, ESMFold2, MSA. Run via `docker run`, no Hub |
| GPU access | `--gpus all` reaches the H100 from inside a container |
| CUDA | toolkit 13.0 |
| Tooling | `foundry-toolkit`, `aws`, `nextflow`, `zstd`, `jq`, `git`, `uv` 0.13.7 |
| Data | `s3://apheris-oslo-hackathon`, read-only. HER2 tables (6 MB) sync to `data/raw/her2-trastuzumab/` — the **git-ignored** path, not `data/` as the vendor docs suggest |

Two environment gotchas worth keeping:

- **`~/.local/bin` is on `PATH` only in a login shell.** Over SSH use
  `ssh ... bash -lc '...'`, or tools installed there will look missing when they are not.
- **Credentials stay out of the repo.** `*.private.json` and `data/raw/` are git-ignored.
  Point tools at the credentials file with an environment variable; never paste a key into
  a prompt. **This repo is public.**

### The smoke test that set our direction

The organisers' own 4-design example — generate → prepare top 2 → Boltz-2 refold → score,
245 s cold. Per-step times in [notes/RUNTIMES.md](../notes/RUNTIMES.md).

| Design | IMGT CDRH3 | vs trastuzumab | vs prior art | ipTM | min_ipae | binder pLDDT | abguard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `her2_cdrh3_2` | `SRWGPPYFYAMDY` | 3 | **2** | 0.946 | 0.331 | 95.6 | **rejected** |
| `her2_cdrh3_3` | `SRWASSSEYYFDY` | 7 | ≥ 3 | 0.965 | 0.310 | 98.5 | passes |

**Half the first designs were rejected, and not by the obvious rule.** Design 2 cleared
trastuzumab at distance 3, then landed 2 mutations from `SRWGPYGFYAMDY` in the FACS
screen. **Prior art is the binding constraint, not trastuzumab** — with 610,109 reference
loops it stays that way, and any ranking has to run the gate *first* or it ranks designs
that cannot be submitted. That single observation is why `04a_novelty` sits where it does
in the funnel.

It also showed that **the design model does not know the novelty rule.** It optimises
predicted fit, and the HER2 weights were trained on the very screen the designs must stay
away from. Expect a real rejection rate and generate with margin.
