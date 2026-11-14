# The pipeline

A modular design pipeline for the HER2 challenge. Every step does one job, reads
`results/designs.parquet`, adds the columns it owns and writes it back.

**Steps never delete rows.** A design that fails a filter gets `dropped_by` set
to the step that rejected it and stays in the table. Three things follow: the
funnel counts are a query rather than a reconstruction, any step can be re-run
without breaking the ones after it, and you can always see which step killed a
design.

---

## Install and verify

```bash
make dev                       # virtualenv and dependencies via uv
uv run her2 steps              # list the steps
uv run her2 config             # print every threshold in force
```

Two things the pipeline needs that are not in this repo:

```bash
# 1. The organisers' tables (6 MB, git-ignored)
aws s3 sync s3://apheris-oslo-hackathon/data/her2-trastuzumab/ \
            data/raw/her2-trastuzumab/

# 2. abguard, from the wheel in the shared uv cache
uv venv --python 3.12 .venv-abguard
VIRTUAL_ENV=$PWD/.venv-abguard uv pip install <path-to>/abguard-0.1.0-py3-none-any.whl

# Verify it is the build WITH the novelty gate, not the one without
./.venv-abguard/bin/abguard --help | grep novelty-report   # must print
```

That last check is not optional. Two wheels on the hackathon VM both call
themselves `0.1.0`; the older one validates the hard filters, reports every
design compliant and never mentions that it skipped prior art. `s02_guard`
refuses to run against it.

---

## Run it

```bash
# The optimisation path, end to end
uv run her2 run --path optimisation

# A smaller run, to see it work in under a minute
uv run her2 run --path optimisation --edit-distances 3 --shortlist-n 40

# One step at a time
uv run her2 step 01_baseline
uv run her2 step 03a_enumerate
uv run her2 step 04a_novelty
```

Every run is deterministic under `--seed` (default 1). Re-running with the same
config reproduces the same designs; if it does not, that is a bug, not noise.

---

## Steps

| Step | Job | Tools | Filter |
| --- | --- | --- | --- |
| `01_baseline` | Trastuzumab's KD, CDR3–HER2 contact map, disulfide site scan | gemmi, SPR table | none, reference only |
| `02_guard` | abguard verdict on every construct | abguard `team-validate` | format, length, cysteines, numbering, novelty |
| `03a_enumerate` | Every CDR3 at 3 and 4 mutations across IMGT 3–12 | numpy, itertools | constraint, not filter |
| `04a_novelty` | Prior-art rejection | numpy pigeonhole scan | Hamming ≥ 3 from all 525,063 |
| `05a_affinity` | FACS surrogate scores and shortlists | scikit-learn | ranking cut, nothing disqualified |
| `06a_optimisation` | Loops become paired constructs with framework packages | gemmi geometry | length ≤ 250 |
| `03b_rfd4` | De novo branch: RFD4-Proteina designs against HER2 (GPU) | Apheris RFD4 image | ≥ 2 CDRH3 mutations from trastuzumab |
| `07_fold` | Co-fold with two Apheris models, measure their spread (GPU) | Boltz-2, OpenFold3 | none — adds scores |
| `08_calibrate` | ipTM cutoff from 454 measured non-binders | numpy | admits ≤ target FPR, by construction |
| `09_epitope` | Contact recovery vs 1N8Z; disulfide SG–SG distance | gemmi | recovery ≥ threshold at the stated cutoff |
| `10_express` | Developability flags | abguard `liability-report` | flags, never drops |
| `11_rank` | Predicted tier, then hierarchical clustering | scipy linkage | one design per cluster |
| `12_stop` | Cut where gain falls below the model spread | measured spread | the stopping rule |
| `13_submit` | Write and validate the deliverable | abguard | must return compliant |
| `14_report` | Figures and `report.html` for the pitch | matplotlib | none |

### Paths

```bash
uv run her2 run --path optimisation   # CPU only, no GPU, no network
uv run her2 run --path denovo         # RFD4 branch, needs the GPU
uv run her2 run --path full           # everything, including figures
```

Run `denovo` in the background while `optimisation` runs on CPU — the GPU is
otherwise idle through steps 3a–6a.

---

## Outputs

| File | For | Contents |
| --- | --- | --- |
| `results/submission.csv` | the organisers | 20 ranked rows, the five columns abguard accepts |
| `results/submission_annotated.csv` | the team | the same 20 with track, mutations and reasons |
| `results/shortlist.csv` | the judges | every scored design, with `dropped_by` naming its rejecting step |
| `results/funnel.json` | the pitch | counts, thresholds and timings per step |
| `results/baseline.json` | everything | trastuzumab's reference numbers |
| `results/designs.parquet` | the pipeline | the full table |

---

## What the thresholds mean

Everything that can change a verdict is in `config.py` and is written into
`funnel.json` alongside the counts it produced.

| Parameter | Default | Why |
| --- | --- | --- |
| `edit_distances` | `(3, 4)` | Distance 3 is 90.0% high binders but only 9.3% clears prior art; distance 4 is 74.6% and 50–70%. Both hedge an abguard rule change. |
| `min_dist_to_provided` | `3` | abguard reports `required_cdrh3_mutations = 3`, verified by probe. Not 4, despite the task PDF's wording. |
| `contact_cutoff_a` | `5.0` | The contact set sits on a cliff: 4.0 Å gives 2 positions, 4.5–5.0 Å gives 4, 6.0 Å gives 6. 5.0 is the stable side. |
| `disulfide_cb_min/max` | `3.5 / 4.5` | Standard geometric window. The textbook VH44–VL100 pair measures 4.38 Å here, at the edge. |
| `max_ddg_package` | `-1.0` | ΔΔG predictors carry ~1 kcal/mol RMSE, so demand more than the error bar. |
| `n_optimisation_ranks` | `12` | Only the top N is tested and N depends on pitch placement, so de novo fills ranks 13–20. |

---

## Tests

```bash
make test        # pytest, including doctests
make lint        # mypy, ruff, black, isort
```

Doctests live on the pure functions in `sequences.py`, `novelty.py` and
`table.py` — the places where an off-by-one silently changes which designs are
legal. Three regression tests encode failures already seen in practice:
trastuzumab itself must fail novelty, a 3-mutation design must clear the
mutation floor, and a construct must come out at 227 aa.
