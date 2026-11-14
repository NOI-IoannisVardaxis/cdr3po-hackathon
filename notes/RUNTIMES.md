# Runtimes — measured on the team VM

One H100, 16 vCPU, 196 GB RAM. These numbers decide how wide the funnel can be.

Everything below is wall-clock on our own machine, not a vendor estimate. Replace a row the
moment you measure it; do not plan against the blank ones.

## Measured, 14 Nov 09:55

The organisers' HER2 example, `bash Scripts/run-her2.sh all` in `~/rfd4`, cold start
(no container running, weights not in page cache). Input `Data/12-her2/her2_cdrh3.yaml`,
weight `her2-novel-2026-11-09`, Boltz-2 for the refold.

| Step | Scale | Time |
| --- | --- | --- |
| `design` | 4 designs, `--recycle-strategy high/front_loaded` | 164 s |
| `refold-prepare` | top 2 | 3 s |
| Boltz-2 `predict` | 2 structures, 1 diffusion sample, templated | 74 s |
| `refold-score` | 2 designs | 4 s |
| **Total** | | **245 s** |

Cold start cost nothing noticeable: the whole run matched the documented ~4 minutes.

### What this implies for the day

Design is the long pole, and it is the only step that scales with the number of designs.
At face value 4 designs in 164 s is ~41 s each, but that includes container and weight
load, so the marginal cost is lower — **measure 100 designs before planning around it**.
Boltz-2 is ~37 s per structure and parallelises across designs, not within one.

A rough plan to beat, assuming the per-design cost holds:

| Batch | Design | Refold (Boltz-2) | Total |
| --- | --- | --- | --- |
| 100 designs, refold top 20 | ~45 min | ~12 min | ~1 h |
| 300 designs, refold top 50 | ~2 h | ~30 min | ~2.5 h |

Designs freeze at 14:00. Start the big generation run early and do the filtering work
while it runs.

## Still to measure

| Step | Tool | Time | Notes |
| --- | --- | --- | --- |
| `design` at 100 replicates | RFD4-Proteina | | the number that sets the funnel width |
| `design` with `--compile true` | RFD4-Proteina | | docs claim faster from ~8 designs per length |
| Second folding model on the same designs | OpenFold3 / Protenix v2 | | needed for the two-model spread |
| Numbering + liability scan | abguard / immunum | | not installed yet |
| Affinity model train + predict | ours | | |
| Thermostability scoring | undecided | | the least-served axis |
| One AMASS query to a citable result | AMASS | | |

## Platform notes worth remembering

- **Nebius first calls are slow**, 30 s to 26 min depending on the model, because the model
  loads on demand. Send long jobs early and keep the run id.
- **Do not use OpenFold2** — it never gets an endpoint up and fails after ~55 minutes.
- Boltz-2, OpenFold3, Protenix v2 and Mol-1 accept a template, which gives binders a fairer
  check than ESMFold2 folding the target unguided.
