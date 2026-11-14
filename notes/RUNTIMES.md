# Runtimes — measured during rehearsal, 13 Nov

The only artefact we carry into the hackathon. These numbers decide how wide
the funnel can be tomorrow.

| Step                                    | Tool              | Time | Notes |
|-----------------------------------------|-------------------|------|-------|
| Numbering a VHH                         | ANARCII           |      |       |
| Numbering an scFv / Fab                 | ANARCII           |      |       |
| Liability / motif scan                  | local             |      |       |
| Fold one binder                         | Apheris OpenFold  |      |       |
| Co-fold one complex                     | JapanFold         |      |       |
| RFD4-Proteina batch (how many designs?) | JapanFold         |      |       |
| One AMASS query -> citable result       | AMASS             |      |       |

The RFD4-Proteina batch time is the critical one: it sets how many candidates
we can afford to generate before the 14:00 freeze.

## Open questions for the organizers
- [ ] Is the target antigen disclosed before the 09:30 brief?
- [ ] How are Apheris Foundry + Claude Max credits granted?
- [ ] Can we have the agent-skill setup docs in advance?  <-- saves an hour tomorrow
- [ ] Is Apheris OpenFold agent-callable, or UI only?     <-- the prize depends on using it

## Contig recipes (guide's practice case, PDB 1ZVY)
Chain A = VHH, chain B = hen egg-white lysozyme. Residue A1 is unresolved.

  Option A, new binder vs chain B:  B1-129, /0, 110-130
  Option B, redesign the VHH loop:  B1-129, /0, A2-96, 20, A117-125

## REMEMBER
Delete all rehearsal glue code before 14 Nov.
The repo is created tomorrow, on the day. Not before.
