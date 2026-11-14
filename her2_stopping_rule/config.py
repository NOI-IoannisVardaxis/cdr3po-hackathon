"""Configuration for the HER2 design pipeline.

Every value that can change a verdict lives here, never in a step. The config is
written into ``results/funnel.json`` alongside the counts it produced, so a run
can always be reproduced from its own output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

# --- Trastuzumab reference sequences -------------------------------------------------
# Taken from row 1 of trastuzumab_her2_spr_all_768.csv (KD 0.152 nM).

TRASTUZUMAB_VH: Final[str] = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
    "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL: Final[str] = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSR"
    "SGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)

#: The IMGT CDR3, 13 residues. This is what abguard extracts and compares.
CDR3_IMGT: Final[str] = "SRWGGDGFYAMDY"

#: Flanks of the CDR3 inside the heavy chain, used to splice a new loop in.
VH_PREFIX: Final[str] = TRASTUZUMAB_VH[: TRASTUZUMAB_VH.index(CDR3_IMGT)]
VH_SUFFIX: Final[str] = TRASTUZUMAB_VH[
    TRASTUZUMAB_VH.index(CDR3_IMGT) + len(CDR3_IMGT) :
]

#: IMGT positions that vary. S1, R2 and Y13 are constant in every provided
#: sequence, so only positions 3-12 (0-based 2..11) are ever mutated.
MUTABLE_IMGT: Final[tuple[int, ...]] = tuple(range(3, 13))

#: The 20 proteinogenic amino acids, in a fixed order so one-hot encodings are stable.
AMINO_ACIDS: Final[str] = "ACDEFGHIKLMNPQRSTVWY"

#: Trastuzumab's measured affinity, the baseline every design is compared against.
BASELINE_KD_NM: Final[float] = 0.152


@dataclass(frozen=True)
class Config:
    """Thresholds and paths for one pipeline run.

    Attributes are grouped by the step that consumes them. Nothing here is read
    from the environment: a run is defined entirely by this object plus the input
    data, which is what makes the pipeline reproducible.

    Attributes:
        repo_root: Directory holding ``data/`` and ``results/``.
        seed: Seed for every random choice in the pipeline.
        edit_distances: CDR3 mutation counts to enumerate, in preference order.
            Distance 3 carries the best affinity prior (90.0% high binders) but
            only 9.3% clears prior art; distance 4 supplies the volume.
        min_dist_to_provided: Minimum Hamming distance from every provided CDR3.
            abguard reports ``required_cdrh3_mutations = 3``; verified empirically.
        min_cdrh3_mutations_denovo: The de novo track's weaker rule, "at least 2
            CDRH3 mutations away from trastuzumab" (task PDF, page 2).
        shortlist_n: Loops kept after affinity ranking.
        n_clusters: Loop clusters to draw finalists from.
        packages_per_loop: Stability packages built per surviving loop.
        contact_cutoff_a: Heavy-atom distance defining a CDR3-HER2 contact. The
            contact set is cutoff-dependent: 4.0 A gives 2 positions, 4.5-5.0 A
            gives 4, 6.0 A gives 6. 5.0 A is the stable side of that cliff.
        min_epitope_recovery: Fraction of native HER2 contacts a refold must keep.
        disulfide_cb_min: Lower bound on C-beta to C-beta distance for a designed
            disulfide site.
        disulfide_cb_max: Upper bound on the same.
        disulfide_sg_max: Maximum SG-SG distance in the refold for the disulfide
            to count as formed. Beyond this the stability rationale is void.
        max_ddg_single: Worst predicted ddG tolerated for any single mutation,
            in kcal/mol. Positive is destabilising.
        max_ddg_package: Predicted ddG the whole package must beat, in kcal/mol.
            Predictors carry ~1 kcal/mol RMSE, so demand more than the error bar.
        iptm_fpr: False-positive rate on measured non-binders used to set the
            ipTM cutoff, instead of picking a round number.
        n_submit: Designs in the final ranked list.
        n_optimisation_ranks: How many of those ranks optimisation designs take.
            De novo fills the remainder, so it is tested only if the pitch wins
            extra slots.
    """

    repo_root: Path = Path(__file__).resolve().parent.parent
    seed: int = 1

    # step 3a enumerate
    edit_distances: tuple[int, ...] = (3, 4)

    # step 4 novelty
    min_dist_to_provided: int = 3
    min_cdrh3_mutations_denovo: int = 2

    # step 5a affinity
    shortlist_n: int = 400
    affinity_max_iter: int = 300
    affinity_learning_rate: float = 0.1
    affinity_max_leaf_nodes: int = 63

    # step 6a stability
    #: Only the native framework survives abguard. An engineered VH-VL
    #: disulfide adds one cysteine to each chain, and both chains already
    #: carry two, so the count goes odd and abguard's per-chain parity check
    #: reads an inter-chain disulfide as two unpaired cysteines. Verified:
    #: all 48 disulfide constructs were rejected, all 48 native ones were not.
    packages_per_loop: int = 1
    disulfide_cb_min: float = 3.5
    disulfide_cb_max: float = 4.5
    disulfide_sg_max: float = 2.5
    max_ddg_single: float = 0.5
    max_ddg_package: float = -1.0

    # step 7 fold
    fold_models: tuple[str, ...] = ("boltz2", "openfold3")
    n_finalists: int = 96

    # step 8 calibrate
    iptm_fpr: float = 0.05
    #: Minimum separation before ipTM is allowed to drop designs. Measured on
    #: the Smorodina complexes, Boltz-2 scores 8,190 non-cognate pairs at mean
    #: 0.860 against 0.896 for 91 cognate ones -- 94% of measured non-binders
    #: clear 0.70. Filtering on a metric that weak is noise dressed as rigour.
    min_iptm_auc: float = 0.70

    # step 9 epitope
    contact_cutoff_a: float = 5.0
    #: Fraction of the *baseline's own* recovered epitope a design must match.
    #: The threshold is derived, not chosen: trastuzumab is folded by the same
    #: model and we measure how much of its native epitope that model even
    #: reproduces. Demanding more of a design than the model manages on the
    #: molecule the epitope was defined from would be meaningless.
    epitope_recovery_fraction: float = 0.9
    #: Fallback only, used when no baseline fold exists to calibrate against.
    #: Any run that falls back records that it did.
    min_epitope_recovery: float = 0.75
    #: Fold without the 1N8Z template, which is both necessary and preferable.
    #: Necessary: the raw cropped CIF is rejected by the template parser
    #: ("errors resolving residue names"), and the cleaned template the
    #: organisers' smoke test used is written by refold-prepare, which only
    #: runs on RFD4 design output, not on our enumerated constructs.
    #: Preferable: a templated fold is told where trastuzumab binds, so
    #: recovering that epitope would be partly an echo of the input. Verified
    #: on the baseline, which co-folds to ipTM 0.952 unprompted.
    fold_without_template: bool = True

    # step 11 rank
    n_clusters: int = 24

    # step 13 submit
    #: Team name. The organisers ask for the submission file to be named
    #: after the team, e.g. cdr3po.csv.
    team_name: str = "TEAM"
    n_submit: int = 20
    n_optimisation_ranks: int = 12

    # paths, derived
    _rel_paths: dict[str, str] = field(
        default_factory=lambda: {
            "facs": "data/raw/her2-trastuzumab/tz_her2_facs_3classes_524346.csv.zip",
            "spr": "data/raw/her2-trastuzumab/trastuzumab_her2_spr_all_768.csv",
            "structure": "../../rfd4/Data/12-her2/her2_fv_1n8z.cif",
            "abguard": ".venv-abguard/bin/abguard",
            "results": "results",
        },
        repr=False,
    )

    def path(self, key: str) -> Path:
        """Resolve a named path against the repository root.

        Args:
            key: One of ``facs``, ``spr``, ``structure``, ``abguard``, ``results``.

        Returns:
            The resolved absolute path.

        Raises:
            KeyError: If the key is not a known path.
        """
        return (self.repo_root / self._rel_paths[key]).resolve()

    @property
    def results_dir(self) -> Path:
        """Directory for all pipeline outputs, created if missing."""
        d = self.path("results")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config for the funnel record.

        Returns:
            A JSON-safe dict with paths rendered as strings.
        """
        raw = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}
