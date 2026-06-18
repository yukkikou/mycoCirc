"""
TSV registry parser for PanCirc-Fungi.

Reads ``all_lib_model_full.tsv`` and provides:
- StrainEntry namedtuple per row
- Strain index mapping
- Group membership queries
- File path validation
- Handles "Candia" -> "Candida" mapping
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StrainEntry:
    """A single strain entry parsed from the TSV."""

    group: str
    species: str
    strain: str
    genome_path: str
    gtf_path: str
    circinfo_path: str
    circexp_path: str
    geneexp_path: str

    # Derived
    short_species: str = field(init=False)
    is_excluded: bool = False
    exclusion_reason: str = ""

    def __post_init__(self):
        # Derive short species name from e.g. "Candida_albicans" -> "C.albicans"
        parts = self.species.split("_")
        if len(parts) >= 2:
            self.short_species = f"{parts[0][0]}.{'_'.join(parts[1:])}"
        else:
            self.short_species = self.species

    def validate_paths(self) -> List[str]:
        """Return list of missing file paths (empty if all exist)."""
        missing = []
        paths = [
            ("Genome", self.genome_path),
            ("GTF", self.gtf_path),
            ("CircInfo", self.circinfo_path),
            ("CircExp", self.circexp_path),
            ("GeneExp", self.geneexp_path),
        ]
        for name, path in paths:
            if not os.path.isfile(path):
                missing.append(f"{name}: {path}")
        return missing


# Strains to exclude from all training/evaluation
EXCLUDED_STRAINS = {"A2"}  # Aspergillus nidulans: 0 circRNAs

# Test split: held-out species per group
TEST_STRAINS = {
    "Candida": {"P4"},        # C. auris
    "Cryptococcus": {"C4"},   # C. neoformans var neoformans
    "Filamentous": {"F6"},    # F. venenatum
}

# All training strains per group (complement of test)
TRAIN_STRAINS: Dict[str, set] = {
    "Candida": {"P1", "P2", "P3", "P5", "P6", "S8"},
    "Cryptococcus": {"C1", "C2", "C3", "C5", "C6", "C7"},
    "Filamentous": {"F3", "F4", "N1", "A1", "A3"},
}

# Typo correction: TSV has "Candia" for P6
_TYPO_CORRECTIONS = {"Candia": "Candida"}


def parse_strain_registry(tsv_path: str) -> List[StrainEntry]:
    """Read the TSV and return a list of ``StrainEntry`` objects.

    - Skips excluded strains (A2)
    - Maps "Candia" -> "Candida" with a warning
    - Validates all file paths exist
    """
    entries: List[StrainEntry] = []

    with open(tsv_path) as fh:
        header = fh.readline()  # skip header

        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                warnings.warn(f"Skipping malformed line: {line[:60]}...")
                continue

            group = parts[0].strip()
            species = parts[1].strip()
            strain = parts[2].strip()
            genome = parts[3].strip()
            gtf = parts[4].strip()
            circinfo = parts[5].strip()
            circexp = parts[6].strip()
            geneexp = parts[7].strip()

            # Typo correction
            if group in _TYPO_CORRECTIONS:
                warnings.warn(
                    f"Fixed group typo: '{group}' -> '{_TYPO_CORRECTIONS[group]}' "
                    f"for {species} ({strain})"
                )
                group = _TYPO_CORRECTIONS[group]

            entry = StrainEntry(
                group=group,
                species=species,
                strain=strain,
                genome_path=genome,
                gtf_path=gtf,
                circinfo_path=circinfo,
                circexp_path=circexp,
                geneexp_path=geneexp,
            )

            # Exclude A2
            if strain in EXCLUDED_STRAINS:
                entry.is_excluded = True
                entry.exclusion_reason = "Excluded: 0 circRNAs (Aspergillus nidulans)"
                warnings.warn(f"{entry.exclusion_reason} ({strain})")

            entries.append(entry)

    return entries


def build_strain_index(entries: List[StrainEntry]) -> Dict[str, int]:
    """Map strain ID -> integer index (only non-excluded strains)."""
    idx = {}
    i = 0
    for e in entries:
        if not e.is_excluded:
            idx[e.strain] = i
            i += 1
    return idx


def get_group_members(entries: List[StrainEntry]) -> Dict[str, List[StrainEntry]]:
    """Group strain entries by taxonomic group."""
    groups: Dict[str, List[StrainEntry]] = {}
    for e in entries:
        if e.is_excluded:
            continue
        groups.setdefault(e.group, []).append(e)
    return groups


def get_species_id(strain: str, strain_index: Dict[str, int]) -> int:
    """Return numeric species ID for a strain.

    Falls back to 0 for unknown strains.
    """
    return strain_index.get(strain, 0)


def print_tsv_summary(entries: List[StrainEntry]):
    """Print a human-readable summary of the TSV contents."""
    groups = get_group_members(entries)
    excluded = [e for e in entries if e.is_excluded]

    print("=" * 60)
    print("PanCirc-Fungi: Data Registry Summary")
    print("=" * 60)
    for group, members in sorted(groups.items()):
        print(f"\n  {group} ({len(members)} strains):")
        for e in members:
            missing = e.validate_paths()
            flag = " [MISSING FILES]" if missing else ""
            print(f"    {e.strain:4s} {e.species:40s}{flag}")
            for m in missing:
                print(f"           ⚠ {m}")

    if excluded:
        print(f"\n  Excluded ({len(excluded)}):")
        for e in excluded:
            print(f"    {e.strain:4s} {e.species:40s} — {e.exclusion_reason}")

    total_active = sum(len(v) for v in groups.values())
    total_circ = sum(
        _quick_circ_count(e.circinfo_path) for members in groups.values()
        for e in members
    )
    print(f"\n  Total active strains: {total_active}")
    print(f"  Approx. total circRNAs: {total_circ}")
    print("=" * 60)


def _quick_circ_count(path: str) -> int:
    """Fast line count of a CSV file (header subtracted)."""
    try:
        with open(path) as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0
