#!/usr/bin/env python3
"""A5 (revision): empirical basis for the mycoCircAtlas "high-confidence"
(p>=0.8) designation + count reconciliation.

Responds to: R1-M4 + R3-2 ("319,860 high-confidence predictions ... empirical
basis and expected false-discovery rate of this designation need to be
demonstrated").

Parts:
 1. Reconcile the manuscript count (319,860) against the current database:
    mycoCircAtlas/predictions/ (p>=0.8 filtered) and species_circ_summary.tsv.
 2. Per-species atlas statistics: n genes total, n p>=0.8, fraction — from
    predictions_all/ row counts (exact, cheap: wc-equivalent).
 3. Empirical precision evidence on external species (from a4 outputs):
    precision at p>=0.8 vs detection stringency, with the explicit caveat that
    "undetected" is not "false" (rRNA-seq detection is condition- and
    depth-dependent).

Outputs (revision/results/):
  a5_atlas_count_reconciliation.csv
  a5_atlas_per_species_stats.csv   (768 rows)
  a5_high_confidence_evidence.md

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a5_atlas_precision.py
"""

from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "revision" / "results"
ATLAS = PROJ / "mycoCircAtlas"

MANUSCRIPT_COUNT = 319_860


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    # ── 1. count reconciliation ───────────────────────────────────────────
    pred_dir = ATLAS / "predictions"
    files = sorted(pred_dir.glob("*.tsv"))
    per_file = {f.stem: count_lines(f) - 1 for f in files}  # minus header
    total_now = sum(per_file.values())

    summary_path = ATLAS / "species_circ_summary.tsv"
    sum_df = pd.read_csv(summary_path, sep="\t")
    sum_total = int(sum_df["circRNA_p0.8"].sum())
    sum_all = int(sum_df["circRNA_all"].sum())

    recon = pd.DataFrame([
        dict(item="manuscript_stated_count", value=MANUSCRIPT_COUNT),
        dict(item="predictions_dir_current", value=total_now),
        dict(item="species_circ_summary_p0.8", value=sum_total),
        dict(item="species_circ_summary_all", value=sum_all),
        dict(item="n_species_predictions_dir", value=len(per_file)),
        dict(item="n_species_summary", value=len(sum_df)),
        dict(item="delta_vs_manuscript", value=total_now - MANUSCRIPT_COUNT),
    ])
    recon.to_csv(OUT / "a5_atlas_count_reconciliation.csv", index=False)
    print(recon.to_string(index=False))

    # ── 2. per-species stats from predictions_all ─────────────────────────
    all_dir = ATLAS / "predictions_all"
    rows = []
    for f in sorted(all_dir.glob("*.tsv")):
        n_all = count_lines(f) - 1
        n_hc = per_file.get(f.stem, 0)
        rows.append(dict(species=f.stem, n_genes=n_all,
                         n_p_ge_08=n_hc,
                         frac_p_ge_08=round(n_hc / n_all, 5) if n_all else 0.0))
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "a5_atlas_per_species_stats.csv", index=False)
    print(f"\n768-species stats: median p>=0.8 fraction = "
          f"{stats['frac_p_ge_08'].median():.4f}, "
          f"mean = {stats['frac_p_ge_08'].mean():.4f}")

    # ── 3. empirical precision evidence (from A4 outputs) ────────────────
    lines = [
        "# A5 — empirical basis for the p>=0.8 \"high-confidence\" atlas tier",
        "",
        "## Atlas count reconciliation",
        f"- Manuscript states **{MANUSCRIPT_COUNT:,}** predictions.",
        f"- Current `mycoCircAtlas/predictions/` holds **{total_now:,}** rows "
        f"across {len(per_file)} species; `species_circ_summary.tsv` sums to "
        f"**{sum_total:,}**.",
        f"- Delta vs manuscript: {total_now - MANUSCRIPT_COUNT:+,} rows "
        "(README/DB drift — the manuscript number must be refreshed or the DB "
        "re-frozen to the submission version).",
        "",
        "## Empirical precision of the p>=0.8 tier (external species)",
        "",
        "Precision estimated against multi-tool RNA-seq detection ground "
        "truth at increasing stringency. NOTE: undetected does NOT equal "
        "false — detection depends on condition, depth and life stage; these "
        "values are therefore a LOWER BOUND on precision.",
        "",
    ]
    a4 = OUT / "a4_threshold_metrics.csv"
    if a4.exists():
        df = pd.read_csv(a4)
        hl = df[(df.thr == 0.8)]
        lines += ["| species | ckpt | GT stringency | precision | "
                  "enrichment vs prevalence | sensitivity |",
                  "|---|---|---|---|---|---|"]
        for _, r in hl.iterrows():
            if r.precision == "" or pd.isna(r.precision):
                continue
            lines.append(
                f"| {r.species} | {r.ckpt} | {r.stringency} | {r.precision} "
                f"| {r.enrichment_vs_prev}x | {r.sensitivity} |")
    else:
        lines.append("(run a4_threshold_calibration.py first — evidence "
                     "table pending)")
    (OUT / "a5_high_confidence_evidence.md").write_text("\n".join(lines) + "\n")
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
