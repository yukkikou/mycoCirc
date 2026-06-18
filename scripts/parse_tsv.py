#!/usr/bin/env python3
"""
TSV validation script for PanCirc-Fungi.

Usage:
    python scripts/parse_tsv.py [--tsv all_lib_model_full.tsv]

Parses the strain registry, validates file paths, reports summary.
"""

import argparse
import sys
import warnings

# Add project root to path
sys.path.insert(0, ".")

from data.tsv_parser import parse_strain_registry, print_tsv_summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate and summarize the PanCirc-Fungi strain registry TSV"
    )
    parser.add_argument(
        "--tsv",
        default="all_lib_model_full.tsv",
        help="Path to the strain registry TSV (default: all_lib_model_full.tsv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any paths are missing",
    )
    args = parser.parse_args()

    print(f"Parsing: {args.tsv}\n")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        entries = parse_strain_registry(args.tsv)

        # Print warnings
        for warning in w:
            print(f"  ⚠ {warning.message}")

    # Validate paths
    all_ok = True
    for e in entries:
        missing = e.validate_paths()
        if missing:
            all_ok = False
            print(f"\n  ✗ {e.strain} ({e.species}): missing {len(missing)} file(s)")
            for m in missing:
                print(f"      {m}")

    # Print summary
    print_tsv_summary(entries)

    if args.strict and not all_ok:
        print("\n  ERROR: Missing files detected (--strict mode)")
        sys.exit(1)
    else:
        print("\n  Done.")


if __name__ == "__main__":
    main()
