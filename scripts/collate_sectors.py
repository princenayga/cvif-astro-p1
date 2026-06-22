#!/usr/bin/env python3
"""
collate_sectors.py — Copy sector light curve images from all planets into
one folder per sector for easy side-by-side inspection.

Source:  results/{planet}/sector_s{sec:03d}.png
Dest:    sector_{sec:02d}/{planet}_s{sec:03d}.png

Usage:
    python collate_sectors.py                          # default sectors
    python collate_sectors.py --sectors 41 75 40 74 81
"""

import argparse
import os
import shutil

RESULTS_DIR    = "results"
DEFAULT_SECTORS = [41, 75, 40, 74, 81]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sectors", type=int, nargs="+", default=DEFAULT_SECTORS)
    args = parser.parse_args()

    planet_dirs = [
        d for d in sorted(os.listdir(RESULTS_DIR))
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d != "survey_summary.csv"
    ]

    for sec in args.sectors:
        out_dir = f"sector_{sec:02d}"
        os.makedirs(out_dir, exist_ok=True)
        copied = 0
        missing = []

        for planet in planet_dirs:
            src = os.path.join(RESULTS_DIR, planet, f"sector_s{sec:03d}.png")
            if os.path.exists(src):
                dst = os.path.join(out_dir, f"{planet}_s{sec:03d}.png")
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing.append(planet)

        print(f"Sector {sec:02d}: {copied} images -> {out_dir}/")
        if missing:
            print(f"  Not in this sector: {missing}")


if __name__ == "__main__":
    main()
