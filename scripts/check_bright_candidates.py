import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RJUP_RSUN = 0.10045
RSUN_AU   = 0.00465047

tepcat   = pd.read_csv("data/planets_ready_for_modeling.csv")
coverage = pd.read_csv("data/tess_coverage_raw.csv")

for col in ["R_b", "Teff", "Period", "R_A", "a(AU)"]:
    tepcat[col] = pd.to_numeric(tepcat[col], errors="coerce")

tepcat["k"]         = tepcat["R_b"] * RJUP_RSUN / tepcat["R_A"]
tepcat["aRs"]       = tepcat["a(AU)"] / (tepcat["R_A"] * RSUN_AU)
tepcat["depth_pct"] = (tepcat["k"] ** 2) * 100
arg = ((1 + tepcat["k"]) / tepcat["aRs"]).clip(-1, 1)
tepcat["T14_hr"]    = (tepcat["Period"] * 24 / np.pi) * np.arcsin(arg)

tepcat2 = tepcat.drop(columns=["has_pdcsap","n_sectors","sector_list"], errors="ignore")
df = tepcat2.merge(coverage[["System","has_pdcsap","n_sectors","sector_list"]],
                   on="System", how="left")

keywords = {
    "Hot Jupiter":   ["HD_209458", "WASP-18", "WASP-121", "HAT-P-07", "KELT-23", "WASP-043"],
    "Mini-Neptune":  ["55_Cnc", "HD_3167", "HD_63433", "Kepler-068", "TOI-1246", "pi_Men"],
}

for ptype, kws in keywords.items():
    print("=" * 75)
    print(f"{ptype}")
    print("=" * 75)
    print(f"  {'System':<22} {'R_b':>6} {'P(d)':>7} {'depth%':>7} {'T14hr':>6} {'Teff':>5} {'Nsec':>5}  sectors[:50]")
    print("-" * 75)
    for kw in kws:
        matches = df[df["System"].str.contains(kw, na=False)]
        for _, r in matches.iterrows():
            sec = str(r["sector_list"])[:50] if pd.notna(r["sector_list"]) else "—"
            print(f"  {r['System']:<22} {r['R_b']:>6.3f} {r['Period']:>7.3f} "
                  f"{r['depth_pct']:>7.3f} {r['T14_hr']:>6.2f} "
                  f"{int(r['Teff']):>5} {int(r['n_sectors']):>5}  {sec}")
    print()
