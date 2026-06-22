#!/usr/bin/env python3
"""
export_transit_csv.py

For one planet at a time, exports 5 CSVs per transit (T-wide windows each):

  pre_baseline.csv      — [tc - 1.5T, tc - 0.5T]   out-of-transit before
  in_transit_raw.csv    — [tc - 0.5T, tc + 0.5T]   raw PDC-SAP flux
  in_transit_model.csv  — batman model at same times
  in_transit_resid.csv  — raw - model residuals
  post_baseline.csv     — [tc + 0.5T, tc + 1.5T]   out-of-transit after

tc refinement (same as export_tess_focused.py):
  1. Sector-level stacking → best t0_sec
  2. Per-transit batman chi-squared minimization (uses full transit shape)
  3. Optional manual fine-tune from tc_manual_offsets.csv

Output structure:
  transit_csvs/
    {planet}/
      sector_{sec:03d}/
        transit_{n:02d}/
          pre_baseline.csv
          in_transit_raw.csv
          in_transit_model.csv
          in_transit_resid.csv
          post_baseline.csv

Usage:
    python export_transit_csv.py --planet KELT-20
    python export_transit_csv.py --planet KELT-20 --sector 40
    python export_transit_csv.py --planet KELT-20 --sector 40 --transit 3
"""

import argparse
import os
import re
import time as time_mod
import warnings
warnings.filterwarnings("ignore")

import batman
import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.optimize import minimize_scalar

BJD_OFFSET  = 2457000.0
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT    = os.path.join(BASE_DIR, "transit_csvs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

LD_TABLE = [
    (7000, 0.20, 0.20),
    (6000, 0.33, 0.25),
    (5000, 0.44, 0.25),
    (4000, 0.55, 0.22),
    (0,    0.65, 0.15),
]
GAP_THRESHOLD_DAYS = 10 / 60 / 24   # 10 minutes


# ── Shared utilities (mirrors export_tess_focused.py) ─────────────────────────

def robust_normalize(lc):
    flux_vals = np.array(lc.flux)
    if hasattr(flux_vals.flat[0], "value"):
        flux_vals = np.array([v.value for v in flux_vals])
    divisor = np.nanpercentile(flux_vals, 75)
    if divisor == 0 or np.isnan(divisor):
        divisor = 1.0
    return lk.LightCurve(time=lc.time, flux=flux_vals / divisor)


def download_sector_lc(host, sector, retries=4):
    delays = [10, 20, 30]
    for attempt in range(retries):
        try:
            sr = lk.search_lightcurve(host, mission="TESS", sector=sector,
                                      author="SPOC", exptime=120)
            if sr is None or len(sr) == 0:
                return None
            lc = sr[0].download(flux_column="pdcsap_flux")
            if lc is None:
                return None
            return robust_normalize(lc.remove_nans())
        except Exception as e:
            if attempt < retries - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                print(f"    Retry {attempt+1} ({e}) — waiting {wait}s")
                time_mod.sleep(wait)
            else:
                print(f"    Download failed: {e}")
                return None


def get_batman_model(time_arr, tc, period, rp, a_rs, inc, ecc, w, u1, u2):
    p = batman.TransitParams()
    p.t0 = tc; p.per = period; p.rp = abs(rp)
    p.a  = abs(a_rs); p.inc = inc; p.ecc = max(0.0, ecc)
    p.w  = w; p.u = [u1, u2]; p.limb_dark = "quadratic"
    try:
        return batman.TransitModel(p, time_arr).light_curve(p)
    except Exception:
        return np.ones_like(time_arr)


def refine_t0_stacking(time, flux, t0_init, period, half_dur, n_bins=40):
    phase = (time - t0_init) % period
    phase[phase > period / 2] -= period
    mask = np.abs(phase) <= 2 * half_dur
    if mask.sum() < 20:
        return t0_init
    p_in, f_in = phase[mask], flux[mask]
    bins = np.linspace(-2 * half_dur, 2 * half_dur, n_bins + 1)
    bct  = 0.5 * (bins[:-1] + bins[1:])
    bt, bf = [], []
    for i in range(n_bins):
        sel = (p_in >= bins[i]) & (p_in < bins[i+1])
        if sel.sum() > 0:
            bt.append(bct[i]); bf.append(np.median(f_in[sel]))
    if len(bt) < 5:
        return t0_init
    bt, bf = np.array(bt), np.array(bf)
    inner = np.abs(bt) <= half_dur
    if inner.sum() < 3:
        inner = np.ones(len(bt), dtype=bool)
    try:
        a2, b2, _ = np.polyfit(bt[inner], bf[inner], 2)
        if a2 <= 0:
            return t0_init
        offset = -b2 / (2 * a2)
        return t0_init + offset if abs(offset) <= half_dur else t0_init
    except Exception:
        return t0_init


def refine_tc_batman(time, flux, tc_guess, period, rp, a_rs, inc, ecc, u1, u2,
                     T_days, search_frac=0.6):
    half_search = search_frac * T_days
    mask = (time >= tc_guess - 1.5 * T_days) & (time <= tc_guess + 1.5 * T_days)
    if mask.sum() < 10:
        return tc_guess
    t_fit, f_fit = time[mask], flux[mask]

    def chi2(tc_shift):
        model = get_batman_model(t_fit, tc_guess + tc_shift,
                                 period, rp, a_rs, inc, ecc, 90.0, u1, u2)
        out = np.abs(t_fit - (tc_guess + tc_shift)) > 0.6 * T_days
        if out.sum() > 3:
            scale = np.median(f_fit[out]) / np.median(model[out])
            model = model * scale
        return np.sum((f_fit - model) ** 2)

    try:
        res   = minimize_scalar(chi2, bounds=(-half_search, half_search), method="bounded")
        shift = np.clip(res.x, -half_search, half_search)
        return tc_guess + shift
    except Exception:
        return tc_guess


def has_gap(time, tc, T_days):
    mask = (time >= tc - 0.5 * T_days) & (time <= tc + 0.5 * T_days)
    t_in = np.sort(time[mask])
    if len(t_in) < 2:
        return True
    return bool(np.any(np.diff(t_in) > GAP_THRESHOLD_DAYS))


def load_manual_offsets(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "tc_manual_offsets.csv")
    offsets = {}
    if not os.path.exists(path):
        return offsets
    df = pd.read_csv(path, comment="#")
    for _, row in df.iterrows():
        key = (str(row["planet"]).strip(), str(row["sector"]), str(row["transit_n"]))
        offsets[key] = float(row["offset_min"]) / 60 / 24
    return offsets


def get_manual_offset(offsets, planet, sector, transit_n):
    for s_key in (str(sector), "all"):
        for t_key in (str(transit_n), "all"):
            v = offsets.get((planet, s_key, t_key))
            if v is not None:
                return v
    return 0.0


def get_ld(teff):
    for thr, u1, u2 in LD_TABLE:
        if teff >= thr:
            return u1, u2
    return 0.44, 0.25


def btjd_to_utc(btjd):
    try:
        return Time(btjd + BJD_OFFSET, format="jd", scale="tdb").iso[:10]
    except Exception:
        return "?"


# ── CSV export ────────────────────────────────────────────────────────────────

def export_transit_csvs(t_win, f_win, model_win, tc_refined, T_days, out_dir):
    """Save 5 T-wide CSVs for one transit."""
    os.makedirs(out_dir, exist_ok=True)
    half_T = T_days / 2.0

    windows = {
        "pre_baseline":    (tc_refined - 1.5*T_days, tc_refined - half_T),
        "in_transit_raw":  (tc_refined - half_T,      tc_refined + half_T),
        "post_baseline":   (tc_refined + half_T,      tc_refined + 1.5*T_days),
    }

    in_mask = (t_win >= tc_refined - half_T) & (t_win <= tc_refined + half_T)
    resid   = f_win - model_win

    for name, (lo, hi) in windows.items():
        mask = (t_win >= lo) & (t_win <= hi)
        df = pd.DataFrame({
            "time_btjd": t_win[mask],
            "flux":      f_win[mask],
        })
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    # Model and residuals — in-transit window only
    pd.DataFrame({
        "time_btjd":   t_win[in_mask],
        "flux_model":  model_win[in_mask],
    }).to_csv(os.path.join(out_dir, "in_transit_model.csv"), index=False)

    pd.DataFrame({
        "time_btjd":  t_win[in_mask],
        "residual":   resid[in_mask],
    }).to_csv(os.path.join(out_dir, "in_transit_resid.csv"), index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",   required=True, type=str)
    parser.add_argument("--sector",   type=int, default=None)
    parser.add_argument("--transit",  type=int, default=None, help="Single transit_n to export")
    args = parser.parse_args()

    pri = pd.read_csv(os.path.join(DATA_DIR, "planets_priority.csv"))

    # ── Source of truth: N30 transit_selection.csv (verified by visual inspection)
    # Falls back to transit_plan_detailed.csv if not found.
    n30_sel_path = os.path.join(RESULTS_DIR, "N30", "transit_selection.csv")
    plan_path    = os.path.join(RESULTS_DIR, "summary", "transit_plan_detailed.csv")

    if os.path.exists(n30_sel_path):
        tsel = pd.read_csv(n30_sel_path)
        p_sel = tsel[tsel["planet"] == args.planet]
        if not p_sel.empty:
            if args.sector:
                p_sel = p_sel[p_sel["sector"] == args.sector]
            if args.transit:
                p_sel = p_sel[p_sel["transit_in_sector"] == args.transit]
            use_n30 = True
            print(f"Using N30 transit_selection.csv — {len(p_sel)} verified transit(s) for {args.planet}")
        else:
            use_n30 = False
    else:
        use_n30 = False

    if not use_n30:
        plan  = pd.read_csv(plan_path)
        p_sel = plan[plan["planet"] == args.planet]
        if p_sel.empty:
            print(f"Planet '{args.planet}' not found in transit_selection.csv or transit plan."); return
        if args.sector:
            p_sel = p_sel[p_sel["sector"] == args.sector]
        if args.transit:
            p_sel = p_sel[p_sel["transit_n"] == args.transit]
        print(f"Using transit_plan_detailed.csv — {len(p_sel)} transit(s) for {args.planet}")

    if p_sel.empty:
        print(f"No transits found for {args.planet} with given filters."); return

    prow = pri[pri["System"] == args.planet]
    if prow.empty:
        print(f"'{args.planet}' not found in planets_priority.csv"); return
    prow = prow.iloc[0]

    host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", args.planet))).strip()
    period = float(prow["Period"])
    T_days = float(prow["pl_trandur"]) / 24.0
    half_T = T_days / 2.0
    rp     = float(prow["pl_ratror"])
    a      = float(prow["pl_ratdor"])
    inc    = float(prow["pl_orbincl"])
    ecc_v  = prow["pl_orbeccen"]
    ecc    = float(ecc_v) if pd.notna(ecc_v) and float(ecc_v) >= 0 else 0.0
    teff   = float(prow["st_teff"]) if pd.notna(prow.get("st_teff")) else 5500.0
    u1, u2 = get_ld(teff)

    manual_offsets = load_manual_offsets()
    if manual_offsets:
        print(f"Loaded {len(manual_offsets)} manual offset(s) from tc_manual_offsets.csv")

    sectors = p_sel["sector"].unique()
    done = 0

    for sector in sectors:
        s_sel = p_sel[p_sel["sector"] == sector]

        print(f"\n[{args.planet}] Sector {sector} — downloading LC...", end=" ", flush=True)
        lc = download_sector_lc(host, sector)
        if lc is None:
            print("FAILED"); continue

        time_arr = np.array(lc.time.value) if hasattr(lc.time, "value") else np.array(lc.time)
        flux_arr = np.array(lc.flux.value) if hasattr(lc.flux, "value") else np.array(lc.flux)
        print(f"{len(time_arr)} pts")

        if use_n30:
            # Use verified tc values directly from transit_selection.csv
            selected_tcs   = list(s_sel["tc_btjd"].astype(float))
            local_transit_ns = list(s_sel["transit_in_sector"].astype(int))
            print(f"  {len(selected_tcs)} verified transits from N30 selection")
        else:
            # Legacy path: compute from t0 stacking
            rep_path = os.path.join(RESULTS_DIR, "survey", args.planet, "sector_report.csv")
            t0_sec = None
            if os.path.exists(rep_path):
                rep = pd.read_csv(rep_path)
                rep = rep[rep["sector"] != "Total"]
                rep["sector"] = rep["sector"].astype(int)
                rr = rep[rep["sector"] == sector]
                if not rr.empty:
                    t0_sec = float(rr["t0_sec_btjd"].values[0])
            if t0_sec is None:
                t0_sec = float(s_sel.iloc[0]["tc_btjd"])
            t0_sec = refine_t0_stacking(time_arr, flux_arr, t0_sec, period, half_T)
            n_lo = int(np.ceil((time_arr.min()  - t0_sec) / period))
            n_hi = int(np.floor((time_arr.max() - t0_sec) / period))
            all_tc = [t0_sec + n * period for n in range(n_lo, n_hi + 1)]
            valid_tcs = []
            for tc_c in all_tc:
                pre  = (time_arr >= tc_c - 1.5*T_days) & (time_arr < tc_c - half_T)
                in_t = (time_arr >= tc_c - half_T)      & (time_arr <= tc_c + half_T)
                post = (time_arr >  tc_c + half_T)       & (time_arr <= tc_c + 1.5*T_days)
                if pre.sum() >= 10 and in_t.sum() >= 10 and post.sum() >= 10 \
                   and not has_gap(time_arr, tc_c, T_days):
                    valid_tcs.append(tc_c)
            selected_tcs    = valid_tcs[:len(s_sel)]
            local_transit_ns = list(range(1, len(selected_tcs) + 1))
            print(f"  {len(valid_tcs)} valid transits found, exporting {len(selected_tcs)}")

        for t_idx, tc_pred in enumerate(selected_tcs):
            transit_n = local_transit_ns[t_idx]

            # Skip if --transit filter is set and this isn't it
            if args.transit and transit_n != args.transit:
                continue

            # Batman template chi-sq refinement
            tc_refined = refine_tc_batman(
                time_arr, flux_arr, tc_pred,
                period, rp, a, inc, ecc, u1, u2, T_days
            )

            # Manual offset
            manual_off = get_manual_offset(manual_offsets, args.planet, sector, transit_n)
            if manual_off != 0.0:
                tc_refined += manual_off
                print(f"    Manual offset: {manual_off*24*60:+.1f} min")

            # 5T display window for CSV
            mask_win = (time_arr >= tc_refined - 2.5*T_days) & \
                       (time_arr <= tc_refined + 2.5*T_days)
            t_win    = time_arr[mask_win]
            f_win    = flux_arr[mask_win]

            # Batman model (Y-anchored to data)
            model = get_batman_model(t_win, tc_refined, period, rp, a, inc, ecc, 90.0, u1, u2)
            out_m = np.abs(t_win - tc_refined) > 0.6 * T_days
            if out_m.sum() > 3:
                scale = np.median(f_win[out_m]) / np.median(model[out_m])
                model = model * scale

            out_dir = os.path.join(OUT_ROOT, args.planet,
                                   f"sector_{sector:03d}", f"transit_{transit_n:02d}")
            export_transit_csvs(t_win, f_win, model, tc_refined, T_days, out_dir)
            done += 1
            tc_utc = btjd_to_utc(tc_refined)
            print(f"  Transit {transit_n:02d}  tc={tc_refined:.5f}  ({tc_utc})  -> {out_dir}/")

    print(f"\nDone. {done} transit(s) exported for {args.planet}.")


if __name__ == "__main__":
    main()
