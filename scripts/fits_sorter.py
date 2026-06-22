#!/usr/bin/env python3
"""
fits_sorter.py — Automate download and organization of FITS files
from the MicroObservatory Image Directory for EXOTIC exoplanet transit analysis.

Modes:
  download  — Scrape and download FITS files by target name and date range
  sort      — Organize local FITS files by reading their FITS headers
  prep      — Generate an EXOTIC-compatible inits.json with NASA planet priors

Usage:
  python fits_sorter.py download --target HATP-32 --days 30 --output ./data
  python fits_sorter.py sort     --input ./unsorted_fits --output ./data
  python fits_sorter.py prep     --target HATP-32 --date 2026-04-15 --data-dir ./data

Dependencies:
  pip install requests beautifulsoup4 astropy numpy colorama
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import requests
from astropy.io import fits
from bs4 import BeautifulSoup
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform ANSI color support
colorama_init(autoreset=True)

# Force UTF-8 output on Windows to avoid cp1252 encoding errors for box-draw chars
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# --- Constants ----------------------------------------------------------------

MO_BASE_URL   = "https://waps.cfa.harvard.edu/microobservatory/MOImageDirectory/"
MO_INDEX_URL  = MO_BASE_URL + "ImageDirectory.php"
NASA_TAP_URL  = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

FITS_EXTENSIONS = (".fits", ".fts", ".fit")

# --- Colored logging helpers --------------------------------------------------

def log_info(msg: str) -> None:
    """Print a cyan informational message."""
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL}  {msg}")

def log_ok(msg: str) -> None:
    """Print a green success message."""
    print(f"{Fore.GREEN}[ OK ]{Style.RESET_ALL}  {msg}")

def log_warn(msg: str) -> None:
    """Print a yellow warning message."""
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL}  {msg}")

def log_err(msg: str) -> None:
    """Print a red error message."""
    print(f"{Fore.RED}[ERR ]{Style.RESET_ALL}  {msg}")

def log_header(title: str) -> None:
    """Print a bold magenta section header."""
    bar = "-" * 62
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{bar}")
    print(f"  {title}")
    print(f"{bar}{Style.RESET_ALL}\n")

def log_summary_row(label: str, value, color: str = Fore.WHITE) -> None:
    """Print one row of a summary table."""
    print(f"  {Fore.CYAN}{label:<18}{Style.RESET_ALL}{color}{value}{Style.RESET_ALL}")

# --- Shared utilities ---------------------------------------------------------

def normalize_target(name: str) -> str:
    """
    Normalize a target/object name for use as a directory name.
    Converts to uppercase and replaces spaces with hyphens.
    e.g.  'hatp 32 b' -> 'HATP-32-B'
    """
    return re.sub(r"\s+", "-", name.strip().upper())


def parse_date_obs(date_str: str) -> Optional[datetime]:
    """
    Parse a DATE-OBS FITS keyword string into a Python datetime.
    Accepts ISO 8601 variants: '2026-04-15T22:30:00', '2026-04-15T22:30:00.123', '2026-04-15'.
    Returns None if parsing fails.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def safe_header(hdr, key: str, default=None):
    """
    Safely retrieve a value from an astropy FITS header.
    Returns `default` if the keyword is missing or the header is None.
    """
    try:
        return hdr[key]
    except (KeyError, TypeError):
        return default


def is_fits_file(path: Path) -> bool:
    """Return True if the file has a known FITS extension."""
    return path.suffix.lower() in FITS_EXTENSIONS


def iter_fits_files(directory: Path):
    """Yield all FITS files found recursively under `directory`."""
    for ext in FITS_EXTENSIONS:
        yield from directory.rglob(f"*{ext}")


# --- MODE 1 — DOWNLOAD --------------------------------------------------------

def _extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Attempt to extract an observation date from a FITS filename.
    Recognises patterns like: 2026-04-01, 20260401, 2026_04_01
    Returns a datetime or None.
    """
    patterns = [
        r"(\d{4})[-_](\d{2})[-_](\d{2})",  # 2026-04-01 / 2026_04_01
        r"(\d{4})(\d{2})(\d{2})",            # 20260401
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _scrape_links(
    soup: BeautifulSoup,
    base_url: str,
) -> tuple[list[dict], list[str]]:
    """
    Extract FITS file links and sub-directory links from a parsed HTML page.

    Returns:
        fits_links : list of {'filename': str, 'url': str}
        dir_links  : list of absolute URL strings for sub-directories
    """
    fits_links: list[dict] = []
    dir_links:  list[str]  = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(base_url, href)

        lower = href.lower()
        if any(lower.endswith(ext) for ext in FITS_EXTENSIONS):
            fits_links.append({"filename": os.path.basename(href), "url": full_url})
        elif href.endswith("/") and href not in ("/", "../", "./"):
            dir_links.append(full_url)

    return fits_links, dir_links


def scrape_image_directory(
    target: str,
    days: int,
    session: requests.Session,
) -> list[dict]:
    """
    Scrape the MicroObservatory Image Directory for FITS files matching
    `target` within the past `days` days.

    Strategy:
      1. Load the main ImageDirectory.php page.
      2. Collect all direct .fits links and sub-directory links.
      3. Recurse one level deep into any sub-directories found.
      4. Filter collected links by target name (filename substring match)
         and by date extracted from the filename.

    Returns a list of record dicts:
      {'filename': str, 'url': str, 'target': str, 'date': datetime | None}
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    target_norm = normalize_target(target)

    log_info(f"Scraping MicroObservatory Image Directory ...")
    log_info(f"Target : {target_norm}  |  Cutoff : {cutoff.strftime('%Y-%m-%d')} (past {days} days)")

    try:
        resp = session.get(MO_INDEX_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log_err(f"Cannot reach MicroObservatory directory: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    all_fits, dir_links = _scrape_links(soup, MO_INDEX_URL)

    # Recurse one level into sub-directories (e.g. date-based folders)
    for dir_url in dir_links:
        try:
            sub = session.get(dir_url, timeout=30)
            sub.raise_for_status()
            sub_fits, _ = _scrape_links(BeautifulSoup(sub.text, "html.parser"), dir_url)
            all_fits.extend(sub_fits)
        except requests.RequestException as exc:
            log_warn(f"Could not load sub-directory {dir_url}: {exc}")

    log_info(f"Total FITS links found on directory page: {len(all_fits)}")

    # -- Filter by target ------------------------------------------------------
    # Normalise both sides to alphanumeric-only for flexible matching
    # e.g. 'HATP-32' matches filenames containing 'hatp32', 'hat-p-32', etc.
    def _alphanum(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    target_key = _alphanum(target_norm)

    results: list[dict] = []
    for link in all_fits:
        if target_key not in _alphanum(link["filename"]):
            continue  # Target name not found in filename — skip

        file_date = _extract_date_from_filename(link["filename"])
        if file_date and file_date < cutoff:
            continue  # File is older than the requested date range

        results.append({
            "filename": link["filename"],
            "url":      link["url"],
            "target":   target_norm,
            "date":     file_date,
        })

    return results


def download_fits_files(
    records: list[dict],
    output_dir: Path,
    dry_run: bool,
    session: requests.Session,
) -> tuple[int, int]:
    """
    Download FITS files described by `records` into
    `output_dir/<TARGET>/<YYYY-MM-DD>/<filename>.fits`.

    Skips files whose destination path already exists.

    Returns:
        (downloaded, skipped) counts
    """
    downloaded = 0
    skipped    = 0

    for rec in records:
        target_dir = output_dir / rec.get("target", "UNKNOWN")
        date_label = rec["date"].strftime("%Y-%m-%d") if rec.get("date") else "unknown-date"
        dest_dir   = target_dir / date_label
        dest_file  = dest_dir   / rec["filename"]

        if dest_file.exists():
            skipped += 1
            log_warn(f"Skip (exists): {rec['filename']}")
            continue

        if dry_run:
            log_info(f"[DRY-RUN] {rec['url']}  ->  {dest_file}")
            downloaded += 1  # Count as "would download"
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            log_info(f"Downloading {rec['filename']} ...")
            r = session.get(rec["url"], timeout=120, stream=True)
            r.raise_for_status()
            with open(dest_file, "wb") as fh:
                for chunk in r.iter_content(chunk_size=131072):  # 128 KiB chunks
                    fh.write(chunk)
            log_ok(f"Saved -> {dest_file}")
            downloaded += 1
        except requests.RequestException as exc:
            log_err(f"Failed to download {rec['filename']}: {exc}")

    return downloaded, skipped


def cmd_download(args: argparse.Namespace) -> None:
    """
    CLI entry point for 'download' mode.
    Scrapes MicroObservatory, filters by target/date, downloads FITS files.
    """
    log_header("MODE 1 — DOWNLOAD")
    output_dir = Path(args.output)

    session = requests.Session()
    session.headers["User-Agent"] = "fits_sorter/1.0 (CVIF Jagna Bohol astronomy pipeline)"

    records     = scrape_image_directory(args.target, args.days, session)
    total_found = len(records)

    log_info(f"Matching FITS files found: {total_found}")

    if total_found == 0:
        log_warn("No files matched — try a different --target spelling or wider --days range.")
        return

    if args.dry_run:
        log_warn("DRY-RUN mode — no files will be downloaded.")

    downloaded, skipped = download_fits_files(records, output_dir, args.dry_run, session)

    # -- Summary ---------------------------------------------------------------
    print(f"\n{Fore.CYAN}{'-'*44}")
    print(f"  Download Summary  —  target: {Fore.WHITE}{Style.BRIGHT}{normalize_target(args.target)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*44}{Style.RESET_ALL}")
    log_summary_row("Found :", total_found)
    log_summary_row("Downloaded :", downloaded, Fore.GREEN)
    log_summary_row("Skipped :", skipped, Fore.YELLOW)
    if args.dry_run:
        print(f"\n  {Fore.YELLOW}(Dry-run: no files were actually saved){Style.RESET_ALL}")
    print()


# --- MODE 2 — SORT/ORGANIZE ---------------------------------------------------

def read_fits_metadata(fits_path: Path) -> Optional[dict]:
    """
    Open a FITS file and extract sorting metadata from the primary HDU header.

    Extracted fields:
      object, date_obs (raw string + parsed datetime), exptime,
      filter, telescope, ra, dec

    Returns a dict on success, or None if the file cannot be opened.
    """
    try:
        with fits.open(fits_path, memmap=False, ignore_missing_simple=True) as hdul:
            hdr = hdul[0].header

            # Object/target name — try multiple common keyword variants
            obj = (
                safe_header(hdr, "OBJECT")
                or safe_header(hdr, "TARGET")
                or safe_header(hdr, "OBJNAME")
                or "UNKNOWN"
            )

            date_obs_raw: str = safe_header(hdr, "DATE-OBS", "") or ""
            date_obs_dt  = parse_date_obs(date_obs_raw) if date_obs_raw else None

            # Filter keyword — try common variants
            filt = (
                safe_header(hdr, "FILTER")
                or safe_header(hdr, "FILTNAME")
                or safe_header(hdr, "FILT")
                or "UNKNOWN"
            )

            # RA/Dec — try standard keywords then WCS fallbacks
            ra  = safe_header(hdr, "RA")  or safe_header(hdr, "CRVAL1")
            dec = safe_header(hdr, "DEC") or safe_header(hdr, "CRVAL2")

            imagetyp = str(safe_header(hdr, "IMAGETYP", "") or "").strip()

            return {
                "filename":     fits_path.name,
                "path":         fits_path,
                "object":       str(obj).strip().upper(),
                "imagetyp":     imagetyp,
                "date_obs_raw": date_obs_raw,
                "date_obs":     date_obs_dt,
                "exptime":      safe_header(hdr, "EXPTIME"),
                "filter":       str(filt).strip(),
                "telescope":    str(safe_header(hdr, "TELESCOP", "UNKNOWN")).strip(),
                "ra":           ra,
                "dec":          dec,
            }
    except Exception as exc:
        log_warn(f"Cannot read FITS header for {fits_path.name}: {exc}")
        return None


def is_dark_frame(meta: dict) -> bool:
    """
    Return True if this FITS file should be classified as a dark calibration frame.

    Detection rules (any one match is sufficient):
      1. IMAGETYP keyword equals 'Dark Frame' or 'DARK' (case-insensitive).
      2. OBJECT keyword contains the word 'dark' (case-insensitive).
    """
    imagetyp = meta.get("imagetyp", "").upper()
    if imagetyp in ("DARK FRAME", "DARK"):
        return True
    if "DARK" in meta.get("object", "").upper():
        return True
    return False


def write_manifest(manifest_path: Path, records: list[dict]) -> None:
    """
    Write a CSV manifest file for EXOTIC batch processing.

    Columns: filename, date_obs, exptime, filter, telescope, ra, dec
    """
    fieldnames = ["filename", "date_obs", "exptime", "filter", "telescope", "ra", "dec"]
    try:
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for meta in records:
                writer.writerow({
                    "filename":  meta["filename"],
                    "date_obs":  meta.get("date_obs_raw", ""),
                    "exptime":   meta.get("exptime", ""),
                    "filter":    meta.get("filter", ""),
                    "telescope": meta.get("telescope", ""),
                    "ra":        meta.get("ra", ""),
                    "dec":       meta.get("dec", ""),
                })
        log_ok(f"Manifest written -> {manifest_path}")
    except OSError as exc:
        log_err(f"Could not write manifest {manifest_path}: {exc}")


def sort_fits_files(input_dir: Path, output_dir: Path, copy: bool = False) -> None:
    """
    Scan `input_dir` recursively for FITS files, read their headers,
    and move (or copy) each file to:
        output_dir/<TARGET>/<YYYY-MM-DD>/<filename>.fits

    After organising each (target, date) group, writes a manifest.csv
    that EXOTIC can use for batch processing.
    """
    all_fits = list(iter_fits_files(input_dir))
    total    = len(all_fits)

    if total == 0:
        log_warn(f"No FITS files found in: {input_dir}")
        return

    log_info(f"Found {total} FITS file(s) in {input_dir}")

    # Accumulate metadata per (target, date) group for the CSV manifest
    manifest_groups: dict[tuple[str, str], list[dict]] = {}
    moved   = 0
    darks   = 0
    failed  = 0

    for fits_path in sorted(all_fits):
        meta = read_fits_metadata(fits_path)
        if meta is None:
            failed += 1
            continue

        target     = normalize_target(meta["object"])
        date_label = meta["date_obs"].strftime("%Y-%m-%d") if meta["date_obs"] else "unknown-date"

        # -- Dark frame detection ----------------------------------------------
        # Dark calibration frames are routed to a '_darks' subfolder so they
        # don't pollute the science frame directories used by EXOTIC.
        if is_dark_frame(meta):
            dest_dir = output_dir / target / date_label / "_darks"
            darks   += 1
            log_info(f"Dark frame detected: {fits_path.name}  ->  {dest_dir.relative_to(output_dir)}")
        else:
            dest_dir = output_dir / target / date_label

        dest_file = dest_dir / fits_path.name

        # Create destination directory if needed
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_err(f"Cannot create directory {dest_dir}: {exc}")
            failed += 1
            continue

        # Skip if destination already exists to avoid overwriting
        if dest_file.exists():
            log_warn(f"Skip (exists): {fits_path.name}")
            # Still record metadata for the manifest
            manifest_groups.setdefault((target, date_label), []).append(meta)
            continue

        # Move or copy the file
        try:
            if copy:
                shutil.copy2(fits_path, dest_file)
            else:
                shutil.move(str(fits_path), dest_file)

            # Update the stored path to the new location
            meta["path"] = dest_file
            rel = dest_file.relative_to(output_dir)
            op  = "Copied" if copy else "Moved"
            log_ok(f"{op}: {fits_path.name}  ->  {rel}")
            moved += 1
        except (OSError, shutil.Error) as exc:
            log_err(f"Cannot {'copy' if copy else 'move'} {fits_path.name}: {exc}")
            failed += 1
            continue

        manifest_groups.setdefault((target, date_label), []).append(meta)

    # -- Write per-(target, date) CSV manifests -------------------------------
    for (target, date_label), records in manifest_groups.items():
        manifest_path = output_dir / target / date_label / "manifest.csv"
        write_manifest(manifest_path, records)

    # -- Summary ---------------------------------------------------------------
    print(f"\n{Fore.CYAN}{'-'*44}")
    print(f"  Sort Summary")
    print(f"{Fore.CYAN}{'-'*44}{Style.RESET_ALL}")
    log_summary_row("Total files :", total)
    log_summary_row("Moved/Copied :", moved, Fore.GREEN)
    log_summary_row("Dark frames :", darks, Fore.BLUE)
    log_summary_row("Failed :", failed, Fore.RED if failed else Fore.WHITE)
    print()


def cmd_sort(args: argparse.Namespace) -> None:
    """CLI entry point for 'sort' mode."""
    log_header("MODE 2 — SORT / ORGANIZE")
    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        log_err(f"Input directory not found: {input_dir}")
        sys.exit(1)

    sort_fits_files(input_dir, output_dir, copy=args.copy)


# --- MODE 3 — EXOTIC PREP -----------------------------------------------------

def fetch_planet_priors(target: str) -> dict:
    """
    Query the NASA Exoplanet Archive TAP (Table Access Protocol) service
    for transit/orbital priors for `target`.

    Columns retrieved:
      pl_name, pl_orbper, pl_ratror, pl_ratdor, pl_orbincl, pl_tranmid, ra, dec

    The query uses a LIKE match against pl_name to handle slight spelling
    variations (e.g. 'HAT-P-32 b' vs 'HATP-32').

    Returns a dict of priors (may be empty if the target is not found or
    a network/parsing error occurs).
    """
    # Build a human-friendly search string: 'HATP-32' -> 'HAT P 32'
    # and let the LIKE wildcard catch variations
    search_name = re.sub(r"[-_]", " ", target).strip()

    adql = (
        "SELECT pl_name, pl_orbper, pl_ratror, pl_ratdor, "
        "pl_orbincl, pl_tranmid, ra, dec "
        "FROM ps "
        f"WHERE pl_name LIKE '%{search_name}%' "
        "AND default_flag=1"
    )

    log_info(f"Querying NASA Exoplanet Archive for '{search_name}' ...")

    try:
        resp = requests.get(
            NASA_TAP_URL,
            params={"query": adql, "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()
    except requests.RequestException as exc:
        log_warn(f"Network error querying NASA Exoplanet Archive: {exc}")
        return {}
    except (ValueError, json.JSONDecodeError) as exc:
        log_warn(f"Unexpected response from NASA Exoplanet Archive: {exc}")
        return {}

    if not rows:
        log_warn(f"No planet priors found for '{search_name}' in NASA Exoplanet Archive.")
        return {}

    row = rows[0]
    priors = {
        "pl_name":  row.get("pl_name", target),
        "period":   row.get("pl_orbper"),    # Orbital period [days]
        "rp_rs":    row.get("pl_ratror"),    # Rp/Rs ratio
        "a_rs":     row.get("pl_ratdor"),    # a/Rs ratio
        "inc":      row.get("pl_orbincl"),   # Inclination [deg]
        "t0":       row.get("pl_tranmid"),   # Mid-transit time [BJD-UTC]
        "ra":       row.get("ra"),
        "dec":      row.get("dec"),
    }
    log_ok(
        f"Priors retrieved — period={priors['period']} d, "
        f"Rp/Rs={priors['rp_rs']}, a/Rs={priors['a_rs']}, inc={priors['inc']}°"
    )
    return priors


def build_exotic_inits(
    target: str,
    date_str: str,
    data_dir: Path,
    priors: dict,
) -> dict:
    """
    Construct an EXOTIC-compatible inits.json dictionary.

    The structure follows the EXOTIC v2 init file specification:
      https://github.com/exotic-bzar/exotic

    FITS file paths are collected from data_dir/<TARGET>/<date_str>/
    and embedded in the 'exotic_settings.fits_files' list.

    Parameters
    ----------
    target   : Normalized target name (e.g. 'HATP-32')
    date_str : Observation date string 'YYYY-MM-DD'
    data_dir : Root data directory produced by 'download' or 'sort' modes
    priors   : Dict returned by fetch_planet_priors()
    """
    target_norm = normalize_target(target)
    obs_dir     = data_dir / target_norm / date_str
    output_dir  = str(obs_dir / "exotic_output")

    # Collect FITS files already in the observation directory
    fits_files: list[str] = sorted(
        str(p) for p in iter_fits_files(obs_dir)
    ) if obs_dir.exists() else []

    if not fits_files:
        log_warn(f"No FITS files found in {obs_dir} — fits_files list will be empty.")

    # Split host star name from planet designation
    # e.g. 'HATP-32' -> host star 'HAT-P-32', planet 'HAT-P-32 b'
    planet_name = priors.get("pl_name", f"{target_norm} b")
    host_name   = priors.get("pl_name", target_norm).rsplit(" ", 1)[0] if priors else target_norm

    inits = {
        "user_info": {
            "Directory with FITS files":  str(obs_dir),
            "Directory to Save Plots":    output_dir,
            "Directory of Flats":         "",
            "Directory of Darks":         "",
            "Directory of Biases":        "",
        },
        "planetary_parameters": {
            "Target Star RA":                              str(priors.get("ra", "")),
            "Target Star Dec":                             str(priors.get("dec", "")),
            "Planet Name":                                 planet_name,
            "Host Star Name":                              host_name,
            "Orbital Period (days)":                       priors.get("period"),
            "Orbital Period Uncertainty":                  None,
            "Published Mid-Transit Time (BJD-UTC)":        priors.get("t0"),
            "Mid-Transit Time Uncertainty":                None,
            "Ratio of Planet to Stellar Radius (Rp/Rs)":  priors.get("rp_rs"),
            "Ratio of Planet to Stellar Radius (Rp/Rs) Uncertainty": None,
            "Ratio of Distance to Stellar Radius (a/Rs)": priors.get("a_rs"),
            "Ratio of Distance to Stellar Radius (a/Rs) Uncertainty": None,
            "Orbital Inclination (deg)":                   priors.get("inc"),
            "Orbital Inclination (deg) Uncertainty":       None,
            "Orbital Eccentricity (0 if null)":            0.0,
            # Stellar parameters — fill in manually or from SIMBAD/TIC
            "Star Effective Temperature (K)":              None,
            "Star Effective Temperature (+) Uncertainty":  None,
            "Star Effective Temperature (-) Uncertainty":  None,
            "Star Metallicity ([FE/H])":                   None,
            "Star Metallicity (+) Uncertainty":            None,
            "Star Metallicity (-) Uncertainty":            None,
            "Star Surface Gravity (log(g))":               None,
            "Star Surface Gravity (+) Uncertainty":        None,
            "Star Surface Gravity (-) Uncertainty":        None,
        },
        "observation_parameters": {
            "Observation date":          date_str,
            "Obs. Latitude":             "",   # Fill in: e.g. '+9.8432' for Jagna
            "Obs. Longitude":            "",   # Fill in: e.g. '124.3691'
            "Obs. Elevation (meters)":   0,
            "Camera Type (CCD or DSLR)": "CCD",
            "Pixel Binning":             "1x1",
            "Filter Name (aavso.org/filters)": "CV",
            "Observing Notes": (
                f"Auto-generated by fits_sorter.py | "
                f"Target: {target_norm} | Date: {date_str}"
            ),
        },
        "exotic_settings": {
            "planet":     planet_name,
            "fits_files": fits_files,
        },
    }
    return inits


def cmd_prep(args: argparse.Namespace) -> None:
    """
    CLI entry point for 'prep' mode.
    Generates an EXOTIC-compatible inits.json for a given target/date,
    optionally pre-filling planet priors from the NASA Exoplanet Archive.
    """
    log_header("MODE 3 — EXOTIC PREP")

    # Validate date argument
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        log_err(f"Invalid date format: '{args.date}' — expected YYYY-MM-DD.")
        sys.exit(1)

    data_dir    = Path(args.data_dir)
    target_norm = normalize_target(args.target)
    obs_dir     = data_dir / target_norm / args.date

    if not obs_dir.exists():
        log_warn(f"Observation directory does not exist yet: {obs_dir}")
        log_warn("Proceeding — FITS file list will be empty; fill it in manually.")

    # Fetch planet priors from NASA Exoplanet Archive
    priors = fetch_planet_priors(args.target)

    # Build the EXOTIC inits dictionary
    inits = build_exotic_inits(args.target, args.date, data_dir, priors)

    # Write inits.json into the observation directory
    obs_dir.mkdir(parents=True, exist_ok=True)
    out_path = obs_dir / "inits.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            # default=str converts None -> 'None' to keep the JSON valid
            json.dump(inits, fh, indent=4, default=str)
        log_ok(f"EXOTIC init file written -> {out_path}")
    except OSError as exc:
        log_err(f"Failed to write inits.json: {exc}")
        sys.exit(1)

    n_fits = len(inits["exotic_settings"]["fits_files"])

    # -- Summary ---------------------------------------------------------------
    print(f"\n{Fore.CYAN}{'-'*44}")
    print(f"  EXOTIC Prep Summary")
    print(f"{Fore.CYAN}{'-'*44}{Style.RESET_ALL}")
    log_summary_row("Target :",      target_norm,  Fore.WHITE + Style.BRIGHT)
    log_summary_row("Date :",        args.date)
    log_summary_row("FITS files :",  n_fits,       Fore.GREEN if n_fits else Fore.YELLOW)
    log_summary_row(
        "Planet priors :",
        "NASA Exoplanet Archive" if priors else "Not found — fill manually",
        Fore.GREEN if priors else Fore.YELLOW,
    )
    log_summary_row("Output :", str(out_path))
    print()
    if not priors:
        print(
            f"  {Fore.YELLOW}Tip:{Style.RESET_ALL} Open {out_path} and fill in the "
            "'planetary_parameters' block manually.\n"
            "  Try searching https://exoplanetarchive.ipac.caltech.edu/ "
            "for the exact planet name.\n"
        )


# --- CLI argument parser ------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with three sub-commands."""
    parser = argparse.ArgumentParser(
        prog="fits_sorter",
        description=(
            "Automate download and organisation of FITS files from MicroObservatory\n"
            "and prepare them for EXOTIC exoplanet transit light-curve analysis.\n\n"
            "Examples:\n"
            "  python fits_sorter.py download --target HATP-32 --days 30 --output ./data\n"
            "  python fits_sorter.py sort     --input ./raw_fits --output ./data\n"
            "  python fits_sorter.py prep     --target HATP-32 --date 2026-04-15 --data-dir ./data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subs = parser.add_subparsers(dest="mode", required=True)

    # -- download --------------------------------------------------------------
    dl = subs.add_parser(
        "download",
        help="Scrape MicroObservatory and download FITS files",
        description=(
            "Scrapes the MicroObservatory Image Directory, filters by target name\n"
            "and date range, and downloads matching FITS files into an organised\n"
            "directory tree: output/<TARGET>/<YYYY-MM-DD>/<file>.fits"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dl.add_argument(
        "--target", required=True, metavar="NAME",
        help="Target object name to search for (e.g. HATP-32, WASP-43, TOI-1516)",
    )
    dl.add_argument(
        "--days", type=int, default=30, metavar="N",
        help="Search the past N days (default: 30)",
    )
    dl.add_argument(
        "--output", default="./data", metavar="DIR",
        help="Root output directory (default: ./data)",
    )
    dl.add_argument(
        "--dry-run", action="store_true",
        help="List matching files without downloading them",
    )

    # -- sort ------------------------------------------------------------------
    sort = subs.add_parser(
        "sort",
        help="Organise existing FITS files using their FITS headers",
        description=(
            "Reads the FITS header of each file in --input and moves (or copies)\n"
            "it to: output/<OBJECT>/<DATE-OBS>/<file>.fits\n"
            "Also writes a manifest.csv per group for EXOTIC batch processing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sort.add_argument(
        "--input", required=True, metavar="DIR",
        help="Directory containing unsorted FITS files (searched recursively)",
    )
    sort.add_argument(
        "--output", default="./data", metavar="DIR",
        help="Root output directory (default: ./data)",
    )
    sort.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of moving them (original files are preserved)",
    )

    # -- prep ------------------------------------------------------------------
    prep = subs.add_parser(
        "prep",
        help="Generate an EXOTIC-compatible inits.json with NASA planet priors",
        description=(
            "Generates an EXOTIC inits.json for a given target and observation date.\n"
            "Planet priors (period, Rp/Rs, a/Rs, inc, T0) are auto-fetched from the\n"
            "NASA Exoplanet Archive TAP service when available."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    prep.add_argument(
        "--target", required=True, metavar="NAME",
        help="Target planet name (e.g. HATP-32, WASP-43b)",
    )
    prep.add_argument(
        "--date", required=True, metavar="YYYY-MM-DD",
        help="Observation date (e.g. 2026-04-15)",
    )
    prep.add_argument(
        "--data-dir", default="./data", metavar="DIR",
        help="Root data directory from 'download' or 'sort' (default: ./data)",
    )

    return parser


# --- Entry point --------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate mode handler."""
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "download": cmd_download,
        "sort":     cmd_sort,
        "prep":     cmd_prep,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
