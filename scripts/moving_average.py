"""
Apply a moving average to a gap-free raw lightcurve CSV.

Input CSV must have columns: time_btjd, flux
Output CSV adds:  trend  (moving average)
                  residual (flux - trend)

Usage:
    python moving_average.py --input transit_csvs/Qatar-1/Qatar-1_S24_gapfree.csv
    python moving_average.py --input transit_csvs/Qatar-1/Qatar-1_S24_gapfree.csv --window 30
    python moving_average.py --input transit_csvs/Qatar-1/Qatar-1_S24_gapfree.csv --window 90 --plot

Window is in number of 2-min cadence points:
    15  pts =  30 min
    30  pts =   1 hr
    90  pts =   3 hr
   180  pts =   6 hr
   360  pts =  12 hr
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CADENCE_MIN = 2   # TESS 2-min cadence


def moving_average(flux, window):
    """Centered moving average using uniform weights. Edges use shrinking windows."""
    return pd.Series(flux).rolling(window=window, center=True, min_periods=1).mean().values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True,  help='Path to gap-free CSV (time_btjd, flux)')
    parser.add_argument('--window', type=int, default=30,
                        help='Moving average window in number of points (default=30 = 1 hr)')
    parser.add_argument('--plot',   action='store_true', help='Show diagnostic plot')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'File not found: {args.input}')
        return

    df = pd.read_csv(args.input)
    if 'flux' not in df.columns:
        print(f'ERROR: no "flux" column found. Columns present: {list(df.columns)}')
        return

    window_min = args.window * CADENCE_MIN
    print(f'Input:  {args.input}  ({len(df)} points)')
    print(f'Window: {args.window} pts = {window_min} min = {window_min/60:.1f} hr')

    df['trend']    = moving_average(df['flux'].values, args.window)
    df['residual'] = df['flux'] - df['trend']

    # Output alongside input file
    base   = os.path.splitext(args.input)[0]
    outpath = f'{base}_ma{args.window}.csv'
    df.to_csv(outpath, index=False)
    print(f'Saved: {outpath}')
    print(f'  residual std  = {df["residual"].std():.6f}')
    print(f'  residual mean = {df["residual"].mean():.6f}')

    if args.plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        name = os.path.basename(args.input)

        axes[0].plot(df['time_btjd'], df['flux'], '.', ms=1, alpha=0.4,
                     color='steelblue', label='raw flux')
        axes[0].plot(df['time_btjd'], df['trend'], '-', lw=1.2, color='red',
                     alpha=0.9, label=f'moving avg ({args.window} pts = {window_min} min)')
        axes[0].set_ylabel('Normalized flux')
        axes[0].set_title(f'{name} — Raw flux + Moving average trend')
        axes[0].legend(fontsize=9)

        axes[1].plot(df['time_btjd'], df['residual'], '.', ms=1, alpha=0.5,
                     color='steelblue')
        axes[1].axhline(0, color='red', lw=0.8, ls='--')
        axes[1].set_ylabel('Residual (flux − trend)')
        axes[1].set_xlabel('BTJD (days)')
        axes[1].set_title(f'Residuals  |  std = {df["residual"].std():.5f}')

        plt.tight_layout()
        outfig = f'{base}_ma{args.window}.png'
        plt.savefig(outfig, dpi=130, bbox_inches='tight')
        print(f'Plot:  {outfig}')
        plt.show()


if __name__ == '__main__':
    main()
