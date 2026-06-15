#!/usr/bin/env python3
"""
Convert PyRPL CurveDB .dat files (spectrum-analyzer 'Save curve' output) to CSV.

Each .dat is a pickled Python list:  [curve_id, params_dict, data]
  - curve_id : int
  - params   : dict (span, window, curve_unit='Vpk^2', display flags, ...)
  - data     : list of arrays; data[0] is the frequency axis (Hz),
               data[1:] are the traces.

Traces are written as columns. Real traces in Vpk^2 also get a dB(Vpk^2)
column (10*log10) to match the GUI's display_unit. Complex traces are split
into real/imag + magnitude(dB)/phase(deg). The params dict is written as
comment lines (#) at the top of each CSV.

Usage:
    python dat_to_csv.py                       # ~/pyrpl_user_dir/curve  ->  .../csv
    python dat_to_csv.py /path/to/curve
    python dat_to_csv.py /path/to/curve  out_dir
"""

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd


def load_dat(path):
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def trace_names(params, n):
    """Best-effort human names for the n traces, based on params."""
    if not isinstance(params, dict):
        return [f"trace_{i}" for i in range(n)]
    names = []
    if params.get("display_input1_baseband"):
        names.append(f"in1_{params.get('input1_baseband', 'ch1')}")
    if params.get("display_input2_baseband"):
        names.append(f"in2_{params.get('input2_baseband', 'ch2')}")
    if params.get("display_cross_amplitude"):
        names.append("cross")
    if len(names) != n:
        names = [f"trace_{i}" for i in range(n)]
    return names


def convert(obj):
    """Return (DataFrame, params_dict) for a CurveDB list object."""
    if isinstance(obj, list) and len(obj) >= 3 and isinstance(obj[1], dict):
        params = obj[1]
        data = obj[2]
    elif isinstance(obj, list) and len(obj) == 2:
        params = obj[0] if isinstance(obj[0], dict) else {}
        data = obj[1]
    else:
        params, data = {}, obj

    arrays = [np.asarray(a) for a in data]
    x = arrays[0].astype(float)
    ys = arrays[1:]
    names = trace_names(params, len(ys))

    df = pd.DataFrame(index=pd.Index(x, name="frequency_Hz"))
    for name, y in zip(names, ys):
        if np.iscomplexobj(y):
            df[f"{name}_re"] = np.real(y)
            df[f"{name}_im"] = np.imag(y)
            mag = np.abs(y)
            df[f"{name}_mag_dB"] = 10.0 * np.log10(mag + 1e-30)
            df[f"{name}_phase_deg"] = np.angle(y, deg=True)
        else:
            y = y.astype(float)
            df[f"{name}_Vpk2"] = y
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"{name}_dB"] = 10.0 * np.log10(np.where(y > 0, y, np.nan))
    return df, params


def convert_dir(in_dir=None, out_dir=None):
    """Convert every .dat in ``in_dir`` to a CSV in ``out_dir``.

    Both default to the standard PyRPL location (~/pyrpl_user_dir/curve and
    its ``csv`` subfolder). Returns the output directory. Call this directly
    from a notebook or other script instead of going through the CLI.
    """
    home = os.path.expanduser("~")
    if in_dir is None:
        in_dir = os.path.join(home, "pyrpl_user_dir", "curve")
    if out_dir is None:
        out_dir = os.path.join(in_dir, "csv")

    if not os.path.isdir(in_dir):
        sys.exit(f"Curve folder not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.dat")),
                   key=lambda p: (len(os.path.basename(p)), p))
    if not files:
        sys.exit(f"No .dat files in {in_dir}")

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            df, params = convert(load_dat(path))
            out_path = os.path.join(out_dir, stem + ".csv")
            with open(out_path, "w", newline="") as fh:
                for k, v in params.items():
                    fh.write(f"# {k}: {v}\n")
                df.to_csv(fh)
            print(f"{os.path.basename(path)} -> {out_path}  "
                  f"({len(df)} rows, {len(df.columns)} cols: {', '.join(df.columns)})")
        except Exception as e:
            print(f"{os.path.basename(path)}: FAILED ({type(e).__name__}: {e})")

    print(f"\nDone. CSVs in {out_dir}")
    return out_dir


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    convert_dir(in_dir, out_dir)


if __name__ == "__main__":
    main()