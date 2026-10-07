
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, glob, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid              


def load(outdir):
    fs = sorted(glob.glob(os.path.join(outdir, "*_s*of*.csv")))
    if not fs:
        raise SystemExit("no result CSVs in " + outdir)
    d = pd.concat([pd.read_csv(f, dtype={"r": "string", "matched_to": "string", "truth": "string",
                                         "spacing": "string"}) for f in fs], ignore_index=True)
    n0 = len(d)
    d = d.drop_duplicates("fit_key", keep="first").reset_index(drop=True)
    return d, n0 - len(d), fs


def explode(d):
    out = []
    for _, row in d.iterrows():
        axes = str(row["axis"]).split(";")
        for A in axes:
            rr = row.copy()
            rr["axis"] = A
            rr["derived"] = ""
            if row["r_scheme"] != "equal":
                if A == "unequal":
                    out.append(rr)
                continue
            r = int(row["r"])
            allowed = grid.AXIS_R[A]
            if A == "unequal":
                if row["arm"] == "single":
                    out.append(rr)
                continue
            if r not in allowed:
                continue
            out.append(rr)
            if row["arm"] == "single":
                for r2 in allowed:
                    if r2 > 1:
                        x = rr.copy(); x["r"] = str(r2); x["derived"] = "single_broadcast"
                        x["r_total"] = r2 * int(row["m"]); out.append(x)
                        if float(row["sigma_or_sigmult"]) == 0.0:
                            y = x.copy(); y["arm"] = "mean"; y["derived"] = "noise0_mean"; out.append(y)
                for arm in ("expanded", "mean"):
                    x = rr.copy(); x["arm"] = arm; x["derived"] = "r1_identity"; out.append(x)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--manifest")
    p.add_argument("--write")
    a = p.parse_args()
    d, ndup, fs = load(a.outdir)
    print("read %d files, %d fits (%d duplicate keys dropped)" % (len(fs), len(d), ndup))
    man = a.manifest or os.path.join(a.outdir, "manifest.csv")
    if os.path.exists(man):
        m = pd.read_csv(man)
        exp = m.groupby("backend").n_fits.sum()
        got = d.assign(grp=d.backend.where(d.backend.isin(grid.TFMS), "classical")).groupby("grp").size()
        print(pd.DataFrame({"expected": exp, "done": got}).fillna(0).astype(int).to_string())
    long = explode(d)
    print("long table: %d rows; per axis:" % len(long))
    print(long.groupby(["axis", "family"]).size().unstack(fill_value=0).to_string())
    if a.write:
        long.to_csv(a.write, index=False, lineterminator="\n")
        print("wrote", a.write)


if __name__ == "__main__":
    main()
