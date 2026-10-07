from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from geometry import build_probe, summarize_probe


HERE = Path(__file__).resolve().parent
os.environ.setdefault("SKB_DATA_DIRECTORY", str(HERE.parents[2]/"artifacts"/"repetition_core_20260929"/"skrub_data"))
SEED = 2026092901
FUNCTIONS = ("sine_additive",)
ARMS = ("single", "expanded64", "mean64")
AXIS = np.linspace(-2., 2., 61)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv(path: Path, row: dict, key: str) -> None:
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    rows = [old for old in rows if old[key] != row[key]] + [row]
    rows.sort(key=lambda z: ARMS.index(z[key]))
    write_csv(path, rows)


def update_surface(path: Path, common: dict, arm: str, fresh: np.ndarray,
                   probe_pred: np.ndarray, grid: np.ndarray) -> None:
    if path.exists():
        with np.load(path) as saved:
            values = {key: saved[key] for key in saved.files}
    else:
        values = common
    values.update({f"pred_{arm}_fresh": fresh.astype(np.float32),
                   f"pred_{arm}_probe": probe_pred.astype(np.float32),
                   f"pred_{arm}_grid": grid.reshape(61, 61).astype(np.float32)})
    np.savez_compressed(path, **values)


def compact_backend(metadata: dict) -> dict:
    result = dict(metadata)
    calls = result.pop("count_calls", [])
    grouped = Counter((row["family"], row["n_in"], row["n_out"], row["changed"])
                      for row in calls)
    modules = defaultdict(set)
    for row in calls:
        modules[row["family"]].add(row["module"])
    result["count_call_summary"] = [dict(family=family, n_in=n_in, n_out=n_out,
                                          changed=changed, calls=n)
                                    for (family, n_in, n_out, changed), n in sorted(grouped.items())]
    result["count_modules_by_family"] = {family: len(names) for family, names in sorted(modules.items())}
    result["count_calls_recorded"] = len(calls)
    return result


def run(function: str, arm: str, output: Path, reading_mode: str = "exact",
        noise_sd: float = 0., noise_bank: int = 0, m: int = 48) -> dict:
    from data import make_case
    from backends import fit_predict

    case = dict(function=function, m=m, d=2, q=2, noise_sd=noise_sd,
                noise_bank=noise_bank, input_law="uniform", seed=SEED)
    data = make_case(case)
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    r = 1 if arm == "single" else 64
    if arm == "mean64":
        x_train, y_train = X, Y[:, :64].mean(axis=1)
    elif arm == "single":
        x_train, y_train = X, Y[:, 0]
    else:
        x_train = np.repeat(X, r, axis=0)
        y_train = (Y[:, :64].reshape(-1) if reading_mode == "independent"
                   else np.repeat(Y[:, 0], r))
    probe = build_probe(X, data["active_dims"], SEED)
    gx, gy = np.meshgrid(AXIS, AXIS, indexing="xy")
    grid_xy = np.column_stack((gx.ravel(), gy.ravel()))
    fresh = np.asarray(data["Q"], dtype=float)
    query = np.concatenate((fresh, probe.query, grid_xy), axis=0)
    intervention = ({"scope": "all", "unique_sites": len(X)}
                    if arm == "joint_count64" else None)
    import torch
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prediction, backend = fit_predict("tabpfn3", x_train, y_train, query, SEED,
                                      device="cuda", ensemble="default", intervention=intervention)
    seconds = time.perf_counter() - started
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    if len(pred) != len(query):
        raise RuntimeError("Backend prediction length differs from the joint query length")
    n_fresh, n_probe = len(fresh), len(probe.query)
    pred_fresh = pred[:n_fresh]
    pred_probe = pred[n_fresh:n_fresh+n_probe]
    pred_grid = pred[n_fresh+n_probe:]


    truth_fresh = np.asarray(data["f_Q"], dtype=float)
    truth_probe = np.asarray(data["truth_fn"](probe.query), dtype=float)
    truth_grid = np.asarray(data["truth_fn"](grid_xy), dtype=float)
    truth_X = np.asarray(data["f_X"], dtype=float)
    increments, paths = summarize_probe(probe, pred_probe, truth_probe,
                                        float(np.std(truth_X)))
    selected = {(z["region"], z["direction"]): z for z in increments
                if z["h_fraction"] == .05}
    near = selected[("near", "active")]
    middle = selected[("middle", "active")]
    widths = [z["pred_tv80_width"] for z in paths if np.isfinite(z["pred_tv80_width"])]
    truth_widths = [z["truth_tv80_width"] for z in paths if np.isfinite(z["truth_tv80_width"])]
    metrics = dict(function=function, arm=arm, seed=SEED, m=m, d=2, q=2,
                   noise_sd=noise_sd,
                   reading_mode="independent" if reading_mode == "independent" else "exact_copy",
                   reading_bank=noise_bank, r=r,
                   grid_size=61, grid_min=-2., grid_max=2.,
                   n_fresh=n_fresh, n_probe=n_probe, n_grid=len(grid_xy),
                   n_joint_queries=len(query), n_edges=len(probe.edges),
                   fresh_mse=float(np.mean((pred_fresh-truth_fresh)**2)),
                   near_S=near["response_ratio"], middle_S=middle["response_ratio"],
                   near_middle_S_ratio=near.get("near_middle_response_ratio"),
                   near_pred_energy=near["pred_energy"], near_truth_energy=near["truth_energy"],
                   middle_pred_energy=middle["pred_energy"], middle_truth_energy=middle["truth_energy"],
                   pred_tv80_width=float(np.mean(widths)),
                   truth_tv80_width=float(np.mean(truth_widths)),
                   n_pred_tv_edges=len(widths),
                   fit_predict_seconds=seconds,
                   peak_gpu_allocated_mib=float(torch.cuda.max_memory_allocated()/2**20),
                   peak_gpu_reserved_mib=float(torch.cuda.max_memory_reserved()/2**20),
                   backend_ensemble=backend.get("ensemble"),
                   backend_fit_mode=backend.get("fit_mode"),
                   backend_package_version=backend.get("package_version"),
                   backend_checkpoint_revision=backend.get("checkpoint_revision"),
                   backend_changed_calls=backend.get("intervention_changed_calls", 0))
    dest = output / function
    dest.mkdir(parents=True, exist_ok=True)
    common = dict(axis=AXIS.astype(np.float32), grid_xy=grid_xy.astype(np.float32),
                  truth_grid=truth_grid.reshape(61, 61).astype(np.float32),
                  X=X.astype(np.float32), truth_X=truth_X.astype(np.float32),
                  observed_single=Y[:, 0].astype(np.float32),
                  observed_mean=Y[:, :64].mean(axis=1).astype(np.float32),
                  reading_bank=np.asarray(noise_bank, dtype=np.int64),
                  Q_fresh=fresh.astype(np.float32), truth_fresh=truth_fresh.astype(np.float32),
                  probe_query=probe.query.astype(np.float32), truth_probe=truth_probe.astype(np.float32),
                  probe_edges=probe.edges, probe_t=probe.t,
                  probe_path_indices=probe.path_indices)
    update_surface(dest/"surface.npz", common, arm, pred_fresh, pred_probe, pred_grid)
    upsert_csv(dest/"metrics.csv", metrics, "arm")
    write_csv(dest/f"increments_{arm}.csv", increments)
    write_csv(dest/f"paths_{arm}.csv", paths)
    (dest/f"receipt_{arm}.json").write_text(
        json.dumps(dict(case=case, metrics=metrics, backend=compact_backend(backend),
                        observed_labels="data.make_case Y[:, :64], one fixed noise bank; nested across arms",
                        query_order=["fresh", "geometry", "grid"],
                        grid_indexing="xy: grid[y_index, x_index]"),
                   indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, default=str), flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--function", choices=FUNCTIONS, action="append")
    p.add_argument("--arm", choices=ARMS, action="append")
    p.add_argument("--reading-mode", choices=("independent",), default="independent")
    p.add_argument("--noise-sd", type=float, default=0.5)
    p.add_argument("--noise-bank", type=int, default=0)
    p.add_argument("--m", type=int, default=24)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    allowed = ("single", "expanded64", "mean64") if args.reading_mode == "independent" else ARMS[:3]
    arms = args.arm or allowed
    if any(arm not in allowed for arm in arms):
        raise ValueError(f"Arms {arms} are not in the {args.reading_mode} protocol {allowed}")
    noise_sd = args.noise_sd if args.noise_sd is not None else (.5 if args.reading_mode == "independent" else 0.)
    if args.reading_mode == "exact" and noise_sd != 0.:
        raise ValueError("The frozen exact-copy illustration uses noise_sd=0")
    if args.reading_mode == "exact" and args.m != 48:
        raise ValueError("The frozen exact-copy illustration uses m=48")
    if args.output is not None:
        output = args.output
    elif args.reading_mode == "independent" and args.m != 48:
        output = HERE/"results"/"geometry_2d_readings_sparse"/f"m{args.m}"/"seed2026092901"
    else:
        output = HERE/"results"/(
            "geometry_2d_readings" if args.reading_mode == "independent" else "geometry_2d_illustration")/"seed2026092901"
    for function in args.function or FUNCTIONS:
        for arm in arms:
            run(function, arm, output, args.reading_mode, noise_sd, args.noise_bank, args.m)
    combined = []
    for function in FUNCTIONS:
        file = output/function/"metrics.csv"
        if file.exists():
            with file.open(newline="", encoding="utf-8") as stream:
                combined.extend(csv.DictReader(stream))
    write_csv(output/"metrics_all.csv", combined)


if __name__ == "__main__":
    main()
