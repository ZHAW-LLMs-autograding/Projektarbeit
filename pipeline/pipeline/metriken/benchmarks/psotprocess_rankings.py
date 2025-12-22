#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# ----------------------------
# KONFIG (hier anpassen)
# ----------------------------

# Root kann entweder:
#  - ein "Sammler-Ordner" sein (z.B. benchmarks/, vectormodels/)
#  - oder direkt ein Modellordner (z.B. vectormodels/char_wb_3_5_min2/)
#
# Beispiel: Nur dieses eine Vektormodell verarbeiten:
ROOT_DIR = Path(".")

# Wenn ROOT_DIR ein Sammler-Ordner ist:
#   - False: nur direkte Subfolder werden als Modellordner betrachtet
#   - True: rekursiv alle Subfolder
#
# Wenn ROOT_DIR direkt ein Modellordner ist, spielt das hier keine Rolle:
RECURSIVE_SEARCH = True

# Input-Dateinamen pro Modellordner
INPUTS = [
    ("predictions_ranking.json", "predictions_ranking", "metrics"),
    ("predictions_ranking_clamped.json", "predictions_ranking_clamped", "metrics_clamped"),
]

# Postprocessing-Namen (werden in Dateinamen verwendet)
POST_NAMES = [
    "norm_equalwidth",
    "direct_nearest_bucket",
    "rank_uniform_bucket",
]

# Feldnamen in den JSON-Eintraegen
KEY_SOURCE = "source"
KEY_QUESTION = "question"
KEY_Y_TRUE = "y_true"
KEY_Y_PRED_IN = "y_pred"
KEY_RANK_TRUE = "ranking_true"

# Finale Note fuer Metriken (wird in *_processed.json geschrieben)
KEY_Y_PRED_POST = "y_pred_post"

# --------------------------------
# SciPy fuer Kendall tau-b
# --------------------------------
try:
    from scipy.stats import kendalltau  # type: ignore
except Exception as e:
    raise RuntimeError("SciPy fehlt. Installiere: pip install scipy") from e


# ---------------------------------------------------------------------
# Bucket-Definitionen (data_source,bucket,count)
# Wir brauchen hier nur die BUCKET-WERTE (sortiert).
# ---------------------------------------------------------------------

BUCKETS_CSV_TEXT = """data_source,bucket,count
Beetle,0.0,308
Beetle,0.33333,1049
Beetle,0.66667,919
Beetle,1.0,1665
CU-NLP,0.0,43
CU-NLP,0.2,39
CU-NLP,0.25,35
CU-NLP,0.4,11
CU-NLP,0.5,24
CU-NLP,0.6,6
CU-NLP,0.75,3
CU-NLP,0.8,5
CU-NLP,1.0,5
DigiKlausur,0.0,94
DigiKlausur,0.5,219
DigiKlausur,1.0,333
Mohler,0.0,13
Mohler,0.2,5
Mohler,0.3,11
Mohler,0.4,29
Mohler,0.5,30
Mohler,0.6,58
Mohler,0.7,69
Mohler,0.8,47
Mohler,0.9,82
Mohler,1.0,286
SAF,0.0,410
SAF,0.1,2
SAF,0.125,3
SAF,0.2,5
SAF,0.25,181
SAF,0.3,1
SAF,0.375,9
SAF,0.38,2
SAF,0.4,14
SAF,0.5,1011
SAF,0.6,22
SAF,0.625,53
SAF,0.63,15
SAF,0.68,11
SAF,0.7,8
SAF,0.75,824
SAF,0.8,41
SAF,0.875,124
SAF,0.88,110
SAF,0.9,11
SAF,1.0,2980
SciEntsBank,0.0,6374
SciEntsBank,0.33333,2641
SciEntsBank,0.66667,6286
SciEntsBank,1.0,10799
Stita,0.0,22
Stita,0.09091,1
Stita,0.2093,9
Stita,0.25581,1
Stita,0.29545,1
Stita,0.39535,12
Stita,0.40909,36
Stita,0.48837,6
Stita,0.5,7
Stita,0.59091,26
Stita,0.60465,26
Stita,0.69767,2
Stita,0.70455,22
Stita,0.7907,22
Stita,0.79545,56
Stita,0.90909,6
Stita,1.0,78
"""


@dataclass(frozen=True)
class BucketSpec:
    values: List[float]  # aufsteigend


def parse_bucket_specs_from_text(csv_text: str) -> Dict[str, BucketSpec]:
    lines = [ln.strip() for ln in csv_text.strip().splitlines() if ln.strip()]
    header = lines[0].split(",")
    if header[:3] != ["data_source", "bucket", "count"]:
        raise ValueError("Unerwarteter Bucket-Header: " + repr(header))

    tmp: Dict[str, set] = defaultdict(set)
    for ln in lines[1:]:
        ds, b, _c = ln.split(",")
        tmp[ds].add(float(b))

    out: Dict[str, BucketSpec] = {}
    for ds, s in tmp.items():
        out[ds] = BucketSpec(values=sorted(s))
    return out


BUCKET_SPECS: Dict[str, BucketSpec] = parse_bucket_specs_from_text(BUCKETS_CSV_TEXT)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.lower() == "nan":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def nearest_bucket_value(value: float, bucket_values: List[float]) -> float:
    best = bucket_values[0]
    best_d = abs(value - best)
    for b in bucket_values[1:]:
        d = abs(value - b)
        if d < best_d:
            best = b
            best_d = d
    return best


def dense_ranks_desc(values: List[float]) -> List[int]:
    """Dense ranking: hoeherer Wert = besser (Rang 1), ties -> gleicher Rang."""
    uniq = sorted(set(values), reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(uniq)}
    return [rank_of[v] for v in values]


def kendall_tau_b_global(r_true: List[int], r_pred: List[int]) -> float:
    if len(r_true) == 0:
        return float("nan")
    tau, _p = kendalltau(r_true, r_pred)  # tau-b bei ties
    return float(tau) if tau is not None else float("nan")


def compute_mae_rmse(y_true: List[float], y_pred: List[float]) -> Tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    abs_err = 0.0
    sq_err = 0.0
    for yt, yp in zip(y_true, y_pred):
        e = yp - yt
        abs_err += abs(e)
        sq_err += e * e
    return abs_err / n, math.sqrt(sq_err / n)


def compute_wrmse(items: List[Dict[str, Any]], y_true_key: str, y_pred_key: str) -> float:
    """
    wRMSE: Gewichtung nach Haeufigkeit des True-Buckets (pro data_source) in den Items selbst.
    true_bucket = nearest bucket value (data_source-spezifisch).
    weight_i = 1 / count(true_bucket).
    Normierung: sum(weights)=n.
    """
    counts: Dict[Tuple[str, float], int] = defaultdict(int)

    for it in items:
        ds = str(it.get(KEY_SOURCE, ""))
        spec = BUCKET_SPECS.get(ds)
        yt = safe_float(it.get(y_true_key))
        if spec is None or yt is None:
            continue
        tb = nearest_bucket_value(yt, spec.values)
        counts[(ds, tb)] += 1

    errs: List[float] = []
    ws: List[float] = []

    for it in items:
        ds = str(it.get(KEY_SOURCE, ""))
        spec = BUCKET_SPECS.get(ds)
        yt = safe_float(it.get(y_true_key))
        yp = safe_float(it.get(y_pred_key))
        if spec is None or yt is None or yp is None:
            continue
        tb = nearest_bucket_value(yt, spec.values)
        c = counts.get((ds, tb), 0)
        if c <= 0:
            continue
        ws.append(1.0 / float(c))
        errs.append(yp - yt)

    n = len(errs)
    if n == 0:
        return float("nan")
    sum_w = sum(ws)
    if sum_w <= 0:
        return float("nan")

    scale = n / sum_w
    w_sq = 0.0
    for e, w in zip(errs, ws):
        w_sq += (w * scale) * (e * e)
    return math.sqrt(w_sq / n)


def uniform_bucket_sizes(m: int, k: int) -> List[int]:
    base = m // k
    rest = m % k
    sizes = [base] * k
    for i in range(rest):
        sizes[i] += 1
    return sizes


# ---------------------------------------------------------------------
# Step 1: Postprocessings -> y_pred_<postname>
# ---------------------------------------------------------------------

def post_norm_equalwidth(items: List[Dict[str, Any]], pred_in: str, pred_out: str) -> None:
    vals_by_ds: Dict[str, List[float]] = defaultdict(list)
    for it in items:
        ds = str(it.get(KEY_SOURCE, ""))
        v = safe_float(it.get(pred_in))
        if v is not None:
            vals_by_ds[ds].append(v)

    minmax: Dict[str, Tuple[float, float]] = {}
    for ds, vs in vals_by_ds.items():
        if vs:
            minmax[ds] = (min(vs), max(vs))

    for it in items:
        ds = str(it.get(KEY_SOURCE, ""))
        spec = BUCKET_SPECS.get(ds)
        v = safe_float(it.get(pred_in))
        if spec is None or v is None:
            it[pred_out] = None
            continue

        mn, mx = minmax.get(ds, (v, v))
        norm = 0.5 if mx == mn else clamp01((v - mn) / (mx - mn))

        k = len(spec.values)
        idx = int(math.floor(norm * k))
        if idx >= k:
            idx = k - 1
        it[pred_out] = spec.values[idx]


def post_direct_nearest_bucket(items: List[Dict[str, Any]], pred_in: str, pred_out: str) -> None:
    for it in items:
        ds = str(it.get(KEY_SOURCE, ""))
        spec = BUCKET_SPECS.get(ds)
        v = safe_float(it.get(pred_in))
        if spec is None or v is None:
            it[pred_out] = None
            continue
        it[pred_out] = nearest_bucket_value(v, spec.values)


def post_rank_uniform_bucket(items: List[Dict[str, Any]], pred_in: str, pred_out: str) -> None:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        ds = str(it.get(KEY_SOURCE, ""))
        q = str(it.get(KEY_QUESTION, ""))
        groups[(ds, q)].append(idx)

    for (ds, _q), idxs in groups.items():
        spec = BUCKET_SPECS.get(ds)
        if spec is None:
            for i in idxs:
                items[i][pred_out] = None
            continue

        scored: List[Tuple[int, float]] = []
        for i in idxs:
            v = safe_float(items[i].get(pred_in))
            scored.append((i, v if v is not None else float("-inf")))
        scored.sort(key=lambda t: t[1], reverse=True)

        m = len(scored)
        k = len(spec.values)
        sizes = uniform_bucket_sizes(m, k)
        buckets_desc = list(reversed(spec.values))

        pos = 0
        for bval, sz in zip(buckets_desc, sizes):
            for _ in range(sz):
                if pos >= m:
                    break
                item_idx, _ = scored[pos]
                items[item_idx][pred_out] = bval
                pos += 1


POSTPROCS: Dict[str, Any] = {
    "norm_equalwidth": post_norm_equalwidth,
    "direct_nearest_bucket": post_direct_nearest_bucket,
    "rank_uniform_bucket": post_rank_uniform_bucket,
}


# ---------------------------------------------------------------------
# Step 2: Ranking aus y_pred_<postname>
# ---------------------------------------------------------------------

def rebuild_ranking_from_pred(items: List[Dict[str, Any]], pred_key: str, out_ranking_key: str) -> None:
    by_q: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        ds = str(it.get(KEY_SOURCE, ""))
        q = str(it.get(KEY_QUESTION, ""))
        by_q[(ds, q)].append(i)

    for (_ds, _q), idxs in by_q.items():
        vals: List[float] = []
        for i in idxs:
            v = safe_float(items[i].get(pred_key))
            vals.append(v if v is not None else float("-inf"))

        ranks = dense_ranks_desc(vals)
        for i, r in zip(idxs, ranks):
            items[i][out_ranking_key] = int(r)


# ---------------------------------------------------------------------
# Step 4 (korrigiert, wie besprochen):
# KEINE zusaetzliche Rank->Bucket Rueckuebersetzung.
# y_pred_post ist einfach die postprocessed Bucket-Note.
# ---------------------------------------------------------------------

def set_y_pred_post_from_postprocessing(items: List[Dict[str, Any]], y_pred_postproc_key: str) -> None:
    for it in items:
        it[KEY_Y_PRED_POST] = it.get(y_pred_postproc_key, None)


# ---------------------------------------------------------------------
# IO + Metrics
# ---------------------------------------------------------------------

def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} ist kein JSON-Array.")
    return data


def save_json_list(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_json_obj(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_tau(items: List[Dict[str, Any]], true_key: str, pred_key: str) -> float:
    r_true: List[int] = []
    r_pred: List[int] = []
    for it in items:
        rt = safe_float(it.get(true_key))
        rp = safe_float(it.get(pred_key))
        if rt is None or rp is None:
            continue
        r_true.append(int(rt))
        r_pred.append(int(rp))
    return kendall_tau_b_global(r_true, r_pred)


def compute_metrics(items: List[Dict[str, Any]], tau: float) -> Dict[str, Any]:
    y_true: List[float] = []
    y_pred: List[float] = []
    for it in items:
        yt = safe_float(it.get(KEY_Y_TRUE))
        yp = safe_float(it.get(KEY_Y_PRED_POST))
        if yt is None or yp is None:
            continue
        y_true.append(yt)
        y_pred.append(yp)

    mae, rmse = compute_mae_rmse(y_true, y_pred)
    wrmse = compute_wrmse(items, y_true_key=KEY_Y_TRUE, y_pred_key=KEY_Y_PRED_POST)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": wrmse,
        "tau_b": tau,
        "num_samples": len(y_true),
    }


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def _is_model_dir(p: Path) -> bool:
    return any((p / in_name).exists() for (in_name, _out_prefix, _m_prefix) in INPUTS)


def find_model_dirs(root: Path) -> List[Path]:
    """
    Wichtig: ROOT_DIR kann selbst ein Modellordner sein.
    Dann geben wir [ROOT_DIR] zurueck.
    """
    if root.exists() and root.is_dir() and _is_model_dir(root):
        return [root]

    model_dirs: List[Path] = []

    if not RECURSIVE_SEARCH:
        for p in root.iterdir():
            if not p.is_dir():
                continue
            if _is_model_dir(p):
                model_dirs.append(p)
        return sorted(model_dirs)

    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if _is_model_dir(p):
            model_dirs.append(p)
    return sorted(model_dirs)


def run_for_file(model_dir: Path, in_name: str, out_prefix: str, metrics_prefix: str) -> None:
    in_path = model_dir / in_name
    if not in_path.exists():
        return

    base_items = load_json_list(in_path)

    for post_name in POST_NAMES:
        fn = POSTPROCS[post_name]

        items: List[Dict[str, Any]] = json.loads(json.dumps(base_items))

        # Step 1: postprocessing -> y_pred_<postname>
        y_pred_postproc_key = f"y_pred_{post_name}"
        fn(items, KEY_Y_PRED_IN, y_pred_postproc_key)

        out_post_path = model_dir / f"{out_prefix}_{post_name}.json"
        save_json_list(out_post_path, items)

        # Step 2: ranking aus y_pred_<postname>
        ranking_key = f"ranking_pred_{post_name}"
        rebuild_ranking_from_pred(items, pred_key=y_pred_postproc_key, out_ranking_key=ranking_key)

        # Step 3: tau
        tau = compute_tau(items, true_key=KEY_RANK_TRUE, pred_key=ranking_key)

        # Step 4 (wie besprochen): y_pred_post = postprocessed Note
        set_y_pred_post_from_postprocessing(items, y_pred_postproc_key=y_pred_postproc_key)

        out_processed_path = model_dir / f"{out_prefix}_{post_name}_processed.json"
        save_json_list(out_processed_path, items)

        # Step 5: metrics auf y_pred_post
        metrics = compute_metrics(items, tau=tau)
        metrics_path = model_dir / f"{metrics_prefix}_{post_name}.json"
        save_json_obj(metrics_path, metrics)


def main() -> None:
    root = ROOT_DIR.resolve()
    model_dirs = find_model_dirs(root)

    if not model_dirs:
        print(f"Keine Modellordner gefunden unter: {root}")
        print("Tipp: Wenn ROOT_DIR direkt ein Modellordner ist, muessen dort die Input-Files liegen:")
        for in_name, _, _ in INPUTS:
            print(f"  - {in_name}")
        return

    processed = 0
    for md in model_dirs:
        for in_name, out_prefix, metrics_prefix in INPUTS:
            run_for_file(md, in_name=in_name, out_prefix=out_prefix, metrics_prefix=metrics_prefix)
        processed += 1

    print(f"Fertig. Modellordner verarbeitet: {processed}")
    for md in model_dirs:
        print(f"  - {md}")


if __name__ == "__main__":
    main()
