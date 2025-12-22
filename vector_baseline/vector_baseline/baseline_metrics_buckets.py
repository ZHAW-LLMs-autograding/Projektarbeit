import os
import sys
import warnings
from contextlib import redirect_stdout
from textwrap import shorten

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ConstantInputWarning
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Config
# =========================
PATH = "/home/kevin/projektarbeit/trainingdata/asag_ranking/ASAG_2024_with_ranking.csv"
OUTDIR = "asag_outputs"
USE_COLOR = True
TOP_N_PRINT = None        # None = alle Zeilen je Frage anzeigen
MAX_QUESTIONS_PRINT = 1_000_000  # nichts mehr verstecken
EPS = 1e-12               # numeric epsilon für thresholds
DEFAULT_THRESHOLD = 0.5   # Fallback, falls kein brauchbarer positiver Normalwert vorhanden ist
SHOW_FULL_EXCERPTS = False  # kompakte Tabelle; True = volle Antworten

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# =========================
# Helpers
# =========================

def colorize(text, fg=None, bold=False, enable=True):
    if not enable or fg is None:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36, "grey": 90}
    parts = []
    if bold: parts.append("1")
    parts.append(str(codes.get(fg, 37)))
    return f"\x1b[{';'.join(parts)}m{text}\x1b[0m"

def tfidf_diag_similarity(ref_texts, stud_texts, vectorizer: TfidfVectorizer):
    texts = ref_texts + stud_texts
    X = vectorizer.fit_transform(texts)
    n = len(ref_texts)
    A = X[:n]
    B = X[n:]
    # TF-IDF (L2) -> Diagonalprodukt entspricht Cosine Similarity
    return A.multiply(B).sum(axis=1).A1

def prepare_dataframe(path):
    data = pd.read_csv(path)
    required = [
        "question","provided_answer","reference_answer","grade",
        "data_source","normalized_grade","weight","index","ranking",
    ]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    # Strings
    for col in ["reference_answer", "provided_answer", "question", "data_source"]:
        data[col] = data[col].fillna("")

    # Numerik
    data["normalized_grade"] = pd.to_numeric(data["normalized_grade"], errors="coerce")
    data["grade"] = pd.to_numeric(data["grade"], errors="coerce")
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce").fillna(1.0).astype(float)
    data["ranking"] = pd.to_numeric(data["ranking"], errors="coerce").astype(int)

    # nur valide Zeilen (grade muss vorhanden sein)
    data = data[pd.to_numeric(data["grade"], errors="coerce").notna()].copy()
    data["grade"] = data["grade"].astype(float)

    # ===== POSITIVER THRESHOLD pro data_source (robust) =====
    norm_sane = data["normalized_grade"].clip(lower=0.0, upper=1.0)
    pos_mask = (norm_sane > 0) & norm_sane.notna()
    if pos_mask.any():
        ds_pos_min = data.loc[pos_mask].groupby("data_source")[["data_source"]].apply(
            lambda g: float(norm_sane.loc[g.index].min())
        )
        global_pos_min = float(norm_sane.loc[pos_mask].min())
    else:
        ds_pos_min = pd.Series(dtype=float)
        global_pos_min = np.nan

    def map_threshold(ds):
        if ds in ds_pos_min.index:
            thr = float(ds_pos_min.loc[ds])
        elif np.isfinite(global_pos_min):
            thr = float(global_pos_min)
        else:
            thr = DEFAULT_THRESHOLD
        if not np.isfinite(thr) or thr <= 0.0 or thr > 1.0:
            thr = DEFAULT_THRESHOLD
        return thr

    data["_ds_min_threshold"] = data["data_source"].map(map_threshold).astype(float).clip(0.0, 1.0)
    return data

def _fmt_cell(val, fmt_int=False, width=6):
    if pd.isna(val):
        return " " * (width - 1) + "-"
    if fmt_int:
        try:
            return f"{int(val):>{width}d}"
        except Exception:
            return " " * (width - 1) + "-"
    return f"{val:>{width}.3f}"

# =========================
# Buckets & Evaluation (per data_source)
# =========================

def eval_with_ranking(df, vec):
    """
    Buckets werden *nur pro data_source* bestimmt (über alle Fragen):
      - Sammle alle vorhandenen Grades > 0 innerhalb derselben data_source.
      - 0 bleibt Floor (wenn sim < threshold_row -> predicted_grade = 0).
      - Sonst snap auf den nächstgelegenen DS-Bucket (Vergleich auf normalisierter Skala b/q_max).
      - rsim = predicted_grade / q_max (0..1) für Ranking.
    """
    sims = tfidf_diag_similarity(df["reference_answer"].tolist(), df["provided_answer"].tolist(), vec)
    base = df.copy()
    base["similarity"] = sims

    # ---- Buckets pro data_source (über alle Fragen) ----
    ds_to_grade_buckets = {}
    for ds, gds in base.groupby("data_source", sort=True):
        uniq = pd.unique(pd.to_numeric(gds["grade"], errors="coerce").dropna())
        uniq = sorted(float(x) for x in uniq if np.isfinite(x) and x > 0)
        ds_to_grade_buckets[ds] = uniq

    # globaler Fallback (falls eine data_source keine >0-Grades hat)
    global_pos_grades = sorted(
        float(x) for x in pd.unique(pd.to_numeric(base["grade"], errors="coerce").dropna())
        if np.isfinite(x) and x > 0
    )

    rows = []
    for q, g in base.groupby("question", sort=True):
        g = g.copy()
        g["true_rank"] = g["ranking"].astype(int)

        # Max Punkte der Frage
        q_max = float(np.nanmax(g["grade"])) if np.isfinite(np.nanmax(g["grade"])) else 1.0
        if q_max <= 0: q_max = 1.0

        pred_grades = []
        snapped_norm = []
        ds_buckets_norm_all_per_row = []
        ds_buckets_grade_all_per_row = []
        threshold_used_per_row = []
        n_buckets_per_row = []
        qmax_per_row = []
        ds_per_row = []

        for _, r in g.iterrows():
            sim = float(r["similarity"])
            ds = r["data_source"]
            threshold_row = float(r["_ds_min_threshold"]) if "_ds_min_threshold" in r else DEFAULT_THRESHOLD
            if not np.isfinite(threshold_row) or threshold_row <= 0.0 or threshold_row > 1.0:
                threshold_row = DEFAULT_THRESHOLD

            # DS-Buckets (voll) oder Fallback
            grade_buckets_all = ds_to_grade_buckets.get(ds, [])
            if not grade_buckets_all:
                grade_buckets_all = global_pos_grades
            if not grade_buckets_all:
                grade_buckets_all = [q_max]  # letzter Fallback

            # eligible (>= Threshold, normalisiert)
            eligible_grade_buckets = [b for b in grade_buckets_all if (b / q_max) + EPS >= threshold_row]

            # Snap / Floor
            if sim + EPS < threshold_row or len(eligible_grade_buckets) == 0:
                bgrade = 0.0
                bnorm = 0.0
            else:
                norm_candidates = np.array([b / q_max for b in eligible_grade_buckets], dtype=float)
                nearest_idx = int(np.argmin(np.abs(norm_candidates - sim)))
                bgrade = float(eligible_grade_buckets[nearest_idx])
                bnorm = float(norm_candidates[nearest_idx])

            pred_grades.append(bgrade)
            snapped_norm.append(bnorm)

            # Für die Kopfzeile (volle DS-Buckets, ohne Floor)
            ds_buckets_norm_all = [b / q_max for b in grade_buckets_all]
            ds_buckets_grade_all = grade_buckets_all

            ds_buckets_norm_all_per_row.append(", ".join(f"{b:.2f}" for b in ds_buckets_norm_all) if ds_buckets_norm_all else "")
            ds_buckets_grade_all_per_row.append(", ".join(f"{b:.2f}" for b in ds_buckets_grade_all) if ds_buckets_grade_all else "")
            threshold_used_per_row.append(threshold_row)
            n_buckets_per_row.append(len(grade_buckets_all))  # ohne Floor
            qmax_per_row.append(q_max)
            ds_per_row.append(ds)

        g["predicted_grade"] = np.array(pred_grades)
        g["rsim"] = np.array(snapped_norm)  # 0..1

        # Ränge nach rsim (absteigend)
        unique_rsim = np.array(sorted(np.unique(g["rsim"]), reverse=True))
        rsim_to_rank = {v: i + 1 for i, v in enumerate(unique_rsim)}
        g["rpred"] = [rsim_to_rank[v] for v in g["rsim"]]

        # Felder für Anzeige (per Zeile)
        g["_ds_buckets_norm_all"] = ds_buckets_norm_all_per_row
        g["_ds_buckets_grade_all"] = ds_buckets_grade_all_per_row
        g["_threshold_used"] = threshold_used_per_row
        g["_n_buckets"] = n_buckets_per_row
        g["_q_max"] = qmax_per_row
        g["_data_source"] = ds_per_row

        for _, r in g.iterrows():
            rows.append(
                {
                    "question": q,
                    "data_source": r["_data_source"],
                    "reference_excerpt": str(r["reference_answer"])[:120],
                    "student_excerpt": str(r["provided_answer"]),
                    "true_rank": int(r["true_rank"]),
                    "rpred": int(r["rpred"]),
                    "similarity": float(r["similarity"]),
                    "rsim": float(r["rsim"]),
                    "grade": float(r["grade"]),
                    "predicted_grade": float(r["predicted_grade"]),
                    "weight": float(r["weight"]),
                    "_ds_buckets_norm_all": r["_ds_buckets_norm_all"],
                    "_ds_buckets_grade_all": r["_ds_buckets_grade_all"],
                    "_threshold_used": r["_threshold_used"],
                    "_n_buckets": r["_n_buckets"],
                    "_q_max": r["_q_max"],
                }
            )

    all_rows = pd.DataFrame(rows)
    return all_rows

# =========================
# Metrics
# =========================

def weighted_squared_error(row):
    return row["weight"] * (row["predicted_grade"] - row["grade"]) ** 2

def compute_rmse(df):
    if df.empty:
        return np.nan
    return float(np.sqrt(np.mean((df["predicted_grade"] - df["grade"]) ** 2)))

def compute_wrmse(df):
    if df.empty:
        return np.nan
    wse = df.apply(weighted_squared_error, axis=1)
    wsum = df["weight"].sum()
    return float(np.sqrt(wse.sum() / wsum)) if wsum > 0 else np.nan

def compute_per_question_metrics(all_rows):
    metrics = []
    for q, g in all_rows.groupby("question", sort=True):
        if len(g) < 2:
            metrics.append({"question": q, "n": len(g), "tau": np.nan, "rmse": np.nan, "wrmse": np.nan})
            continue
        tau, _ = kendalltau(g["rpred"], g["true_rank"])
        metrics.append({
            "question": q,
            "n": len(g),
            "tau": tau,
            "rmse": compute_rmse(g),
            "wrmse": compute_wrmse(g)
        })
    return pd.DataFrame(metrics)

# =========================
# Pretty printing (schlank, DS-Buckets oben)
# =========================

def print_question_block(df_q_full, q, use_color=USE_COLOR, max_excerpt=90):
    df_q_full = df_q_full.copy()
    df_q_disp = df_q_full.sort_values(["rsim", "similarity"], ascending=[False, False])

    line = "=" * 130
    print(line)
    print(colorize(f"QUESTION: {q}", fg="cyan", bold=True, enable=use_color))
    ref = df_q_disp["reference_excerpt"].iloc[0] if not df_q_disp.empty else ""
    print(colorize("Ref:", fg="grey", enable=use_color), shorten(ref, width=105, placeholder="…"))
    print("-" * 130)

    # q_max ist pro Frage konstant
    q_max = float(df_q_disp["_q_max"].iloc[0]) if not df_q_disp.empty else 1.0

    # === DS-Buckets (voll) pro data_source einmalig ausgeben ===
    print(colorize("Buckets per data_source (normalized | grade-scale):", fg="blue", bold=True, enable=use_color))
    for ds, grp in df_q_disp.groupby("data_source", sort=False):
        row0 = grp.iloc[0]
        ds_norm_all = row0["_ds_buckets_norm_all"] or ""
        ds_grade_all = row0["_ds_buckets_grade_all"] or ""
        nb = int(row0["_n_buckets"])
        print(colorize(f"  - {ds}: n={nb} | norm: {ds_norm_all} | grade (max={q_max:.2f}): {ds_grade_all}", fg="blue", enable=use_color))

    print("-" * 130)

    # kompakte Tabellenspalten, inkl. ds
    header = f"{'#':>2}  {'sim':>6}  {'rsim':>6}  {'grade':>6}  {'p.grade':>7}  {'true_r':>6}  {'rpred':>6}  {'Δr':>4}  {'ds':<12}  student_excerpt"
    print(colorize(header, fg="yellow", bold=True, enable=use_color))
    print("-" * 130)

    rows_iter = df_q_disp.iterrows()
    if TOP_N_PRINT is not None:
        rows_iter = list(rows_iter)[:TOP_N_PRINT]

    for i, (_, r) in enumerate(rows_iter, start=1):
        dr = int(r["rpred"]) - int(r["true_rank"])
        fg = "green" if dr == 0 else "yellow" if abs(dr) <= 2 else "red"
        excerpt = r["student_excerpt"] if SHOW_FULL_EXCERPTS else shorten(r["student_excerpt"], width=max_excerpt, placeholder='…')
        ds = r["data_source"] if isinstance(r["data_source"], str) else str(r["data_source"])
        row = (
            f"{i:>2}  {r['similarity']:>6.3f}  {r['rsim']:>6.3f}  {r['grade']:>6.2f}  {r['predicted_grade']:>7.2f}  "
            f"{int(r['true_rank']):>6}  {int(r['rpred']):>6}  {dr:+4}  {ds:<12.12}  {excerpt}"
        )
        print(colorize(row, fg=fg, enable=use_color))

    print(line)
    m = compute_per_question_metrics(df_q_full).iloc[0]
    print(colorize(f"Summary: τ={_fmt_cell(m.tau)}  RMSE={_fmt_cell(m.rmse)}  WRMSE={_fmt_cell(m.wrmse)}", fg="magenta", enable=use_color))
    print(line)

def write_full_pretty_output(all_rows, path_txt):
    """
    Schreibt die vollständige Pretty-Ansicht (alle Fragen) in eine Textdatei – ohne ANSI-Farben.
    """
    os.makedirs(os.path.dirname(path_txt), exist_ok=True)
    with open(path_txt, "w", encoding="utf-8") as f, redirect_stdout(f):
        print("=== Full Pretty per-question view (no ANSI colors) ===\n")
        for q, grp in all_rows.groupby("question", sort=True):
            print_question_block(grp, q, use_color=False)

# =========================
# Main
# =========================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Sanity-Checks
    raw_len = len(pd.read_csv(PATH))
    print(colorize(f"\n=== Sanity Checks ===", fg="cyan", bold=True, enable=USE_COLOR))
    print(colorize(f"Raw CSV rows:           {raw_len}", fg="cyan", enable=USE_COLOR))

    df = prepare_dataframe(PATH)
    print(colorize(f"Rows after grade filter: {len(df)}", fg="cyan", enable=USE_COLOR))

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    all_rows = eval_with_ranking(df, vec)
    print(colorize(f"Rows in all_rows:        {len(all_rows)}", fg="cyan", enable=USE_COLOR))

    # Export: all_rows + metrics
    all_rows_csv = os.path.join(OUTDIR, "all_rows.csv")
    all_rows.to_csv(all_rows_csv, index=False)
    metrics = compute_per_question_metrics(all_rows)
    metrics_csv = os.path.join(OUTDIR, "metrics.csv")
    metrics.to_csv(metrics_csv, index=False)

    sum_n = int(metrics["n"].sum(skipna=True)) if not metrics.empty else 0
    print(colorize(f"Sum of per-question n:   {sum_n}", fg="cyan", enable=USE_COLOR))

    # Terminal: alle Fragen anzeigen
    print("\n=== Per-Question Metrics ===")
    hdr = f"{'Question':30s}  {'n':>3}  {'tau':>6}  {'RMSE':>8}  {'WRMSE':>8}"
    print(colorize(hdr, fg="yellow", bold=True, enable=USE_COLOR))
    for r in metrics.itertuples(index=False):
        color = "green" if (r.tau or 0) >= 0.7 else "yellow" if (r.tau or 0) >= 0.4 else "red"
        print(colorize(f"{r.question[:30]:30s}  {r.n:>3}  {r.tau:>6.3f}  {r.rmse:>8.3f}  {r.wrmse:>8.3f}", fg=color, enable=USE_COLOR))

    print("\n=== Pretty per-question view (per-DS buckets) ===")
    total_questions = all_rows["question"].nunique()
    for i, (q, grp) in enumerate(all_rows.groupby("question", sort=True), start=1):
        if i > MAX_QUESTIONS_PRINT:
            hidden = total_questions - MAX_QUESTIONS_PRINT
            if hidden > 0:
                print(colorize(f"... ({hidden} more questions hidden in terminal)", fg="grey", enable=USE_COLOR))
            break
        print_question_block(grp, q, use_color=USE_COLOR, max_excerpt=90)

    # Vollständige Pretty-Ansicht in Datei (ohne ANSI)
    full_txt_path = os.path.join(OUTDIR, "full_output.txt")
    write_full_pretty_output(all_rows, full_txt_path)

    print("\n=== Global Summary ===")
    print(colorize(
        f"Global RMSE={compute_rmse(all_rows):.3f}  WRMSE={compute_wrmse(all_rows):.3f}  Avg τ={metrics['tau'].mean():.3f}",
        fg="magenta", bold=True, enable=USE_COLOR
    ))

    print(colorize(f"\nSaved: {metrics_csv}", fg="cyan", enable=USE_COLOR))
    print(colorize(f"Saved: {all_rows_csv}", fg="cyan", enable=USE_COLOR))
    print(colorize(f"Saved full pretty output: {full_txt_path}", fg="cyan", enable=USE_COLOR))

if __name__ == "__main__":
    main()
