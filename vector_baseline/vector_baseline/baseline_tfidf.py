#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASAG – TF-IDF Similarity + Ranking-Alignment (Single-File Version)
-------------------------------------------------------------------
- Lädt den Datensatz
- Sweep über mehrere TF-IDF-Konfigurationen (inkl. char_wb 3–5, min_df=2)
- Berechnet:
    * Globale & per-Frage Korrelationen (Similarity vs. normalized_grade)
    * Ranking-Pipeline wie in Skript 2:
        - Zentroid-Snapping (rsim) pro Frage
        - rpred-Ränge aus rsim (dicht; gleiche rsim -> gleicher Rang)
        - Per-Frage Ranking-Metriken: Kendall-τ, Spearman ρ (auf Rängen),
          MARE, Pairwise-Accuracy, Top-k-Accuracy, Top1-Rate
    * CSV-Outputs pro Konfiguration
    * Optional: Pretty-Ansicht pro Frage (TRUE→rpred→sim sortiert)
- Ground-Truth:
    * GT_MODE="auto": wenn 'ranking' vorhanden & vollständig -> nutze 'ranking'
                      sonst dichte Ränge aus 'normalized_grade' (beste Note Rang 1)
    * GT_MODE="ranking" oder "grade" erzwingen möglich

Tough love: Wenn die Metriken schlecht sind, liegt’s nicht an diesem Skript,
sondern an deiner Repräsentation. Dreh an Vectorizer & Preprocessing. 🚀
"""

import os
import warnings
from textwrap import shorten
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, ConstantInputWarning
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Config
# =========================
PATH = "/home/kevin/projektarbeit/trainingdata/asag_ranking/ASAG_2024_with_ranking.csv"
OUTDIR = "asag_outputs_aligned"
TOPK = 3
PRINT_PRETTY = True     # Pretty-Ausgabe pro Frage
USE_COLOR = True        # ANSI-Farben für die Konsole
GT_MODE = "auto"        # "auto" | "ranking" | "grade"
TOP_N_PRINT = None      # None = alle Zeilen je Frage anzeigen

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# =========================
# Helpers (UI)
# =========================

def colorize(text, fg=None, bold=False, enable=True):
    if not enable or fg is None:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36, "grey": 90, "white": 37}
    parts = []
    if bold:
        parts.append("1")
    parts.append(str(codes.get(fg, 37)))
    return f"\x1b[{';'.join(parts)}m{text}\x1b[0m"

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
# Core helpers
# =========================

def fisher_z_mean(rs, ns=None, weighted=True):
    """
    Fisher-z gemitteltes r (robust gegen Varianz der r-Schätzungen).
    """
    rs = np.array([r for r in rs if np.isfinite(r) and -0.999999 < r < 0.999999], dtype=float)
    if rs.size == 0:
        return np.nan, 0
    zs = np.arctanh(rs)
    if weighted and ns is not None:
        ns = np.array([n for n in ns if n is not None], dtype=float)
        if ns.size == zs.size:
            w = np.maximum(ns - 3.0, 1.0)
            zbar = np.average(zs, weights=w)
        else:
            zbar = np.mean(zs)
    else:
        zbar = np.mean(zs)
    rbar = np.tanh(zbar)
    return float(rbar), int(rs.size)

def tfidf_diag_similarity(ref_texts, stud_texts, vectorizer: TfidfVectorizer):
    """
    Cosine-Ähnlichkeit entlang der Diagonalen (gleiche Indexpaare), da TF-IDF L2-normalisiert.
    """
    texts = ref_texts + stud_texts
    X = vectorizer.fit_transform(texts)
    n = len(ref_texts)
    A = X[:n]
    B = X[n:]
    return A.multiply(B).sum(axis=1).A1

def safe_corr(x, y, method="pearson"):
    """Return (r, p, n). Bei n<3 oder Varianz=0 -> (nan, nan, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3 or np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return np.nan, np.nan, int(n)
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p), int(n)

def prepare_dataframe(path):
    data = pd.read_csv(path)
    # Pflichtspalten absichern
    required_base = ["question", "provided_answer", "reference_answer", "normalized_grade"]
    for col in required_base:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")
    # Robustheit
    for col in ["reference_answer", "provided_answer", "question"]:
        data[col] = data[col].fillna("")
    data = data[pd.to_numeric(data["normalized_grade"], errors="coerce").notna()].copy()
    data["normalized_grade"] = data["normalized_grade"].astype(float)

    # Optionale Ranking-Spalte typisieren, wenn vorhanden
    if "ranking" in data.columns:
        data["ranking"] = pd.to_numeric(data["ranking"], errors="coerce")
    return data

def add_true_rank(df, mode="auto"):
    """
    Legt 'true_rank' pro Zeile an.
    - mode='ranking': zwingt Nutzung von df['ranking']
    - mode='grade'  : erzeugt Ränge aus normalized_grade (beste Note -> Rang 1, dichte Ränge)
    - mode='auto'   : nimmt 'ranking', wenn komplett vorhanden (keine NaNs), sonst grade->rank
    """
    df = df.copy()
    has_full_ranking = ("ranking" in df.columns) and df["ranking"].notna().all()
    use_ranking = (mode == "ranking") or (mode == "auto" and has_full_ranking)

    if use_ranking:
        df["true_rank"] = df["ranking"].astype(int)
        df["gt_type"] = "ranking"
    else:
        grades_desc = sorted(df["normalized_grade"].dropna().unique(), reverse=True)
        grade2rank = {g: i + 1 for i, g in enumerate(grades_desc)}
        df["true_rank"] = df["normalized_grade"].map(grade2rank).astype(int)
        df["gt_type"] = "grade→rank"
    return df

# =========================
# Ranking-Evaluation (aligned mit Skript 2)
# =========================

def eval_with_ranking(df, vec, gt_mode="auto"):
    """
    - Berechnet Similarity
    - Snapped Similarity -> rsim über Noten-Zentroiden (Median der Similarities je Note, pro Frage)
    - rpred (dichte Ränge) aus rsim
    - Gibt pro Frage sortierte Rows + per-Frage Ranking-Metriken + globale Corrs zurück
    """
    sims = tfidf_diag_similarity(df["reference_answer"].tolist(), df["provided_answer"].tolist(), vec)
    base = df.copy()
    base["similarity"] = sims

    # Globale Korrelationen (roh, nur Info)
    try:
        pearson_r, _ = pearsonr(base["similarity"], base["normalized_grade"])
    except Exception:
        pearson_r = np.nan
    try:
        spearman_r, _ = spearmanr(base["similarity"], base["normalized_grade"])
    except Exception:
        spearman_r = np.nan

    rows = []
    for q, grp in base.groupby("question", sort=True):
        g = add_true_rank(grp, mode=gt_mode).copy()

        # Zentroiden (Median Similarity je NOTE) -> absteigend sortiert (bessere Note höher)
        centroids = g.groupby("normalized_grade")["similarity"].median().sort_values(ascending=False)
        cvals = centroids.values.astype(float)
        simv = g["similarity"].values.astype(float)

        if len(cvals) == 1:
            rsim = np.full(len(g), float(cvals[0]), dtype=float)
        else:
            dists = np.abs(simv[:, None] - cvals[None, :])
            rsim = cvals[np.argmin(dists, axis=1)]
        g["rsim"] = rsim

        # rpred: dichte Ränge aus rsim (gleiches rsim -> gleicher Rang), absteigend
        unique_rsim = np.array(sorted(np.unique(rsim), reverse=True))
        rsim_to_rank = {v: i + 1 for i, v in enumerate(unique_rsim)}
        g["rpred"] = [int(rsim_to_rank[v]) for v in rsim]

        # Informativ: Rang aus roher Similarity (dicht, absteigend)
        unique_sim = np.array(sorted(np.unique(simv), reverse=True))
        sim_to_rank = {v: i + 1 for i, v in enumerate(unique_sim)}
        g["pred_rank"] = [int(sim_to_rank[v]) for v in simv]

        for _, r in g.iterrows():
            rows.append(
                {
                    "question": q,
                    "reference_excerpt": str(r["reference_answer"])[:120],
                    "student_excerpt": str(r["provided_answer"])[:200],
                    "true_rank": int(r["true_rank"]),
                    "pred_rank": int(r["pred_rank"]),
                    "rpred": int(r["rpred"]),
                    "similarity": float(r["similarity"]),
                    "rsim": float(r["rsim"]),
                    "grade": float(r["normalized_grade"]),
                }
            )

    all_rows = pd.DataFrame(rows).sort_values(
        ["question", "true_rank", "rpred", "similarity"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )

    # Per-Question Ranking-Metriken (gegen rpred vs true_rank)
    metrics = []
    for q, g in all_rows.groupby("question", sort=True):
        n = len(g)
        if n < 2:
            metrics.append(
                {"question": q, "n": int(n), "tau": np.nan, "rho": np.nan, "mare": np.nan,
                 "pair_acc": np.nan, "discordant": 0, "ties": 0, "pairs_total": 0,
                 "swap_lb": 0, "topk_acc": np.nan, "top1_match": np.nan}
            )
            continue

        # Rang-Korrelationen
        try:
            tau, _ = kendalltau(g["rpred"], g["true_rank"])
        except Exception:
            tau = np.nan
        try:
            rho, _ = spearmanr(g["rpred"], g["true_rank"])
        except Exception:
            rho = np.nan

        # Rank-Fehler
        mare = float(np.mean(np.abs(g["rpred"] - g["true_rank"])))

        # Pairwise-Accuracy (ohne Ties)
        rp = -g["rpred"].values.astype(float)
        rt = -g["true_rank"].values.astype(float)
        C = D = T = 0
        m = len(rt)
        for i in range(m):
            for j in range(i + 1, m):
                ds = np.sign(rp[i] - rp[j])
                dt = np.sign(rt[i] - rt[j])
                if dt == 0 or ds == 0:
                    T += 1
                elif ds == dt:
                    C += 1
                else:
                    D += 1
        pair_acc = C / (C + D) if (C + D) > 0 else np.nan
        swap_lb = D
        pairs_total = (n * (n - 1)) // 2

        # Top-k & Top-1 (gegen rpred)
        k = min(TOPK, n)
        top_true_ids = set(g.sort_values("true_rank").head(k).index)
        top_pred_ids = set(g.sort_values("rpred").head(k).index)
        topk_acc = len(top_true_ids & top_pred_ids) / k if k > 0 else np.nan
        top1_match = 1 if g.sort_values("true_rank").index[0] in top_pred_ids else 0

        metrics.append(
            {
                "question": q,
                "n": int(n),
                "tau": float(tau) if tau is not None else np.nan,
                "rho": float(rho) if rho is not None else np.nan,
                "mare": float(mare),
                "pair_acc": float(pair_acc) if pair_acc is not None else np.nan,
                "discordant": int(D),
                "ties": int(T),
                "pairs_total": int(pairs_total),
                "swap_lb": int(swap_lb),
                "topk_acc": float(topk_acc) if topk_acc is not None else np.nan,
                "top1_match": int(top1_match),
            }
        )
    metrics_df = pd.DataFrame(metrics)

    # Globale Ranking-Zusammenfassung (über Fragen gemittelt)
    valid = metrics_df.dropna(subset=["tau", "rho", "pair_acc"])
    summary_rank = {
        "avg_tau": float(valid["tau"].mean()) if not valid.empty else np.nan,
        "avg_rho": float(valid["rho"].mean()) if not valid.empty else np.nan,
        "avg_mare": float(metrics_df["mare"].mean()) if not metrics_df.empty else np.nan,
        "avg_pairacc": float(valid["pair_acc"].mean()) if not valid.empty else np.nan,
        "avg_topk": float(metrics_df["topk_acc"].mean()) if not metrics_df.empty else np.nan,
        "top1_rate": float(metrics_df["top1_match"].mean()) if not metrics_df.empty else np.nan,
    }

    return {
        "corr_global": {"pearson_r": pearson_r, "spearman_r": spearman_r},
        "rows": all_rows,
        "per_question_metrics": metrics_df,
        "rank_summary": summary_rank,
    }

# =========================
# Pretty printing (pro Frage)
# =========================

def print_question_block(df_q_full, q, top_n=TOP_N_PRINT, use_color=USE_COLOR, max_excerpt=90):
    df_q_full = df_q_full.copy()
    df_q_full["true_rank"] = pd.to_numeric(df_q_full["true_rank"], errors="coerce").astype(int)
    df_q_full["rpred"] = pd.to_numeric(df_q_full["rpred"], errors="coerce").astype(int)
    df_q_full["pred_rank"] = pd.to_numeric(df_q_full["pred_rank"], errors="coerce").astype(int)

    # Sortierung: TRUE → rpred → sim (stabil)
    df_q_disp = df_q_full.sort_values(
        ["true_rank", "rpred", "similarity"],
        ascending=[True, True, False],
        kind="mergesort",
    )

    if isinstance(top_n, int) and top_n > 0:
        df_q_disp = df_q_disp.head(top_n)

    line = "=" * 120
    print(line)
    print(colorize(f"QUESTION: {q}", fg="cyan", bold=True, enable=use_color))
    ref = df_q_disp["reference_excerpt"].iloc[0] if not df_q_disp.empty else ""
    print(colorize("Ref:", fg="grey", enable=use_color), shorten(str(ref), width=105, placeholder="…"))

    print("-" * 120)
    header = f"{'#':>2}  {'true':>4}  {'pred':>4}  {'rpred':>5}  {'Δr':>3}  {'sim':>6}  {'rsim':>6}  {'grade':>5}  student_excerpt"
    print(colorize(header, fg="yellow", bold=True, enable=use_color))
    print("-" * 120)

    for i, r in enumerate(df_q_disp.itertuples(index=False), start=1):
        dr = int(r.rpred) - int(r.true_rank)
        if dr == 0:
            fg = "green"
        elif abs(dr) <= 2:
            fg = "yellow"
        else:
            fg = "red"
        row = (
            f"{i:>2}  "
            f"{int(r.true_rank):>4}  "
            f"{int(r.pred_rank):>4}  "
            f"{int(r.rpred):>5}  "
            f"{dr:>+3}  "
            f"{getattr(r, 'similarity', np.nan):>6.3f}  "
            f"{getattr(r, 'rsim', np.nan):>6.3f}  "
            f"{getattr(r, 'grade', np.nan):>5.2f}  "
            f"{shorten(str(getattr(r, 'student_excerpt', '')), width=max_excerpt, placeholder='…')}"
        )
        print(colorize(row, fg=fg, enable=use_color))
    print(line)

    # Summary (gegen rpred)
    mdf = compute_per_question_metrics(df_q_full, topk=TOPK)
    mr = mdf.iloc[0]
    grades = sorted(df_q_full["grade"].dropna().unique())
    by_grade = df_q_full.groupby("grade")["similarity"].mean().to_dict()

    print("-" * 120)
    stats_line1 = (
        f"Summary:  τ={_fmt_cell(mr.tau)}  ρ={_fmt_cell(mr.rho)}  MARE={mr.mare:>5.2f}  "
        f"PairAcc={_fmt_cell(mr.pair_acc)}  Swaps↓={_fmt_cell(mr.swap_lb, fmt_int=True, width=3)}  "
        f"Top{TOPK}={_fmt_cell(mr.topk_acc, width=5)}  Top1={_fmt_cell(mr.top1_match, fmt_int=True, width=2)}"
    )
    print(colorize(stats_line1, fg="magenta", enable=use_color))

    stats_line2 = "Grades: [" + ", ".join(f"{g:.2f}" for g in grades) + "]  Avg sim per grade: "
    pairs = [f"{g:.2f}:{by_grade.get(g, np.nan):.3f}" for g in grades]
    print(colorize(stats_line2 + "; ".join(pairs), fg="grey", enable=use_color))
    print(line)

def compute_per_question_metrics(all_rows: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    metrics = []
    for q, g in all_rows.groupby("question", sort=True):
        g = g.copy()
        n = len(g)
        if n < 2:
            metrics.append(
                {
                    "question": q,
                    "n": int(n),
                    "tau": np.nan,
                    "rho": np.nan,
                    "mare": np.nan,
                    "pair_acc": np.nan,
                    "discordant": 0,
                    "ties": 0,
                    "pairs_total": 0,
                    "swap_lb": 0,
                    "topk_acc": np.nan,
                    "top1_match": np.nan,
                }
            )
            continue

        try:
            tau, _ = kendalltau(g["rpred"], g["true_rank"])
        except Exception:
            tau = np.nan
        try:
            rho, _ = spearmanr(g["rpred"], g["true_rank"])
        except Exception:
            rho = np.nan

        mare = float(np.mean(np.abs(g["rpred"] - g["true_rank"])))

        # Pairwise-Accuracy (ohne Ties)
        rp = -g["rpred"].to_numpy(float)
        rt = -g["true_rank"].to_numpy(float)
        C = D = T = 0
        for i in range(n):
            for j in range(i + 1, n):
                ds = np.sign(rp[i] - rp[j])
                dt = np.sign(rt[i] - rt[j])
                if ds == 0 or dt == 0:
                    T += 1
                elif ds == dt:
                    C += 1
                else:
                    D += 1
        pair_acc = C / (C + D) if (C + D) > 0 else np.nan
        swap_lb = D

        k = min(topk, n)
        top_true = set(g.sort_values("true_rank").head(k).index)
        top_pred = set(g.sort_values("rpred").head(k).index)
        topk_acc = len(top_true & top_pred) / k if k > 0 else np.nan
        top1_match = 1 if g.sort_values("true_rank").index[0] in top_pred else 0

        metrics.append(
            {
                "question": q,
                "n": int(n),
                "tau": float(tau) if tau is not None else np.nan,
                "rho": float(rho) if rho is not None else np.nan,
                "mare": float(mare),
                "pair_acc": float(pair_acc) if pair_acc is not None else np.nan,
                "discordant": int(D),
                "ties": int(T),
                "pairs_total": int((n * (n - 1)) // 2),
                "swap_lb": int(swap_lb),
                "topk_acc": float(topk_acc) if topk_acc is not None else np.nan,
                "top1_match": int(top1_match),
            }
        )
    return pd.DataFrame(metrics)

# =========================
# Summary-Formatter
# =========================

def format_corr_summary(corr_summary, by_q_corr, rank_summary, name):
    g = corr_summary
    bq = by_q_corr
    rs = rank_summary
    return (
        f"{name:20s}  "
        f"global r(Pearson)={g['pearson_r']:6.3f} (vs. grade)  "
        f"global ρ(Spearman)={g['spearman_r']:6.3f}  |  "
        f"mean r_by_q={bq['mean_pearson_r_simple']:6.3f}  Fisher-w={bq['fisher_pearson_weighted']:6.3f}  ||  "
        f"τ̄={rs['avg_tau']:5.3f}  PairAcc̄={rs['avg_pairacc']:5.3f}  Top{TOPK}̄={rs['avg_topk']:5.3f}  Top1̄={rs['top1_rate']:5.3f}"
    )

# =========================
# Main
# =========================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = prepare_dataframe(PATH)

    # ---- TF-IDF-Konfigurationen (inkl. char_wb 3–5 min_df=2 wie in Skript 2) ----
    configs = OrderedDict({
        "baseline_unigram": TfidfVectorizer(),
        "unigram_stop": TfidfVectorizer(stop_words='english'),
        "uni_bigram": TfidfVectorizer(ngram_range=(1,2)),
        "uni_bigram_stop": TfidfVectorizer(stop_words='english', ngram_range=(1,2)),
        "unigram_sublinear": TfidfVectorizer(sublinear_tf=True),
        "char_wb_3_5": TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5)),
        "char_wb_3_5_min2": TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2),  # EMPFOHLEN
        "char_wb_4_6": TfidfVectorizer(analyzer='char_wb', ngram_range=(4,6)),
        "unigram_min2_max0.9": TfidfVectorizer(min_df=2, max_df=0.9),
        "uni_bigram_min2_max0.9": TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9),
    })

    print(colorize("\n=== TF-IDF Config Sweep (Corr + Ranking, aligned) ===", fg="cyan", bold=True, enable=USE_COLOR))
    results = []

    for name, vec in configs.items():
        try:
            # ---- (A) Korrelationen wie Script 1 (Similarity vs Noten) ----
            sims = tfidf_diag_similarity(data["reference_answer"].tolist(),
                                         data["provided_answer"].tolist(), vec)
            df_tmp = data.copy()
            df_tmp["similarity"] = sims

            pearson_r, pearson_p, n_global = safe_corr(df_tmp["similarity"], df_tmp["normalized_grade"], method="pearson")
            spear_r, spear_p, _ = safe_corr(df_tmp["similarity"], df_tmp["normalized_grade"], method="spearman")

            per_q_stats = []
            for q, grp in df_tmp.groupby("question"):
                r_p, p_p, n_p = safe_corr(grp["similarity"], grp["normalized_grade"], method="pearson")
                r_s, p_s, n_s = safe_corr(grp["similarity"], grp["normalized_grade"], method="spearman")
                per_q_stats.append({
                    "question": q,
                    "pearson_r": r_p, "pearson_p": p_p, "n_p": n_p,
                    "spearman_r": r_s, "spearman_p": p_s, "n_s": n_s
                })

            pearsons = [d["pearson_r"] for d in per_q_stats if np.isfinite(d["pearson_r"])]
            ns_p = [d["n_p"] for d in per_q_stats if np.isfinite(d["pearson_r"])]

            mean_r_by_q_pearson = float(np.nanmean(pearsons)) if pearsons else np.nan
            fisher_pearson_w, _ = fisher_z_mean(pearsons, ns_p, weighted=True)

            corr_summary = {
                "pearson_r": pearson_r, "pearson_p": pearson_p, "n": n_global,
                "spearman_r": spear_r, "spearman_p": spear_p
            }
            by_q_corr = {
                "mean_pearson_r_simple": mean_r_by_q_pearson,
                "fisher_pearson_weighted": fisher_pearson_w,
            }

            # Save per-question corr table (optional but handy)
            per_q_df = pd.DataFrame(per_q_stats)
            per_q_path = os.path.join(OUTDIR, f"{name}_per_question_correlations.csv")
            per_q_df.to_csv(per_q_path, index=False)

            # ---- (B) Ranking-Block wie Skript 2 ----
            rank_out = eval_with_ranking(data, vec, gt_mode=GT_MODE)

            # Dateien wie in Script 2
            rows_path = os.path.join(OUTDIR, f"{name}_ranking_rows_by_true.csv")
            rank_out["rows"].to_csv(rows_path, index=False)

            metrics_path = os.path.join(OUTDIR, f"{name}_per_question_metrics_rpred.csv")
            rank_out["per_question_metrics"].to_csv(metrics_path, index=False)

            # Ausgabe
            print(format_corr_summary(corr_summary, by_q_corr, rank_out["rank_summary"], name))
            results.append((name, corr_summary, by_q_corr, rank_out))

            # Optional Pretty pro Frage (für diese Konfiguration)
            if PRINT_PRETTY:
                print(colorize(f"\n=== Pretty per-question view — {name} (sorted by TRUE rank) ===", fg="blue", bold=True, enable=USE_COLOR))
                for q, grp in rank_out["rows"].groupby("question", sort=True):
                    print_question_block(grp, q, top_n=TOP_N_PRINT, use_color=USE_COLOR, max_excerpt=90)

                # Globale Zusammenfassung für diese Konfiguration
                mdf = rank_out["per_question_metrics"]
                valid = mdf.dropna(subset=["tau", "rho", "pair_acc"])
                mean_tau = valid["tau"].mean() if not valid.empty else np.nan
                mean_rho = valid["rho"].mean() if not valid.empty else np.nan
                mean_mare = mdf["mare"].mean() if not mdf.empty else np.nan
                mean_pairacc = valid["pair_acc"].mean() if not valid.empty else np.nan
                total_swaps = mdf["swap_lb"].sum() if not mdf.empty else 0

                print(colorize("\n=== Global Summary (vs. rpred) ===", fg="magenta", bold=True, enable=USE_COLOR))
                print(colorize(
                    f"Avg τ={_fmt_cell(mean_tau)}  Avg ρ={_fmt_cell(mean_rho)}  "
                    f"Avg MARE={mean_mare:>5.2f}  Avg PairAcc={_fmt_cell(mean_pairacc)}  "
                    f"Total Swaps↓={int(total_swaps)}",
                    fg="magenta", enable=USE_COLOR
                ))

        except Exception as e:
            print(colorize(f"{name:20s}  ERROR -> {e}", fg="red", bold=True, enable=USE_COLOR))

    # "Beste" Konfigurationen nach Korrelation & Ranking
    if results:
        best_corr = max(results, key=lambda x: (x[1]["pearson_r"], x[2]["fisher_pearson_weighted"]))
        best_rank = max(results, key=lambda x: (x[3]["rank_summary"]["avg_tau"],
                                                x[3]["rank_summary"]["avg_pairacc"],
                                                x[3]["rank_summary"]["avg_topk"]))
        print(colorize("\n--- Best Scores ---", fg="yellow", bold=True, enable=USE_COLOR))
        print(colorize(
            f"Best by Corr: {best_corr[0]} | r={best_corr[1]['pearson_r']:.3f}  "
            f"Fisher-w(byQ)={best_corr[2]['fisher_pearson_weighted']:.3f}",
            fg="yellow", enable=USE_COLOR))
        rs = best_rank[3]["rank_summary"]
        print(colorize(
            f"Best by Ranking: {best_rank[0]} | τ̄={rs['avg_tau']:.3f}  "
            f"PairAcc̄={rs['avg_pairacc']:.3f}  Top{TOPK}̄={rs['avg_topk']:.3f}  Top1̄={rs['top1_rate']:.3f}",
            fg="yellow", enable=USE_COLOR))

if __name__ == "__main__":
    main()
