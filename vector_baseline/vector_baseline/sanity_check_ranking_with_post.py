import os
import warnings
from textwrap import shorten

import numpy as np
import pandas as pd
from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    ConstantInputWarning,
)
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Config
# =========================
PATH = "home/kevin/projektarbeit/pipeline/trainingsdata/ASAG2024/student_ranking/rank_test.parquet"
OUTDIR = "asag_outputs"
TOPK = 3
USE_COLOR = True
TOP_N_PRINT = None  # None = alle Zeilen je Frage anzeigen

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# =========================
# Helpers
# =========================

def colorize(text, fg=None, bold=False, enable=True):
    if not enable or fg is None:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36, "grey": 90}
    parts = []
    if bold:
        parts.append("1")
    parts.append(str(codes.get(fg, 37)))
    return f"\x1b[{';'.join(parts)}m{text}\x1b[0m"

#calculates the cosin similarity between the ref answer and the student answer
def tfidf_diag_similarity(ref_texts, stud_texts, vectorizer: TfidfVectorizer):
    texts = ref_texts + stud_texts
    X = vectorizer.fit_transform(texts)
    n = len(ref_texts)
    A = X[:n]
    B = X[n:]
    # Cosine-Sim entlang der Diagonalen (L2-normalisiert) ??? muss das so
    return A.multiply(B).sum(axis=1).A1

# check if data is prepared like it needs to be
def prepare_dataframe(path):
    data = pd.read_csv(path)
    required = [
        "question","provided_answer","reference_answer","grade",
        "data_source","normalized_grade","weight","index","ranking",
    ]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    for col in ["reference_answer", "provided_answer", "question"]:
        data[col] = data[col].fillna("")

    # take the normalized grade and check it its not NaN
    data = data[pd.to_numeric(data["normalized_grade"], errors="coerce").notna()].copy()
    #
    data["normalized_grade"] = data["normalized_grade"].astype(float)
    #
    data["ranking"] = pd.to_numeric(data["ranking"], errors="coerce").astype(int)
    return data

# ???
def _pairwise_counts(y_score, y_true):
    idx = np.arange(len(y_true))
    C = D = T = 0
    for i in idx:
        vi = y_score[i]
        wi = y_true[i]
        for j in range(i + 1, len(y_true)):
            ds = np.sign(vi - y_score[j])
            dt = np.sign(wi - y_true[j])
            if dt == 0 or ds == 0:
                T += 1
            elif ds == dt:
                C += 1
            else:
                D += 1
    P = (len(y_true) * (len(y_true) - 1)) // 2
    return C, D, T, P

#cell formatting????
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
# Core evaluation
# =========================

def eval_with_ranking(df, vec):
    """
    - Berechnet Similarity
    - Erzeugt pro Frage 'rsim' durch Snapping der Similarities auf grade-basierte Zentroiden
    - Baut 'rpred' (rounded rank) nur aus 'rsim' (gleiche rsim -> gleicher Rang)
    - Baut auch 'pred_rank' (nur informativ: sortiert nach roher Similarity)
    - Gibt sortierte DataFrames (by true) + globale Korrelationen zurück
    """
    sims = tfidf_diag_similarity(df["reference_answer"].tolist(), df["provided_answer"].tolist(), vec)
    base = df.copy()
    base["similarity"] = sims

    # globale Korrelation rein informativ
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
        g = grp.copy()

        # True Rank aus CSV (1 = best)
        g["true_rank"] = g["ranking"].astype(int)

        # ---------- Schwellen/Anker über Grade-Zentroiden (pro Frage) ----------
        # Für jede vorhandene Note: Median der Similarities als Anker
        centroids = g.groupby("normalized_grade")["similarity"].median().sort_values(ascending=False)
        # Falls nur 1 Gradestufe, rsim = dieser Zentroid
        if len(centroids) == 1:
            rsim = np.full(len(g), float(centroids.iloc[0]), dtype=float)
        else:
            # Snap jede Similarity auf den nächsten Zentroid (L2)
            cvals = centroids.values.astype(float)
            simv = g["similarity"].values.astype(float)
            # Distanzmatrix (N x K)
            dists = np.abs(simv[:, None] - cvals[None, :])
            nearest_idx = np.argmin(dists, axis=1)
            rsim = cvals[nearest_idx]

        g["rsim"] = rsim

        # ---------- rpred (Rounded Rank) NUR aus rsim ----------
        # gleiche rsim => gleicher Rang (dichte Ränge, absteigend)
        unique_rsim = np.array(sorted(np.unique(rsim), reverse=True))
        rsim_to_rank = {v: i + 1 for i, v in enumerate(unique_rsim)}
        g["rpred"] = [rsim_to_rank[v] for v in rsim]

        # ---------- pred_rank (nur Info): rohes Similarity-Ranking ----------
        # (gleiche sim teilen sich denselben Platz -> "dense" über unique Werte)
        unique_sim = np.array(sorted(np.unique(g["similarity"].values), reverse=True))
        sim_to_rank = {v: i + 1 for i, v in enumerate(unique_sim)}
        g["pred_rank"] = [sim_to_rank[v] for v in g["similarity"].values]

        # Sammle Ausgabezeilen
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

    all_rows = pd.DataFrame(rows)

    # Sortierung für alle Ausgaben: question → true_rank → rpred → similarity (stabil)
    sorted_by_true = all_rows.sort_values(
        ["question", "true_rank", "rpred", "similarity"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )

    # Eine alternative Sicht (falls du willst): nach rpred
    sorted_by_pred = all_rows.sort_values(
        ["question", "rpred", "true_rank", "similarity"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )

    info = {
        "n_unique_grades": int(df["normalized_grade"].nunique(dropna=True)),
    }
    return pearson_r, spearman_r, sorted_by_true, sorted_by_pred, info

# =========================
# Metrics (gegen rpred)
# =========================

def compute_per_question_metrics(all_rows: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    metrics = []
    for q, g in all_rows.groupby("question", sort=True):
        g = g.copy()
        n = len(g)
        if n < 2:
            metrics.append(
                {   
                    # The question identifier used to group all student answers for one task
                    "question": q,
                    # Number of student answers available for this question (sample size)
                    "n": int(n),
                    # Kendall’s tau: rank correlation based on concordant vs. discordant answer pairs (−1 to +1)
                    "tau": np.nan,
                    # Spearman’s rho: correlation between predicted and true ranks (−1 to +1)
                    "rho": np.nan,
                    # Mean Absolute Rank Error: average absolute difference between predicted and true ranks (lower is better)
                    "mare": np.nan,
                    # Pairwise Accuracy: proportion of answer pairs that are correctly ordered by the model (0 to 1)
                    "pair_acc": np.nan,
                    # Number of discordant pairs (answers whose relative order is incorrect)
                    "discordant": 0,
                    # Number of tied pairs (answers with identical ranks in either prediction or ground truth)
                    "ties": 0,
                    # Total number of possible answer pairs: n*(n−1)/2
                    "pairs_total": 0,
                    # Lower bound on the number of swaps needed to correct the predicted ranking (equals discordant pairs)
                    "swap_lb": 0,
                    # Top-k Accuracy: fraction of overlap between top-k predicted and true answers (e.g., k=3)
                    "topk_acc": np.nan,
                    # Top-1 Match: 1 if the best predicted answer matches the true best answer, else 0
                    "top1_match": np.nan,
                }

            )
            continue

        # Korrelationen der Ränge (rounded)
        try:
            tau, _ = kendalltau(g["rpred"], g["true_rank"])
        except Exception:
            tau = np.nan
        try:
            rho, _ = spearmanr(g["rpred"], g["true_rank"])
        except Exception:
            rho = np.nan

        mare = float(np.mean(np.abs(g["rpred"] - g["true_rank"])))

        # Pairwise auf Basis von rpred <-> true
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

        # Top-k Accuracy & Top-1 (gegen rpred)
        k = min(topk, n)
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
                "pairs_total": int((n * (n - 1)) // 2),
                "swap_lb": int(swap_lb),
                "topk_acc": float(topk_acc) if topk_acc is not None else np.nan,
                "top1_match": int(top1_match),
            }
        )
    return pd.DataFrame(metrics)

# =========================
# Pretty printing per question
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

# =========================
# Main
# =========================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = prepare_dataframe(PATH)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    pearson_r, spearman_r, df_true, df_pred, info = eval_with_ranking(data, vec)

    print(f"Global Pearson r = {pearson_r:.3f}" if not pd.isna(pearson_r) else "Global Pearson r = NaN")
    print(f"Global Spearman ρ = {spearman_r:.3f}" if not pd.isna(spearman_r) else "Global Spearman ρ = NaN")
    print(f"Unique grades (global): {info['n_unique_grades']}")

    # Speichern
    out_true = os.path.join(OUTDIR, "ranking_sanity_all_by_true.csv")
    out_pred = os.path.join(OUTDIR, "ranking_sanity_all_by_rpred.csv")
    df_true.to_csv(out_true, index=False)
    df_pred.to_csv(out_pred, index=False)

    # Per-Question Metriken (gegen rpred)
    metrics_df = compute_per_question_metrics(df_true, topk=TOPK)
    out_metrics = os.path.join(OUTDIR, "per_question_metrics_rpred.csv")
    metrics_df.to_csv(out_metrics, index=False)

    # Übersichtstabelle
    print("\n=== Per-Question Ranking Metrics (vs. rpred) ===")
    hdr = f"{'Question':30s}  {'n':>3}  {'tau':>6}  {'rho':>6}  {'MARE':>6}  {'PairAcc':>7}  {'Swaps↓':>7}  {'Top3':>5}  {'Top1':>4}"
    print(colorize(hdr, fg="yellow", bold=True))
    for r in metrics_df.itertuples(index=False):
        line = (
            f"{str(r.question)[:30]:30s}  "
            f"{_fmt_cell(r.n, fmt_int=True, width=3)}  "
            f"{_fmt_cell(r.tau)}  "
            f"{_fmt_cell(r.rho)}  "
            f"{(r.mare if not pd.isna(r.mare) else np.nan):>6.2f}"
        )
        line += (
            f"  {_fmt_cell(r.pair_acc)}  "
            f"{_fmt_cell(r.swap_lb, fmt_int=True, width=7)}  "
            f"{_fmt_cell(r.topk_acc, width=5)}  "
            f"{_fmt_cell(r.top1_match, fmt_int=True, width=4)}"
        )
        fg = "green" if (not pd.isna(r.tau) and r.tau >= 0.7) else ("yellow" if (not pd.isna(r.tau) and r.tau >= 0.4) else "red")
        print(colorize(line, fg=fg))

    # Pretty view: nach TRUE sortiert
    print("\n=== Pretty per-question view (sorted by TRUE rank) ===")
    for q, grp in df_true.groupby("question", sort=True):
        print_question_block(grp, q, top_n=TOP_N_PRINT, use_color=USE_COLOR, max_excerpt=90)

    # Globale Zusammenfassung
    valid = metrics_df.dropna(subset=["tau", "rho", "pair_acc"])
    mean_tau = valid["tau"].mean() if not valid.empty else np.nan
    mean_rho = valid["rho"].mean() if not valid.empty else np.nan
    mean_mare = metrics_df["mare"].mean()
    mean_pairacc = valid["pair_acc"].mean() if not valid.empty else np.nan
    total_swaps = metrics_df["swap_lb"].sum()

    print("\n=== Global Summary (vs. rpred) ===")
    print(
        colorize(
            f"Avg τ={_fmt_cell(mean_tau)}  Avg ρ={_fmt_cell(mean_rho)}  Avg MARE={mean_mare:>5.2f}  "
            f"Avg PairAcc={_fmt_cell(mean_pairacc)}  Total Swaps↓={int(total_swaps)}",
            fg="magenta",
            bold=True,
            enable=USE_COLOR,
        )
    )

if __name__ == "__main__":
    main()
