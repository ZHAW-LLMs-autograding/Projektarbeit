import os
import warnings
from itertools import combinations
from textwrap import shorten

import numpy as np
import pandas as pd
from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    rankdata,
    ConstantInputWarning,
)
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Config
# =========================
PATH = "/home/kevin/projektarbeit/trainingdata/asag_ranking/ASAG_2024_with_ranking.csv"
OUTDIR = "asag_outputs"
TOPK = 3  # für Top-k-Accuracy pro Frage
USE_COLOR = True
TOP_N_PRINT = None  # None = zeige ALLE Zeilen pro Frage in der Terminalansicht

# Warnungen unterdrücken, die bei konstanten Eingaben entstehen (z. B. identische Ränge)
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


def tfidf_diag_similarity(ref_texts, stud_texts, vectorizer: TfidfVectorizer):
    texts = ref_texts + stud_texts
    X = vectorizer.fit_transform(texts)
    n = len(ref_texts)
    A = X[:n]
    B = X[n:]
    # L2-normalisierte Cosine-Ähnlichkeit entlang der Diagonalen
    return A.multiply(B).sum(axis=1).A1


def prepare_dataframe(path):
    data = pd.read_csv(path)
    required = [
        "question",
        "provided_answer",
        "reference_answer",
        "grade",
        "data_source",
        "normalized_grade",
        "weight",
        "index",
        "ranking",
    ]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    # Leertexte absichern
    for col in ["reference_answer", "provided_answer", "question"]:
        data[col] = data[col].fillna("")

    # Typen absichern
    data = data[pd.to_numeric(data["normalized_grade"], errors="coerce").notna()].copy()
    data["normalized_grade"] = data["normalized_grade"].astype(float)
    data["ranking"] = pd.to_numeric(data["ranking"], errors="coerce").astype(int)
    return data


def _pairwise_counts(y_score, y_true):
    """Zählt Paar-Übereinstimmung (C), -Fehler (D), -Ties (T), Gesamtpaare (P)."""
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
# Calibration utils
# =========================

def _snap_to_nearest_grade(vals: np.ndarray, unique_grades: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    ug = np.asarray(unique_grades, dtype=float)
    diffs = np.abs(vals[:, None] - ug[None, :])
    idx = np.argmin(diffs, axis=1)
    return ug[idx]


def calibrate_similarity_to_grade(similarity: np.ndarray, grades: np.ndarray, method: str = "isotonic"):
    """Mappt Cosine-Similarity → (kontinuierliche) Grade-Schätzung und snapped auf vorhandene Grade-Stufen.
    method: 'isotonic' (monotone Regression) oder 'centroid' (Nearest-mean pro Grade).
    Returns dict with keys: pred_grade_cont, pred_grade_snap, unique_grades.
    """
    similarity = np.asarray(similarity, dtype=float)
    grades = np.asarray(grades, dtype=float)
    unique_grades = np.sort(np.unique(grades[~np.isnan(grades)]))

    if method == "centroid":
        # Einfach: pro Grade den Median der Similarities und dann nearest-centroid
        df_tmp = pd.DataFrame({"sim": similarity, "grade": grades})
        centroids = df_tmp.groupby("grade")["sim"].median().to_dict()
        # lineare Interpolation über die Grade-Reihenfolge (monoton erzwingen)
        xs = np.array([centroids[g] for g in unique_grades])
        ys = unique_grades.copy()
        # Umkehrfunktion: für gegebenes sim finde nächstliegende Grade-Ähnlichkeit
        pred_grade_snap = _snap_to_nearest_grade(similarity[:, None], ys[None, :])  # placeholder, überschreiben unten
        # Besser: mappe sim → nächstes xs, gib dessen y zurück
        diffs = np.abs(similarity[:, None] - xs[None, :])
        idx = np.argmin(diffs, axis=1)
        pred_grade_cont = ys[idx]
        pred_grade_snap = ys[idx]
        return {"pred_grade_cont": pred_grade_cont, "pred_grade_snap": pred_grade_snap, "unique_grades": unique_grades}

    # Default: Isotonic Regression (monotone, robust gegen Nichtlinearität)
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=float(unique_grades.min()), y_max=float(unique_grades.max()), out_of_bounds="clip")
        pred_cont = iso.fit_transform(similarity, grades)
        pred_snap = _snap_to_nearest_grade(pred_cont, unique_grades)
        return {"pred_grade_cont": pred_cont, "pred_grade_snap": pred_snap, "unique_grades": unique_grades}
    except Exception:
        # Fallback: centroid
        return calibrate_similarity_to_grade(similarity, grades, method="centroid")


# =========================
# Core evaluation
# =========================

def eval_with_ranking(df, vec):
    """Berechnet Similarity, rundet sie abhängig von der Anzahl Noten, kalibriert sie auf Notenskala,
    erzeugt true/pred-Rank pro Frage und gibt sortierte DataFrames zurück plus globale Korrelationen."""
    sims = tfidf_diag_similarity(df["reference_answer"].tolist(), df["provided_answer"].tolist(), vec)
    df = df.copy()
    df["similarity_raw"] = sims

    # Ähnlichkeiten runden, abhängig von Anzahl unterschiedlicher Grades (macht Ranking robuster)
    n_unique_grades = df["normalized_grade"].nunique(dropna=True)
    if n_unique_grades <= 3:
        decimals = 3
    elif n_unique_grades <= 6:
        decimals = 4
    else:
        decimals = 5
    df["similarity"] = df["similarity_raw"].round(decimals)

    # === Neu: Kalibrierung similarity -> Grade (global) ===
    calib = calibrate_similarity_to_grade(df["similarity"].values, df["normalized_grade"].values, method="isotonic")
    df["pred_grade_cont"] = calib["pred_grade_cont"]
    df["pred_grade_snap"] = calib["pred_grade_snap"]
    unique_grades = calib["unique_grades"]

    # Globale Korrelation (informativ)
    try:
        pearson_r, _ = pearsonr(df["similarity"], df["normalized_grade"])
    except Exception:
        pearson_r = np.nan
    try:
        spearman_r, _ = spearmanr(df["similarity"], df["normalized_grade"])
    except Exception:
        spearman_r = np.nan

    rows = []
    for q, grp in df.groupby("question", sort=True):
        g = grp.copy()
        # TRUE RANK direkt aus CSV-Spalte "ranking" (1 = bestes)
        g["true_rank"] = g["ranking"].astype(int)

        # ===== Predicted Rank (calibrated to grades): pred_rank =====
        g = g.sort_values(["pred_grade_snap", "similarity"], ascending=[False, False])
        g["pred_rank"] = rankdata(-g["pred_grade_snap"].values, method="average")
        same_bin = g.groupby("pred_grade_snap")["similarity"].rank(ascending=False, method="average")
        g["pred_rank"] = (g["pred_rank"] * 1e6 + same_bin).rank(method="first").astype(int)

        # ===== Neue Variante: Rounded-Ranking (per Frage) =====
        # Idee: Cos-Sim in so viele Quantil-Buckets schneiden wie es in DIESER Frage verschiedene Noten gibt.
        #      Danach Ranking primär nach Bucket (höher besser), sekundär nach Similarity.
        m_grades = max(1, g["normalized_grade"].nunique(dropna=True))
        try:
            # qcut kann bei vielen Ties weniger als m_grades Buckets erzeugen -> duplicates='drop'
            qlabels = list(range(m_grades))
            qb = pd.qcut(g["similarity"], q=m_grades, labels=qlabels, duplicates='drop')
            # größere Buckets sollen "besser" sein → invertieren
            qb = qb.astype(int)
            g["rounded_bucket"] = qb.max() - qb
        except Exception:
            # Fallback: equal-width bins
            bins = np.linspace(g["similarity"].min(), g["similarity"].max(), num=m_grades+1)
            qb = pd.cut(g["similarity"], bins=bins, labels=False, include_lowest=True)
            qb = qb.fillna(0).astype(int)
            g["rounded_bucket"] = qb.max() - qb

        # Rounded-Rank berechnen
        g = g.sort_values(["rounded_bucket", "similarity"], ascending=[False, False])
        g["rounded_rank"] = rankdata(-g["rounded_bucket"].values, method="average")
        same_rb = g.groupby("rounded_bucket")["similarity"].rank(ascending=False, method="average")
        g["rounded_rank"] = (g["rounded_rank"] * 1e6 + same_rb).rank(method="first").astype(int)

        for _, r in g.iterrows():
            rows.append(
                {
                    "question": q,
                    "reference_excerpt": str(r["reference_answer"])[:120],
                    "student_excerpt": str(r["provided_answer"])[:200],
                    "true_rank": int(r["true_rank"]),
                    "pred_rank": int(r["pred_rank"]),
                    "rounded_rank": int(r["rounded_rank"]),
                    "rounded_bucket": int(r["rounded_bucket"]),
                    "pred_grade": float(r["pred_grade_snap"]),
                    "similarity": float(r["similarity"]),
                    "grade": float(r["normalized_grade"]),
                }
            )

    all_rows = pd.DataFrame(rows)

    # Sortierung: zuerst Frage, dann True-Rang, danach Pred-Rang
    sorted_by_true = all_rows.sort_values(["question", "true_rank", "pred_rank"], ascending=[True, True, True], kind="mergesort")
    sorted_by_pred = all_rows.sort_values(["question", "pred_rank", "true_rank"], ascending=[True, True, True], kind="mergesort")
    info = {
        "n_unique_grades": int(n_unique_grades),
        "similarity_round_decimals": int(decimals),
        "unique_grades": list(map(float, unique_grades)),
    }
    return pearson_r, spearman_r, sorted_by_true, sorted_by_pred, info


def compute_per_question_metrics(all_rows: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    """Ranking-Metriken pro Frage.
    Erwartet Spalten: question, true_rank, pred_rank, similarity, grade.
    Gibt u. a. tau, rho, MARE, PairAcc, Swaps↓ (discordant pairs) und Top-k-Accuracy zurück.
    """
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

        # Rang-Korrelationen (kleinere Ränge = besser). scipy arbeitet direkt mit den Rangwerten.
        try:
            tau, _ = kendalltau(g["pred_rank"], g["true_rank"])  # tau-b, robust gg. Ties
        except Exception:
            tau = np.nan
        try:
            rho, _ = spearmanr(g["pred_rank"], g["true_rank"])  # Spearman ρ
        except Exception:
            rho = np.nan

        mare = float(np.mean(np.abs(g["pred_rank"] - g["true_rank"])))

        # Pairwise counts auf Basis der RANGWERTE (kleiner = besser). Für Vergleich nehmen wir negatives Ranking als "Score" (größer = besser)
        C, D, T, P = _pairwise_counts(-g["pred_rank"].values.astype(float), -g["true_rank"].values.astype(float))
        pair_acc = C / (C + D) if (C + D) > 0 else np.nan
        swap_lb = D  # untere Schranke für benötigte Adjacent-Swaps

        # Top-k Accuracy & Top-1
        k = min(topk, n)
        top_true_ids = set(g.sort_values("true_rank").head(k).index)
        top_pred_ids = set(g.sort_values("pred_rank").head(k).index)
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
                "pairs_total": int(P),
                "swap_lb": int(swap_lb),
                "topk_acc": float(topk_acc) if topk_acc is not None else np.nan,
                "top1_match": int(top1_match),
            }
        )
    return pd.DataFrame(metrics)


# =========================
# Pretty printing per question
# =========================

def print_question_block(df_q_full, q, top_n=TOP_N_PRINT, by="true", use_color=USE_COLOR, max_excerpt=90):
    # Sicherstellen, dass Ränge integer sind
    df_q_full = df_q_full.copy()
    df_q_full["true_rank"] = pd.to_numeric(df_q_full["true_rank"], errors="coerce").astype(int)
    df_q_full["pred_rank"] = pd.to_numeric(df_q_full["pred_rank"], errors="coerce").astype(int)

    # Strikte Sortierung (stabil): zuerst TRUE-Rank, dann Pred, dann Similarity
    # Immer strikt nach TRUE-Rank sortieren (stabil), danach pred_rank, dann similarity
    df_q_disp = df_q_full.sort_values(
        ["true_rank", "pred_rank", "similarity"],
        ascending=[True, True, False],
        kind="mergesort",
    )

    # Alle oder top_n
    if isinstance(top_n, int) and top_n > 0:
        df_q_disp = df_q_disp.head(top_n)

    # Header
    line = "=" * 110
    print(line)
    print(colorize(f"QUESTION: {q}", fg="cyan", bold=True, enable=use_color))

    # Referenzzeile (erste Referenz anzeigen)
    ref = df_q_disp["reference_excerpt"].iloc[0] if not df_q_disp.empty else ""
    print(colorize("Ref:", fg="grey", enable=use_color), shorten(str(ref), width=100, placeholder="…"))

    # Tabelle
    print("-" * 110)
    header = f"{'#':>2}  {'true':>4}  {'pred':>4}  {'Δ':>2}  {'sim':>6}  {'pG':>4}  {'grade':>5}  student_excerpt"
    print(colorize(header, fg="yellow", bold=True, enable=use_color))
    print("-" * 110)

    for i, r in enumerate(df_q_disp.itertuples(index=False), start=1):
        d = int(r.pred_rank) - int(r.true_rank)
        # Farbe je nach Trefferqualität
        if d == 0:
            fg = "green"
        elif abs(d) <= 2:
            fg = "yellow"
        else:
            fg = "red"
        row = (
            f"{i:>2}  "
            f"{int(r.true_rank):>4}  "
            f"{int(r.pred_rank):>4}  "
            f"{d:>+2}  "
            f"{r.similarity:>6.3f}  "
            f"{getattr(r, 'pred_grade', np.nan):>4.2f}  "
            f"{r.grade:>5.2f}  "
            f"{shorten(str(r.student_excerpt), width=max_excerpt, placeholder='…')}"
        )
        print(colorize(row, fg=fg, enable=use_color))
    print(line)

    # ======= Summary für diese Frage (auf FULL berechnet) =======
    # Metriken
    mdf = compute_per_question_metrics(df_q_full, topk=TOPK)
    mr = mdf.iloc[0]

    # Grades in dieser Frage & mittlere Similarity pro Grade
    grades = sorted(df_q_full["grade"].dropna().unique())
    by_grade = df_q_full.groupby("grade")["similarity"].mean().to_dict()

    print("-" * 110)
    stats_line1 = (
        f"Summary:  τ={_fmt_cell(mr.tau)}  ρ={_fmt_cell(mr.rho)}  MARE={mr.mare:>5.2f}  "
        f"PairAcc={_fmt_cell(mr.pair_acc)}  Swaps↓={_fmt_cell(mr.swap_lb, fmt_int=True, width=3)}  "
        f"Top{TOPK}={_fmt_cell(mr.topk_acc, width=5)}  Top1={_fmt_cell(mr.top1_match, fmt_int=True, width=2)}"
    )
    print(colorize(stats_line1, fg="magenta", bold=False, enable=use_color))

    stats_line2 = "Grades: [" + ", ".join(f"{g:.2f}" for g in grades) + "]  Avg sim per grade: "
    pairs = [f"{g:.2f}:{by_grade.get(g, np.nan):.3f}" for g in grades]
    print(colorize(stats_line2 + "; ".join(pairs), fg="grey", enable=use_color))

    print(line)
    print(colorize(f"QUESTION: {q}", fg="cyan", bold=True, enable=use_color))

    # Referenzzeile (erste Referenz anzeigen)
    ref = df_q_disp["reference_excerpt"].iloc[0] if not df_q_disp.empty else ""
    print(colorize("Ref:", fg="grey", enable=use_color), shorten(str(ref), width=100, placeholder="…"))

    # Tabelle
    print("-" * 110)
    header = f"{'#':>2}  {'true':>4}  {'pred':>4}  {'Δ':>2}  {'sim':>6}  {'grade':>5}  student_excerpt"
    print(colorize(header, fg="yellow", bold=True, enable=use_color))
    print("-" * 110)

    for i, r in enumerate(df_q_disp.itertuples(index=False), start=1):
        d = int(r.pred_rank) - int(r.true_rank)
        # Farbe je nach Trefferqualität
        if d == 0:
            fg = "green"
        elif abs(d) <= 2:
            fg = "yellow"
        else:
            fg = "red"
        row = (
            f"{i:>2}  "
            f"{int(r.true_rank):>4}  "
            f"{int(r.pred_rank):>4}  "
            f"{d:>+2}  "
            f"{r.similarity:>6.3f}  "
            f"{r.grade:>5.2f}  "
            f"{shorten(str(r.student_excerpt), width=max_excerpt, placeholder='…')}"
        )
        print(colorize(row, fg=fg, enable=use_color))

    # ======= Summary für diese Frage (auf FULL berechnet) =======
    # Metriken
    mdf = compute_per_question_metrics(df_q_full, topk=TOPK)
    mr = mdf.iloc[0]

    # Grades in dieser Frage & mittlere Similarity pro Grade
    grades = sorted(df_q_full["grade"].dropna().unique())
    by_grade = df_q_full.groupby("grade")["similarity"].mean().to_dict()

    print("-" * 110)
    stats_line1 = (
        f"Summary:  τ={_fmt_cell(mr.tau)}  ρ={_fmt_cell(mr.rho)}  MARE={mr.mare:>5.2f}  "
        f"PairAcc={_fmt_cell(mr.pair_acc)}  Swaps↓={_fmt_cell(mr.swap_lb, fmt_int=True, width=3)}  "
        f"Top{TOPK}={_fmt_cell(mr.topk_acc, width=5)}  Top1={_fmt_cell(mr.top1_match, fmt_int=True, width=2)}"
    )
    print(colorize(stats_line1, fg="magenta", bold=False, enable=use_color))

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

    # Vectorizer (kannst du anpassen)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    pearson_r, spearman_r, df_true, df_pred, info = eval_with_ranking(data, vec)

    print(f"Global Pearson r = {pearson_r:.3f}" if not pd.isna(pearson_r) else "Global Pearson r = NaN")
    print(f"Global Spearman ρ = {spearman_r:.3f}" if not pd.isna(spearman_r) else "Global Spearman ρ = NaN")
    print(f"Similarity rounding: {info['similarity_round_decimals']} decimals  |  Unique grades: {info['n_unique_grades']}")

    # Speichern der Ranking-Tabellen
    out_true = os.path.join(OUTDIR, "ranking_sanity_all_by_true.csv")
    out_pred = os.path.join(OUTDIR, "ranking_sanity_all_by_pred.csv")
    df_true.to_csv(out_true, index=False)
    df_pred.to_csv(out_pred, index=False)

    # ===== Metriken pro Frage (gesamt) =====
    metrics_df = compute_per_question_metrics(df_true, topk=TOPK)
    out_metrics = os.path.join(OUTDIR, "per_question_metrics.csv")
    metrics_df.to_csv(out_metrics, index=False)

    # Hübsche per-Question Metrik-Tabelle
    print("\n=== Per-Question Ranking Metrics ===")
    hdr = f"{'Question':30s}  {'n':>3}  {'tau':>6}  {'rho':>6}  {'MARE':>6}  {'PairAcc':>7}  {'Swaps↓':>7}  {'Top3':>5}  {'Top1':>4}"
    print(colorize(hdr, fg="yellow", bold=True))
    for r in metrics_df.itertuples(index=False):
        line = (
            f"{str(r.question)[:30]:30s}  "
            f"{_fmt_cell(r.n, fmt_int=True, width=3)}  "
            f"{_fmt_cell(r.tau)}  "
            f"{_fmt_cell(r.rho)}  "
            f"{r.mare:>6.2f}" if not pd.isna(r.mare) else f"{'-':>6}"
        )
        line += (
            f"  {_fmt_cell(r.pair_acc)}  "
            f"{_fmt_cell(r.swap_lb, fmt_int=True, width=7)}  "
            f"{_fmt_cell(r.topk_acc, width=5)}  "
            f"{_fmt_cell(r.top1_match, fmt_int=True, width=4)}"
        )
        fg = "green" if (not pd.isna(r.tau) and r.tau >= 0.7) else ("yellow" if (not pd.isna(r.tau) and r.tau >= 0.4) else "red")
        print(colorize(line, fg=fg))

    # Terminal-Layout: pro Frage kompakte Ansicht (TRUE-Rank)
    print("\n=== Pretty per-question view (sorted by TRUE rank) ===")
    for q, grp in df_true.groupby("question", sort=True):
        print_question_block(grp, q, top_n=TOP_N_PRINT, by="true", use_color=USE_COLOR, max_excerpt=90)

    # Alternate view (PRED)
    print("\n=== Alternate view (sorted by PRED rank) ===")
    for q, grp in df_pred.groupby("question", sort=True):
        print_question_block(grp, q, top_n=TOP_N_PRINT, by="pred", use_color=USE_COLOR, max_excerpt=90)

    # ===== Globale Zusammenfassung über alle Fragen =====
    valid = metrics_df.dropna(subset=["tau", "rho", "pair_acc"])  # nur valide
    mean_tau = valid["tau"].mean() if not valid.empty else np.nan
    mean_rho = valid["rho"].mean() if not valid.empty else np.nan
    mean_mare = metrics_df["mare"].mean()
    mean_pairacc = valid["pair_acc"].mean() if not valid.empty else np.nan
    total_swaps = metrics_df["swap_lb"].sum()

    print("\n=== Global Summary ===")
    print(
        colorize(
            f"Avg τ={_fmt_cell(mean_tau)}  Avg ρ={_fmt_cell(mean_rho)}  Avg MARE={mean_mare:>5.2f}  "
            f"Avg PairAcc={_fmt_cell(mean_pairacc)}  Total Swaps↓={int(total_swaps)}  "
            f"Unique grades={info['n_unique_grades']}  Sim rounded to {info['similarity_round_decimals']} decimals",
            fg="magenta",
            bold=True,
            enable=USE_COLOR,
        )
    )


if __name__ == "__main__":
    main()
