from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CATEGORY_MAP = {
    "absurd": "Absurde Antwort",
    "correct_clean": "Korrekt (klar formuliert)",
    "correct_awkward": "Korrekt (umständlich formuliert)",
    "formal_good_but_off_topic": "Formal korrekt, aber am Thema vorbei",
    "keywords": "Nur Stichwörter",
    "partial": "Teilweise korrekt",
    "synonym_correct": "Korrekt (Synonyme)",
    "wrong": "Falsch"
}


# ----------------------------------------------------------
# FIXED ABSOLUTE PATHS
# ----------------------------------------------------------

BASE = Path("/home/sercan/asag_pipeline/pipeline/utils/BERT-Leakage-Test")

RESULTS_KNOWN = BASE / "results.json"
RESULTS_UNSEEN = BASE / "results_unseen.json"
OUTDIR = BASE / "plots"
OUTDIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

def load_results(path, label):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["dataset"] = label
    df["category"] = df["id"].apply(lambda x: "_".join(x.split("_")[1:]))
    df["question_group"] = df["id"].apply(lambda x: x.split("_")[0])
    return df

df["dataset"] = df["dataset"].replace({
    "known": "Bekannte Fragen",
    "unseen": "Ungesehene Fragen"
})


df["category_de"] = df["category"].map(CATEGORY_MAP)

df_known = load_results(RESULTS_KNOWN, "known")
df_unseen = load_results(RESULTS_UNSEEN, "unseen")
df = pd.concat([df_known, df_unseen], ignore_index=True)



print("[INFO] Loaded", len(df), "rows")

# ----------------------------------------------------------
# PLOT 1 — Boxplot
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="category", y="prediction", hue="dataset")
plt.title("Prediction Distribution per Category (Known vs Unseen)")
plt.tight_layout()
plt.savefig(OUTDIR / "plot_category_boxplot.png")
plt.close()

# ----------------------------------------------------------
# PLOT 2 — Stripplot
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.stripplot(data=df, x="question_group", y="prediction", hue="dataset", jitter=True)
plt.title("Predictions per Question Group (Known vs Unseen)")
plt.tight_layout()
plt.savefig(OUTDIR / "plot_question_vs_score.png")
plt.close()

# ----------------------------------------------------------
# PLOT 3 — Distribution
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.histplot(df_known["prediction"], kde=True, label="Known", color="blue")
sns.histplot(df_unseen["prediction"], kde=True, label="Unseen", color="red", alpha=0.6)
plt.title("Distribution of Predictions: Known vs Unseen")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "plot_known_vs_unseen_distributions.png")
plt.close()

# ----------------------------------------------------------
# PLOT 4 — Scatter
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(df_known)), y=df_known["prediction"], label="Known")
sns.scatterplot(x=range(len(df_unseen)), y=df_unseen["prediction"], label="Unseen")
plt.title("Scatterplot Predictions (Known vs Unseen)")
plt.tight_layout()
plt.savefig(OUTDIR / "plot_scatter_known_unseen.png")
plt.close()

# ----------------------------------------------------------
# PLOT 5 — Heatmap
# ----------------------------------------------------------

pivot = df.pivot_table(index="dataset", columns="category", values="prediction", aggfunc="mean")

pivot = df.pivot_table(
    index="dataset",
    columns="category_de",
    values="prediction",
    aggfunc="mean"
)


plt.figure(figsize=(10, 5))
sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0)
plt.title("Mittlere BERT-Modellvorhersage pro Antwortkategorie")
plt.tight_layout()
plt.savefig(OUTDIR / "plot_heatmap_averages.png")
plt.close()

print("[OK] All plots saved to:", OUTDIR)
