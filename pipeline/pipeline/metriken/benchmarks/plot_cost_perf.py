#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = PLOTS_DIR / "kosten_leistungs_analyse_inferenz_1minus_wrmse.png"


# ---------------------------------------------------------------------
# HARTE WERTE AUS DER TABELLE
# (wRMSE, clamped falls besser; geplottet wird 1 − wRMSE)
# ---------------------------------------------------------------------

MODELS: List[Tuple[str, float, float]] = [
    ("BERT finetuned",                     6.6,   0.262),
    ("SentenceTransformer BERT",          12.0,   0.355),
    ("LLaMA 3.1 8B base",                150.0,   0.654),
    ("LLaMA 3.1 8B instruct",            155.0,   0.374),
    ("LLaMA 3.1 8B base LoRA finetuned",  160.0,   0.375),
    ("LLaMA 3.1 8B instruct LoRA finetuned", 165.0, 0.346),
]

VEKTORMODELL_NAME = "Vektormodell (char wb 3–5 min2)"
VEKTORMODELL_WRMSE = 0.464

GPT4O_NAME = "GPT-4o"
GPT4O_WRMSE = 0.287
GPT4O_INF_MS: Optional[float] = None  # bleibt Referenzlinie


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------

def main() -> None:
    fig = plt.figure(figsize=(9.2, 6))
    ax = plt.gca()

    xs: List[float] = []
    ys: List[float] = []

    # Punkte plotten und Farben merken
    eintraege = []  # (name, x, y, farbe)
    for name, x_ms, wrmse in MODELS:
        score = 1.0 - wrmse
        sc = ax.scatter(x_ms, score, marker="x", zorder=3)
        farbe = sc.get_facecolor()[0] if len(sc.get_facecolor()) else sc.get_edgecolor()[0]
        eintraege.append((name, x_ms, score, farbe))
        xs.append(x_ms)
        ys.append(score)

    # -------------------------------------------------------------
    # Referenzlinie: Vektormodell
    # -------------------------------------------------------------
    vektor_score = 1.0 - VEKTORMODELL_WRMSE
    ax.axhline(vektor_score, linestyle="dashed", linewidth=1.5)
    ax.text(
        min(xs),
        vektor_score,
        f"{VEKTORMODELL_NAME} ({vektor_score:.3f})",
        fontsize=9,
        va="top",
        ha="left",
    )

    # -------------------------------------------------------------
    # Referenzlinie: GPT-4o
    # -------------------------------------------------------------
    gpt_score = 1.0 - GPT4O_WRMSE
    ax.axhline(gpt_score, linestyle="dotted", linewidth=1.5)
    ax.text(
        min(xs),
        gpt_score,
        f"{GPT4O_NAME} ({gpt_score:.3f})",
        fontsize=9,
        va="bottom",
        ha="left",
    )

    # -------------------------------------------------------------
    # Seiten-Beschriftung (Legenden-Spalte)
    # -------------------------------------------------------------
    eintraege_sortiert = sorted(eintraege, key=lambda e: e[2], reverse=True)

    x_seite = 1.02
    y_start = 0.90
    y_abstand = 0.07

    ax.text(
        x_seite,
        0.97,
        "Modelle",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    for i, (name, _, _, farbe) in enumerate(eintraege_sortiert):
        ax.text(
            x_seite,
            y_start - i * y_abstand,
            name,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            color=farbe,
        )

    # -------------------------------------------------------------
    # Achsen & Layout
    # -------------------------------------------------------------
    ax.set_xlabel("Inferenzzeit pro Antwort (ms)")
    ax.set_ylabel("1 − wRMSE (hoeher = besser)")
    ax.set_title("Kosten-Leistungs-Analyse: Inferenz vs. (1 − wRMSE)")

    # >>> NEU: Y-Intervall 0.3 – 0.8
    ax.set_ylim(0.3, 0.8)

    ax.set_xlim(min(xs) - 5, max(xs) + 10)
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Platz rechts fuer Seiten-Labels
    plt.subplots_adjust(right=0.72)

    plt.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    print(f"Plot gespeichert unter: {OUT_PNG}")


if __name__ == "__main__":
    main()
