# plot_style.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd


def _get_times_new_roman() -> Optional[fm.FontProperties]:
    """Versuche, Times New Roman ueber bekannte Pfade / Systemfonts zu laden."""
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    ]

    for c in candidates:
        if Path(c).exists():
            return fm.FontProperties(fname=c)

    # Fallback: alle Systemfonts scannen
    for path in fm.findSystemFonts():
        lower = path.lower()
        if "times_new_roman" in lower or "times.ttf" in lower:
            return fm.FontProperties(fname=path)

    return None


# Wird beim Import einmal gesucht
TNR_PROP = _get_times_new_roman()


def apply_default_style() -> None:
    """Globaler Style fuer alle Plots ohne globale Font-Family zu setzen."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Keine font.family in rcParams -> keine findfont-Warnings mehr
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "0.3",
            "axes.linewidth": 0.8,
            "figure.autolayout": True,
        }
    )


def apply_axes_font(ax: plt.Axes, title: Optional[str] = None, pad: int = 10) -> None:
    """Setzt Titel, Achsenlabels und Ticklabels auf Times New Roman (falls verfuegbar)."""
    if TNR_PROP is None:
        if title is not None:
            ax.set_title(title, pad=pad)
        return

    # Titel
    if title is not None:
        ax.set_title(title, pad=pad, fontproperties=TNR_PROP)

    # Achsenlabels
    ax.xaxis.label.set_fontproperties(TNR_PROP)
    ax.yaxis.label.set_fontproperties(TNR_PROP)

    # Ticklabels
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(TNR_PROP)


def apply_legend_font(legend: Optional[plt.Legend]) -> None:
    """Setzt die Legendenschrift auf Times New Roman (falls verfuegbar)."""
    if legend is None or TNR_PROP is None:
        return

    for text in legend.get_texts():
        text.set_fontproperties(TNR_PROP)

    if legend.get_title() is not None:
        legend.get_title().set_fontproperties(TNR_PROP)


def plot_metrics_table(
    df: pd.DataFrame,
    title: str = "Model metrics",
    value_format: str = ".3f",
):
    apply_default_style()

    n_rows, n_cols = df.shape
    figsize = (2 + 1.5 * n_cols, 1 + 0.4 * (n_rows + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    cell_text = []
    for _, row in df.iterrows():
        formatted = [
            "-" if pd.isna(v) else format(v, value_format) if isinstance(v, (int, float)) else str(v)
            for v in row
        ]
        cell_text.append(formatted)

    table = ax.table(
        cellText=cell_text,
        rowLabels=df.index.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.3)

    # Times New Roman explizit fuer jede Zelle + Titel setzen
    if TNR_PROP is not None:
        for cell in table.get_celld().values():
            cell.set_text_props(fontproperties=TNR_PROP)
        ax.set_title(title, pad=10, fontproperties=TNR_PROP)
    else:
        ax.set_title(title, pad=10)

    return fig, ax
