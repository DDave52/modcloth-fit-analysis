from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def plot_cluster_radar(
    clustered_df: pd.DataFrame,
    feature_cols: Sequence[str],
    title: str = "Cluster profiles (normalized)",
) -> "px.Figure":
    """
    Radar chart normalizzato 0–1 per confrontare feature su scale diverse.
    """
    prof = clustered_df.groupby("cluster")[list(feature_cols)].mean().reset_index()
    prof_norm = prof.copy()
    for c in feature_cols:
        mn, mx = prof_norm[c].min(), prof_norm[c].max()
        prof_norm[c] = 0.0 if mx == mn else (prof_norm[c] - mn) / (mx - mn)

    prof_long = prof_norm.melt(id_vars="cluster", var_name="feature", value_name="value")

    fig = px.line_polar(
        prof_long,
        r="value",
        theta="feature",
        color="cluster",
        line_close=True,
        markers=True,
        title=title,
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend_title_text="Cluster",
    )
    return fig


def plot_cluster_parallel(
    profile_df: pd.DataFrame,
    feature_cols: Sequence[str],
    title: str = "Cluster comparison — Parallel Coordinates",
) -> "px.Figure":
    """
    Parallel coordinates sui profili cluster (di solito medie).
    """
    # profile_df spesso ha cluster come index -> riportiamo a colonna
    df = profile_df.reset_index().rename(columns={"index": "cluster"})
    if "cluster" not in df.columns:
        df.insert(0, "cluster", df.index)

    fig = px.parallel_coordinates(
        df,
        dimensions=list(feature_cols),
        color="cluster",
        title=title,
    )
    return fig


def plot_cluster_pca(
    X_scaled: np.ndarray,
    clustered_df: pd.DataFrame,
    feature_cols: Sequence[str],
    title_prefix: str = "PCA scatter",
    random_state: int = 42,
) -> "px.Figure":
    """
    PCA 2D projection dello spazio feature scalato.
    Nota: PCA è una proiezione, non una “prova” della bontà dei cluster.
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = clustered_df.copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]

    hover_cols = [c for c in feature_cols if c in pca_df.columns]
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_data=hover_cols,
        title=f"{title_prefix} — explained variance: {pca.explained_variance_ratio_.sum():.2%}",
    )
    return fig


# -----------------------------
# EDA utilities (bar/box)
# -----------------------------

def plot_bar_pct(
    df: pd.DataFrame,
    *,
    x: str,
    y: Union[str, Sequence[str]],
    n_col: str,
    title: str,
    orientation: str = "v",
    pad_ratio: float = 0.12,   # quanto “spazio extra” sopra (o a destra) per le etichette
    min_pad: float = 0.02,     # padding minimo assoluto (in unità 0-1)
) -> "px.Figure":
    """
    Barplot (Plotly) con etichetta del TOTALE 'xx.x% (n=...)' non tagliata.

    - Se y è una lista (es. ["large","small"]) fa stacked bars e stampa l'etichetta sul totale.
    - y deve essere proporzione (0-1).
    - n_col è il volume (conteggio righe).
    """
    plot_df = df.copy()
    y_cols = [y] if isinstance(y, str) else list(y)

    # Totale per ogni x (somma delle componenti se stacked)
    plot_df["_total"] = plot_df[y_cols].sum(axis=1)

    # Label totale
    plot_df["_total_label"] = (
        (plot_df["_total"] * 100).round(1).astype(str)
        + "% (n=" + plot_df[n_col].astype(int).astype(str) + ")"
    )

    # Bar (stack se più colonne)
    fig = px.bar(
        plot_df,
        x=x if orientation == "v" else "_total",
        y=y_cols if orientation == "v" else x,
        orientation=orientation,
        title=title,
        barmode="stack" if len(y_cols) > 1 else "relative",
    )

    # Testo del totale sopra la barra (o a destra se orizzontale)
    if orientation == "v":
        fig.add_trace(
            go.Scatter(
                x=plot_df[x],
                y=plot_df["_total"],
                mode="text",
                text=plot_df["_total_label"],
                textposition="top center",
                showlegend=False,
            )
        )
        fig.update_yaxes(tickformat=".0%", automargin=True)

        ymax = float(np.nanmax(plot_df["_total"])) if len(plot_df) else 1.0
        pad = max(min_pad, ymax * pad_ratio)
        fig.update_yaxes(range=[0, min(1.0, ymax + pad)])

        fig.update_layout(margin=dict(t=90, b=60, l=60, r=40))

    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df["_total"],
                y=plot_df[x],
                mode="text",
                text=plot_df["_total_label"],
                textposition="middle right",
                showlegend=False,
            )
        )
        fig.update_xaxes(tickformat=".0%", automargin=True)

        xmax = float(np.nanmax(plot_df["_total"])) if len(plot_df) else 1.0
        pad = max(min_pad, xmax * pad_ratio)
        fig.update_xaxes(range=[0, min(1.0, xmax + pad)])

        fig.update_layout(margin=dict(t=90, b=60, l=80, r=80))

    return fig


def plot_box_simple(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    points: bool = False,
) -> "px.Figure":
    """Boxplot Plotly semplice (per report/notebook)."""
    return px.box(df, x=x, y=y, points="all" if points else False, title=title)
