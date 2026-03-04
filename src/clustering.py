from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLS = [
    "mismatch_rate",
    "avg_quality",
    "avg_height",
    "dominant_size",
    "category_diversity",
    "n_reviews",
]


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggrega il dataset a livello utente per costruire feature stabili per clustering.

    Output colonne (minimo):
    - user_id, n_reviews, mismatch_rate, avg_quality, avg_height, dominant_size,
      dominant_category, category_diversity
    """
    if "user_id" not in df.columns:
        raise ValueError("La colonna 'user_id' non è presente nel dataframe.")

    required = {"mismatch", "quality", "height_cm", "size", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne richieste per build_user_features: {sorted(missing)}")

    user_features = (
        df.groupby("user_id")
        .agg(
            n_reviews=("mismatch", "count"),
            mismatch_rate=("mismatch", "mean"),
            avg_quality=("quality", "mean"),
            avg_height=("height_cm", "mean"),
            dominant_size=("size", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            dominant_category=("category", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            category_diversity=("category", "nunique"),
        )
        .reset_index()
    )
    return user_features


def prepare_clustering_matrix(
    user_cluster: pd.DataFrame,
    feature_cols: Sequence[str] = tuple(DEFAULT_FEATURE_COLS),
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Estrae la matrice di feature per il clustering e applica standardizzazione (z-score).

    Returns:
        cluster_features: dataframe con sole feature (righe = utenti) che verrà clusterizzato
        X_scaled: matrice numpy standardizzata
        scaler: StandardScaler fittato
    """
    missing = [c for c in feature_cols if c not in user_cluster.columns]
    if missing:
        raise ValueError(f"Mancano colonne in user_cluster: {missing}")

    cluster_features = user_cluster[list(feature_cols)].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)
    return cluster_features, X_scaled, scaler


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_range: Iterable[int] = range(2, 10),
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """
    Calcola inertia + metriche interne per un range di k.
    Utile per elbow e sanity check.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _label_personas(profile_df: pd.DataFrame) -> pd.Series:
    """
    Etichette 'persona' semplici basate su mediane di mismatch_rate e avg_quality.
    """
    q_med = profile_df["avg_quality"].median()
    m_med = profile_df["mismatch_rate"].median()

    def label(row: pd.Series) -> str:
        m = row["mismatch_rate"]
        q = row["avg_quality"]
        if m < m_med and q >= q_med:
            return "Fit-driven soddisfatti"
        if m < m_med and q < q_med:
            return "Esigenti ma coerenti"
        if m >= m_med and q >= q_med:
            return "Tolleranti al mismatch"
        return "A rischio mismatch (friction)"

    return profile_df.apply(label, axis=1)


def build_cluster_profile(
    user_cluster: pd.DataFrame,
    clustered_df: pd.DataFrame,
    feature_cols: Sequence[str] = tuple(DEFAULT_FEATURE_COLS),
) -> pd.DataFrame:
    """
    Costruisce un profilo cluster con:
    - users count
    - medie delle feature
    - moda di dominant_size e dominant_category (se presenti)
    - label persona
    """
    # aggregated means
    agg_dict = {c: (c, "mean") for c in feature_cols if c in clustered_df.columns}
    profile = (
        clustered_df.groupby("cluster")
        .agg(users=("cluster", "count"), **agg_dict)
        .round(3)
        .sort_values("mismatch_rate" if "mismatch_rate" in clustered_df.columns else "users")
    )

    # mode helpers (use original user-level columns, then map to cluster)
    tmp = user_cluster.copy()
    tmp = tmp.loc[clustered_df.index].copy()
    tmp["cluster"] = clustered_df["cluster"].values

    if "dominant_size" in tmp.columns:
        mode_size = tmp.groupby("cluster")["dominant_size"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        profile["dominant_size_mode"] = mode_size

    if "dominant_category" in tmp.columns:
        mode_cat = tmp.groupby("cluster")["dominant_category"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        profile["dominant_category_mode"] = mode_cat

    # persona
    if {"mismatch_rate", "avg_quality"}.issubset(profile.columns):
        profile["persona"] = _label_personas(profile)

    return profile


def run_kmeans_user_clustering(
    user_cluster: pd.DataFrame,
    X_scaled: np.ndarray,
    cluster_features: pd.DataFrame,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 10,
    feature_cols: Sequence[str] = tuple(DEFAULT_FEATURE_COLS),
):
    """
    Esegue KMeans su X_scaled e ritorna:
    - clustered_df: feature_df + colonna cluster (stesso index di cluster_features)
    - cluster_profile_labeled: profilo cluster con persona e mode
    - kmeans model
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X_scaled)

    clustered_df = cluster_features.copy()
    clustered_df["cluster"] = labels

    profile = build_cluster_profile(
        user_cluster=user_cluster,
        clustered_df=clustered_df,
        feature_cols=feature_cols,
    )

    return clustered_df, profile, kmeans
