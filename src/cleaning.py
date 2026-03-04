"""Utility di pulizia per il progetto Start2Impact (ModCloth).

Funzioni principali:
- parse_height_to_cm: converte stringhe tipo '5ft 6in' in cm.
- clean_dataset: crea feature derivate e filtra outlier/rarità per analisi più stabili.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd


_HEIGHT_PATTERN = re.compile(r"(\d+)ft\s*(\d+)in")


def parse_height_to_cm(height_str: Optional[str]) -> float:
    """Converte '5ft 6in' -> cm. Ritorna np.nan se mancante/non parsabile."""
    if height_str is None or pd.isna(height_str):
        return np.nan

    match = _HEIGHT_PATTERN.search(str(height_str))
    if not match:
        return np.nan

    feet = int(match.group(1))
    inches = int(match.group(2))
    total_inches = feet * 12 + inches
    return round(total_inches * 2.54, 1)


def add_mismatch_flag(df: pd.DataFrame, *, fit_col: str = "fit") -> pd.DataFrame:
    """Aggiunge `mismatch` = 1 se fit != 'fit' (cioè small/large), altrimenti 0."""
    out = df.copy()
    if fit_col not in out.columns:
        raise KeyError(f"Colonna '{fit_col}' non trovata nel DataFrame.")
    out["mismatch"] = (out[fit_col] != "fit").astype(int)
    return out


def filter_valid_sizes(df: pd.DataFrame, *, size_col: str = "size", min_count: int = 50) -> pd.DataFrame:
    """Mantiene solo le taglie con almeno `min_count` osservazioni (stabilità statistica)."""
    out = df.copy()
    if size_col not in out.columns:
        raise KeyError(f"Colonna '{size_col}' non trovata nel DataFrame.")
    size_counts = out[size_col].value_counts(dropna=False)
    valid_sizes = size_counts[size_counts >= min_count].index
    return out[out[size_col].isin(valid_sizes)].copy()


def filter_height_range(
    df: pd.DataFrame,
    *,
    height_cm_col: str = "height_cm",
    min_height: float = 140,
    max_height: float = 200,
) -> pd.DataFrame:
    """Filtra altezze fuori range (outlier / input errati)."""
    out = df.copy()
    if height_cm_col not in out.columns:
        raise KeyError(f"Colonna '{height_cm_col}' non trovata nel DataFrame.")
    return out[(out[height_cm_col] >= min_height) & (out[height_cm_col] <= max_height)].copy()


def create_height_bucket(
    df: pd.DataFrame,
    *,
    height_cm_col: str = "height_cm",
    bins=(0, 160, 170, 180, 250),
    labels=("<160", "160-170", "170-180", "180+"),
) -> pd.DataFrame:
    """Crea `height_bucket` (categorica) a partire da `height_cm`."""
    out = df.copy()
    if height_cm_col not in out.columns:
        raise KeyError(f"Colonna '{height_cm_col}' non trovata nel DataFrame.")
    out["height_bucket"] = pd.cut(out[height_cm_col], bins=bins, labels=labels)
    return out


def clean_dataset(
    df: pd.DataFrame,
    *,
    min_size_count: int = 50,
    min_height: float = 140,
    max_height: float = 200,
) -> pd.DataFrame:
    """Pipeline di cleaning usata in tutto il progetto.

    1) mismatch flag
    2) parsing altezza
    3) filtro taglie rare
    4) filtro outlier altezza
    5) bucket altezza
    """
    out = df.copy()

    out = add_mismatch_flag(out)
    out["height_cm"] = out.get("height").apply(parse_height_to_cm)

    out = filter_valid_sizes(out, min_count=min_size_count)
    out = filter_height_range(out, min_height=min_height, max_height=max_height)
    out = create_height_bucket(out)

    return out
