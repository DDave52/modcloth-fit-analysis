"""Metriche e tabelle di supporto alle decisioni (ModCloth / Start2Impact)."""

import pandas as pd


def mismatch_rate(df: pd.DataFrame) -> float:
    """Mismatch complessivo.

    Se esiste la colonna `mismatch` (0/1) usa quella; altrimenti deriva da `fit != 'fit'`.
    """
    if "mismatch" in df.columns:
        return df["mismatch"].mean()
    if "fit" not in df.columns:
        raise KeyError("Serve la colonna 'mismatch' o 'fit' per calcolare il mismatch rate.")
    return (df["fit"] != "fit").mean()


def fit_share(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Distribuzione di fit (fit/small/large) per gruppo."""
    out = (
        df.groupby(by)["fit"]
        .value_counts(normalize=True, dropna=False)
        .unstack()
        .fillna(0)
    )
    for c in ["fit", "small", "large"]:
        if c not in out.columns:
            out[c] = 0.0
    return out[["fit", "small", "large"]]


def add_group_counts(table: pd.DataFrame, df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggiunge la colonna `n` (conteggio osservazioni per gruppo)."""
    counts = df[by].value_counts(dropna=False).rename("n")
    return table.join(counts, how="left").sort_values("n", ascending=False)


def mismatch_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Tabella decisionale: n, share fit/small/large, mismatch (= small+large)."""
    share = fit_share(df, by=by)
    share["mismatch"] = share["small"] + share["large"]
    return add_group_counts(share, df, by=by)
