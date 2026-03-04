"""Utility di caricamento dataset (portabilità repo GitHub).

Obiettivi:
- Evitare riferimenti a file locali assoluti.
- Supportare due modalità:
  1) se il file raw esiste già in data/raw -> lo usa
  2) altrimenti prova a scaricarlo da un URL pubblico (es. Hugging Face) e lo salva in data/raw

Nota: l'URL è lasciato volutamente come TODO (da sostituire con link HF pubblico).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# TODO: sostituisci con URL pubblico (Hugging Face) quando pronto
HF_DATASET: str = "https://huggingface.co/datasets/DDave5/modcloth-reviews-raw/resolve/main/modcloth_final_data.json"


def download_file(url: str, dest_path: Path, *, force_download: bool = False, timeout: int = 60) -> Path:
    """Scarica un file da URL e lo salva su disco.

    - Se dest_path esiste e force_download=False, non riscarica.
    - Usa stream per file grandi.
    """
    if dest_path.exists() and not force_download:
        return dest_path

    # Import locale per non imporre dependency se si usa solo il file locale
    import requests  # type: ignore

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return dest_path


def load_modcloth_raw(
    *,
    project_root: Path,
    local_relpath: Path = Path("data/raw/modcloth_final_data.json"),
    url: Optional[str] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Carica il dataset ModCloth raw (JSON Lines).

    Se il file locale esiste -> carica.
    Altrimenti -> scarica da url (o HF_DATASET_URL_TODO) e salva nel path standard.
    """
    local_path = (project_root / local_relpath).resolve()

    if local_path.exists():
        return pd.read_json(local_path, lines=True)

    url = url or HF_DATASET
    if "TODO" in url or url.strip().startswith("<<<"):
        raise ValueError(
            "URL dataset non impostato. Sostituisci HF_DATASET in src/data_loading.py "
            "oppure passa esplicitamente `url=...` a load_modcloth_raw()."
        )

    download_file(url, local_path, force_download=force_download)
    return pd.read_json(local_path, lines=True)
