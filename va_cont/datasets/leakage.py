"""Cross-dataset leakage mitigation for combined DEAM + MERP training.

MERP includes four DEAM anchor tracks (same audio as DEAM MEMD excerpts, different
song IDs). If one copy lands in train and the other in val, metrics are inflated.

We exclude MERP ``deam_*`` anchor IDs from all MERP splits when loading data.
DEAM numeric IDs (115, 343, 745, 1334) are kept in DEAM splits only.
"""

from __future__ import annotations

import logging

# MERP song_id → same audio as DEAM numeric song_id
MERP_DEAM_ANCHOR_SONG_IDS: frozenset[str] = frozenset({
    "deam_115",
    "deam_343",
    "deam_745",
    "deam_1334",
})

DEAM_ANCHOR_NUMERIC_IDS: frozenset[str] = frozenset({
    "115",
    "343",
    "745",
    "1334",
})

# Legacy alias used by merp.py / deam.py
MERP_DEAM_ANCHOR_IDS = MERP_DEAM_ANCHOR_SONG_IDS


def filter_songs_for_combined_training(dataset_name: str, song_ids: list[str]) -> list[str]:
    """
    Apply cross-dataset blacklist when building combined train/val loaders.

    Removes MERP DEAM-anchor duplicates; other datasets pass through unchanged.
    """
    if dataset_name != "merp":
        return song_ids
    filtered = [s for s in song_ids if s not in MERP_DEAM_ANCHOR_SONG_IDS]
    dropped = len(song_ids) - len(filtered)
    if dropped:
        logging.info(
            f"Leakage blacklist: removed {dropped} MERP DEAM-anchor song(s) "
            f"from {dataset_name} split"
        )
    return filtered
