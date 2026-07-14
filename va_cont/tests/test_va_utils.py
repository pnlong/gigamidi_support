"""Tests for VA alignment utilities."""

import numpy as np
import pytest

from va_utils import aggregate_va_to_bars, resample_va_dict
from pretrain_model.midi_features import handcrafted_features_from_bar_events, HANDCRAFTED_FEATURE_DIM


def test_handcrafted_features_from_events():
    events = [
        {"name": "Note_Pitch", "value": 60},
        {"name": "Note_Pitch", "value": 64},
        {"name": "Note_Velocity", "value": 80},
        {"name": "Note_Duration", "value": 480},
    ]
    feats = handcrafted_features_from_bar_events(events)
    assert feats.shape == (HANDCRAFTED_FEATURE_DIM,)
    assert feats[0] > 0  # note count


def test_resample_va_dict():
    v = {0.0: 0.0, 1.0: 1.0}
    a = {0.0: -1.0, 1.0: 0.5}
    t, rv, ra = resample_va_dict(v, a, target_hz=10.0, min_time=0.0)
    assert len(t) == 11
    assert rv[0] == pytest.approx(0.0)
    assert rv[-1] == pytest.approx(1.0)
    assert ra[0] == pytest.approx(-1.0)


def test_aggregate_va_to_bars():
    bar_resol = 480 * 4  # 4/4 bar
    ticks = np.array([0, 100, 200, bar_resol, bar_resol + 50], dtype=np.int32)
    valence = np.array([0.0, 0.2, 0.4, 1.0, -1.0], dtype=np.float32)
    arousal = np.array([0.0, 0.1, 0.3, 0.5, 0.7], dtype=np.float32)
    labels = aggregate_va_to_bars(ticks, valence, arousal, bar_resol, n_bars=2)
    assert len(labels) == 2
    assert labels[0][0] == 0
    assert labels[1][0] == 1
    assert labels[0][1] == pytest.approx(np.mean([0.0, 0.2, 0.4, 1.0]))


def test_deam_min_annotation_time():
    v = {10.0: 0.5, 20.0: 0.8}
    a = {10.0: -0.2, 20.0: 0.3}
    t, _, _ = resample_va_dict(v, a, target_hz=10.0, min_time=15.0)
    assert len(t) == 0
