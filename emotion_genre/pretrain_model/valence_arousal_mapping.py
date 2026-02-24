"""
Valence–arousal mapping for XMIDI emotion classes.

Maps the 11 XMIDI emotion labels (same order as prepare_labels.EMOTIONS / emotion_to_index.json)
to (valence, arousal) pairs on roughly [-1, 1]. Based on circumplex model of affect.
"""

# Order must match EMOTIONS in prepare_labels.py: exciting, warm, happy, romantic, funny, sad, angry, lazy, quiet, fear, magnificent
EMOTION_INDEX_TO_VALENCE_AROUSAL = [
    (0.8, 0.9),   # 0: exciting  - high arousal, positive valence
    (0.7, 0.4),   # 1: warm     - positive valence, medium arousal
    (0.9, 0.7),   # 2: happy    - high positive, medium-high arousal
    (0.6, 0.3),   # 3: romantic - positive valence, medium arousal
    (0.8, 0.7),   # 4: funny    - positive valence, high arousal
    (-0.8, -0.6), # 5: sad      - negative valence, low arousal
    (-0.7, 0.8),  # 6: angry    - negative valence, high arousal
    (-0.3, -0.7), # 7: lazy     - low arousal, slightly negative
    (0.1, -0.6),  # 8: quiet    - low arousal, neutral/slight positive
    (-0.6, 0.7),  # 9: fear     - negative valence, high arousal
    (0.9, 0.85),  # 10: magnificent - high positive, high arousal
]


def get_va_for_emotion_index(emotion_index: int) -> tuple:
    """Return (valence, arousal) for emotion class index 0..10."""
    return EMOTION_INDEX_TO_VALENCE_AROUSAL[emotion_index]
