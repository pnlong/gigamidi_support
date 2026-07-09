# Dataset adapters

Python adapters in this directory implement the shared `VADatasetSource` interface (`base.py`) for **DEAM**, **Memo2496**, and **MERP**. They define where raw files live, how V/A annotations are loaded, and where derived artifacts are written.

**For the full raw → converted-MIDI pipeline (paper reference), see [../PIPELINE.md](../PIPELINE.md#dataset-processing-methodology-paper-reference).**

| Adapter | File | Songs | Native V/A rate | Annotation start |
|---------|------|-------|-----------------|------------------|
| DEAM | `deam.py` | ~1802 | 2 Hz (500 ms) | 15 s |
| Memo2496 | `memo2496.py` | ~2496 | 1 Hz (1000 ms) | 0 s |
| MERP | `merp.py` | 54 | 10 Hz (100 ms) | 0 s |

```python
from datasets import get_dataset
ds = get_dataset("deam")  # or "memo2496", "merp"
v, a = ds.load_audio_va_annotations("1000")  # {time_sec: value} dicts
```

Implementation details that differ by dataset (especially MERP annotation source choice) are documented in the pipeline methodology section, not duplicated here.
