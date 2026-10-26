"""The one scalar/event output is metrics.jsonl. No renamed reward masquerades as success."""
from __future__ import annotations
import datetime
import json
import math
from pathlib import Path
import numpy as np

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return clean(x.item() if x.ndim == 0 else x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and (not math.isfinite(x)):
        return None
    return x

class Metrics:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, step, **values):
        record = clean(dict(event=event, step=int(step), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values))
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + '\n')

    def scalar_metrics(self, mapping):
        result = {}
        for k, v in mapping.items():
            if hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            try:
                a = np.asarray(v)
            except (ValueError, TypeError):
                continue
            if a.size == 1 and a.dtype.kind in 'bfiu':
                result[k] = clean(a.reshape(()))
        return result

class MeanMetrics:
    """Average update scalars, but aggregate explicit ratios by their weights.

    A pair ``name_numerator``/``name_denominator`` represents a sufficient
    statistic, not two update means. Both fields are summed across the log
    window and ``name`` is their quotient. An empty population is JSON null,
    so a window without positive rewards cannot look like perfect prediction.
    """

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, values):
        for k, v in values.items():
            if v is None:
                continue
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def pop(self):
        out = {k: v / self.counts[k] for k, v in self.sums.items()}
        for key in self.sums:
            if not key.endswith('_numerator'):
                continue
            name = key[:-len('_numerator')]
            denominator = name + '_denominator'
            if denominator not in self.sums:
                raise ValueError(f'Missing metric denominator for {name}')
            numerator, weight = self.sums[key], self.sums[denominator]
            if weight < 0:
                raise ValueError(f'Negative metric denominator for {name}')
            out[key], out[denominator] = numerator, weight
            out[name] = numerator / weight if weight else None
        self.sums.clear()
        self.counts.clear()
        return out


def action_summary(counts, prefix='real_action'):
    """Counts/fractions of actual acknowledged environment actions, in ID order."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or (counts < 0).any():
        raise ValueError('Action counts must be a nonnegative one-dimensional array')
    total = int(counts.sum())
    fractions = counts.astype(np.float64) / total if total else np.zeros_like(counts, dtype=np.float64)
    mode = int(counts.argmax()) if total else None
    return {
        prefix + '_counts': counts.tolist(),
        prefix + '_fractions': fractions.tolist(),
        prefix + '_steps': total,
        prefix + '_mode': mode,
        prefix + '_mode_fraction': float(fractions[mode]) if total else None,
    }
