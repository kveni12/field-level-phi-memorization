"""Load the dedicated pretraining-like corpus used to fit J-Lens."""
import json
from pathlib import Path
import random


def _load_fit_texts(path: Path, *, n: int, seed: int) -> tuple[list[str], str]:
    rows = [str(row['text']) for line in path.read_text().splitlines()
            if line.strip() and (row := json.loads(line)).get('text')]
    random.Random(seed).shuffle(rows)
    if len(rows) < n:
        raise ValueError(f'fit corpus {path} has {len(rows)} texts; need {n}')
    return rows[:n], f'file:{path}'
