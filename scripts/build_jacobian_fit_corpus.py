#!/usr/bin/env python3
"""Build a pretrain-like prompt corpus for Jacobian-lens fitting.

Anthropic averages J over ~1000 prompts from a pretraining-like distribution.
Fitting on synthetic canaries biases the lens — prefer FineWeb (web pretrain
mix) or WikiText, not dossier / Playground text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _normalize(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _window(text: str, *, min_chars: int, max_chars: int, rng) -> str | None:
    t = _normalize(text)
    if len(t) < min_chars:
        return None
    if len(t) <= max_chars:
        return t
    # Random start so long docs don't always contribute the same prefix.
    start = rng.randint(0, len(t) - max_chars)
    return t[start : start + max_chars].strip()


def sample_fineweb(
    *,
    n: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    config: str = "sample-10BT",
    scan_multiplier: int = 20,
) -> list[dict]:
    """Stream FineWeb and take one window per document (diverse, no full download)."""
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=config,
        split="train",
        streaming=True,
    )
    # Shuffle buffer so we don't take only the first shard's head.
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    limit = max(n * scan_multiplier, n + 100)
    print(
        f"[fit-corpus] streaming HuggingFaceFW/fineweb:{config} "
        f"(target={n}, scan_cap={limit})",
        flush=True,
    )
    for row in ds:
        scanned += 1
        if scanned % 500 == 0:
            print(f"[fit-corpus] scanned={scanned} kept={len(out)}", flush=True)
        piece = _window(
            str(row.get("text") or ""),
            min_chars=min_chars,
            max_chars=max_chars,
            rng=rng,
        )
        if piece is None or piece in seen:
            continue
        seen.add(piece)
        out.append(
            {
                "id": f"fineweb_fit_{len(out):04d}",
                "source": f"HuggingFaceFW/fineweb:{config}",
                "doc_id": row.get("id"),
                "url": row.get("url"),
                "text": piece,
            }
        )
        if len(out) >= n:
            break
        if scanned >= limit:
            break
    return out


def sample_wikitext103(
    *,
    n: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> list[dict]:
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    articles: list[str] = []
    buf: list[str] = []
    for row in ds:
        line = (row.get("text") or "").rstrip("\n")
        if not line.strip():
            continue
        if re.match(r"^\s*=\s[^=].*=\s*$", line) and buf:
            articles.append("\n".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        articles.append("\n".join(buf))

    rng.shuffle(articles)
    out: list[dict] = []
    seen: set[str] = set()
    for art in articles:
        piece = _window(art, min_chars=min_chars, max_chars=max_chars, rng=rng)
        if piece is None or piece in seen:
            continue
        seen.add(piece)
        out.append(
            {
                "id": f"wikitext103_fit_{len(out):04d}",
                "source": "wikitext-103-raw-v1",
                "text": piece,
            }
        )
        if len(out) >= n:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--source",
        choices=("fineweb", "wikitext103"),
        default="fineweb",
        help="Pretrain-like source (default: FineWeb sample-10BT).",
    )
    p.add_argument(
        "--fineweb-config",
        default="sample-10BT",
        help="FineWeb config name (default: sample-10BT).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl (default depends on --source).",
    )
    args = p.parse_args()

    if args.output is None:
        args.output = Path(
            "data/jacobian_fit/fineweb_n1000.jsonl"
            if args.source == "fineweb"
            else "data/jacobian_fit/wikitext103_n1000.jsonl"
        )

    if args.source == "fineweb":
        rows = sample_fineweb(
            n=args.n,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
            config=args.fineweb_config,
        )
    elif args.source == "wikitext103":
        rows = sample_wikitext103(
            n=args.n,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
        )
    else:
        raise SystemExit(f"unknown source {args.source}")

    if len(rows) < args.n:
        raise SystemExit(f"only got {len(rows)} prompts; need {args.n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens = [len(r["text"]) for r in rows]
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "source": args.source,
                "n": len(rows),
                "chars_min": min(lens),
                "chars_median": sorted(lens)[len(lens) // 2],
                "chars_max": max(lens),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
