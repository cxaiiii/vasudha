from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterator
import random
import logging

from .loader import VasudhaDatasetLoader

logger = logging.getLogger(__name__)

@dataclass
class MixtureSpec:
    sources: Dict[str, float]  # dataset_name → sampling weight
    seed: int = 42
    buffer_size: int = 500


class StreamingDatasetMixture:
    """Weighted sampling mixture — no interleave_datasets, no feature resolution."""

    def __init__(self, spec: MixtureSpec):
        self.spec = spec

    def build(self) -> Iterator[dict]:
        """
        Build a weighted-sampling generator over all sources.

        Each source yields dicts with a 'text' key.  We keep one live iterator
        per source and sample from them according to their weights.  When a
        source is exhausted we drop it and renormalise.  This is pure Python,
        zero HF overhead, and starts yielding rows the instant the first
        source returns its first example.
        """
        active = {n: w for n, w in self.spec.sources.items() if w > 0}
        if not active:
            raise ValueError("All sources have weight 0 — nothing to train on.")

        rng = random.Random(self.spec.seed)

        # Build lazy iterators — each one is already mapped to {"text": ...}
        iterators: dict[str, Iterator] = {}
        weights: dict[str, float] = {}

        for name, weight in active.items():
            logger.info(f"Adding {name} to mixture with weight {weight}")
            loader = VasudhaDatasetLoader(
                source=name,
                buffer_size=self.spec.buffer_size,
                seed=self.spec.seed,
            )
            iterators[name] = iter(loader.load_normalized())
            weights[name] = weight

        logger.info(f"Mixture ready — {len(iterators)} sources, yielding rows on demand")

        while iterators:
            # Weighted random pick
            names = list(iterators.keys())
            w = [weights[n] for n in names]
            total = sum(w)
            probs = [x / total for x in w]
            chosen = rng.choices(names, weights=probs, k=1)[0]

            try:
                row = next(iterators[chosen])
                yield row
            except StopIteration:
                logger.info(f"Source exhausted: {chosen}")
                del iterators[chosen]
                del weights[chosen]

    def __repr__(self) -> str:
        return f"StreamingDatasetMixture(sources={list(self.spec.sources.keys())}, seed={self.spec.seed})"


def build_mixture(
    sources: dict[str, float],
    streaming: bool = True,
    seed: int = 42,
    buffer_size: int = 500,
) -> Iterator[dict]:
    """Convenience function to build a dataset mixture."""
    spec = MixtureSpec(sources=sources, seed=seed, buffer_size=buffer_size)
    mixture = StreamingDatasetMixture(spec)
    return mixture.build()
