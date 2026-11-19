"""Algorithmic synthetic name and identifier generation with collision checks."""

from __future__ import annotations

import itertools
import random
import string
from dataclasses import dataclass

from memo_diff.dataset.rng import choice, make_rng


GIVEN_SYLLABLES = (
    "zor", "nel", "via", "keth", "ora", "quin", "bel", "sar", "mira", "tov",
    "ren", "lux", "pha", "dor", "yen", "kal", "ios", "ume", "bri", "sol",
    "tav", "eli", "nox", "ari", "fen", "jor", "syl", "cae", "vin", "hal",
    "pex", "rua", "gim", "tha", "wes", "ilo", "nar", "cy", "dru", "oma",
)

# Prefer readable given/family names for people-facing protected/public records.
FIRST_NAMES = (
    "Mira", "Cy", "Elena", "Jonah", "Asha", "Marcus", "Nora", "Theo", "Priya", "Owen",
    "Lena", "Samir", "Hannah", "Diego", "Clara", "Malik", "Ivy", "Noah", "Rhea", "Victor",
    "Sofia", "Amir", "Juliet", "Kai", "Nina", "Omar", "Grace", "Felix", "Tara", "Leo",
    "Anya", "Hugo", "Maya", "Seth", "Iris", "Adrian", "Lucia", "Ben", "Zoe", "Daniel",
)

LAST_NAMES = (
    "Solen", "Vale", "Harwick", "Ellison", "Brenner", "Quill", "Marsh", "Ashford",
    "Keller", "Ortega", "Nguyen", "Patel", "Brooks", "Sato", "Anders", "Moreno",
    "Whitaker", "Costa", "Rahman", "Singh", "Adler", "Kim", "Torres", "Bennett",
    "Hoffman", "Silva", "Cohen", "Park", "Diaz", "Fleming",
)

SURNAME_SYLLABLES = (
    "quill", "vance", "thorne", "wex", "bramble", "skarn", "hollow", "crest",
    "mire", "flint", "grove", "shard", "vale", "rook", "ashen", "cinder",
    "spire", "drift", "marsh", "glint", "fern", "cobble", "riven", "sable",
    "plume", "kestrel", "amber", "frost", "lantern", "copper", "onyx", "pearl",
)

ORG_PREFIXES = (
    "Aether", "North", "Cedar", "Iron", "Lumen", "Quartz", "Coral", "Atlas",
    "Helix", "Nimbus", "Prairie", "Obsidian", "Silver", "Crimson", "Verdant",
    "Astra", "Granite", "Cobalt", "Amber", "Polar", "Saffron", "Ember",
)

ORG_SUFFIXES = (
    "Labs", "Systems", "Works", "Institute", "Consortium", "Archives",
    "Foundry", "Observatory", "Collective", "Dynamics", "Holdings", "Bureau",
    "Station", "Assembly", "Conservancy", "Fleet", "Library", "Yard",
)

PLACE_MODIFIERS = (
    "Upper", "Lower", "East", "West", "Outer", "Inner", "New", "Old", "Port",
    "Cape", "Mount", "Lake", "Ridge", "Bay", "Isle", "Delta",
)

PLACE_ROOTS = (
    "Kaldrin", "Sorrel", "Virelia", "Thornwick", "Mossbarrow", "Cinderfall",
    "Bluefen", "Starreach", "Glassmere", "Ironbar", "Softwind", "Deeporchard",
    "Redharbor", "Quietspan", "Brightmarsh", "Stonewick", "Foghollow",
)

PRODUCT_TOKENS = (
    "Nexus", "Pulse", "Vector", "Cascade", "Orbit", "Lattice", "Signal",
    "Forge", "Prism", "Anchor", "Relay", "Harbor", "Keystone", "Ledger",
)


@dataclass
class NameRegistry:
    """Tracks generated strings to avoid collisions within a generation run."""

    entities: set[str]
    identifiers: set[str]
    texts: set[str]

    @classmethod
    def empty(cls) -> "NameRegistry":
        return cls(entities=set(), identifiers=set(), texts=set())

    def register_entity(self, name: str) -> str:
        if name in self.entities:
            raise ValueError(f"Entity collision: {name}")
        self.entities.add(name)
        return name

    def register_id(self, identifier: str) -> str:
        if identifier in self.identifiers:
            raise ValueError(f"Identifier collision: {identifier}")
        self.identifiers.add(identifier)
        return identifier

    def register_text(self, text: str) -> str:
        if text in self.texts:
            raise ValueError(f"Text collision: {text[:80]}")
        self.texts.add(text)
        return text


def _title_join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def synthesize_place_name(rng: random.Random, registry: NameRegistry, max_tries: int = 500) -> str:
    for attempt in range(max_tries):
        if rng.random() < 0.5:
            name = f"{choice(rng, PLACE_MODIFIERS)} {choice(rng, PLACE_ROOTS)}"
        else:
            name = choice(rng, PLACE_ROOTS)
        if attempt > 50 or name in registry.entities:
            # Expand beyond the finite root list with a synthetic sector code.
            name = f"{name} Sector-{rng.randint(10, 9999)}"
        if name not in registry.entities:
            return registry.register_entity(name)
    raise RuntimeError("Failed to synthesize unique place name")


def synthesize_org_name(rng: random.Random, registry: NameRegistry, max_tries: int = 500) -> str:
    for attempt in range(max_tries):
        name = f"{choice(rng, ORG_PREFIXES)} {choice(rng, ORG_SUFFIXES)}"
        if rng.random() < 0.3:
            name = f"{name} of {choice(rng, PLACE_ROOTS)}"
        if attempt > 40 or name in registry.entities:
            name = f"{name} {rng.randint(10, 9999)}"
        if name not in registry.entities:
            return registry.register_entity(name)
    raise RuntimeError("Failed to synthesize unique organization name")


def synthesize_person_name(rng: random.Random, registry: NameRegistry, max_tries: int = 500) -> str:
    for attempt in range(max_tries):
        # Readable person names by default (syllable mash only as collision fallback).
        if attempt < 400:
            name = f"{choice(rng, FIRST_NAMES)} {choice(rng, LAST_NAMES)}"
        else:
            first = choice(rng, GIVEN_SYLLABLES) + (choice(rng, GIVEN_SYLLABLES) if rng.random() < 0.45 else "")
            first = first[:1].upper() + first[1:]
            last = choice(rng, SURNAME_SYLLABLES)
            last = last[:1].upper() + last[1:]
            name = _title_join(first, last)
        if attempt > 80 or name in registry.entities:
            name = f"{name} {rng.choice(string.ascii_uppercase)}{rng.randint(10, 99)}"
        if name not in registry.entities:
            return registry.register_entity(name)
    raise RuntimeError("Failed to synthesize unique person name")


def synthesize_machine_name(rng: random.Random, registry: NameRegistry, max_tries: int = 500) -> str:
    for _ in range(max_tries):
        token = choice(rng, PRODUCT_TOKENS)
        suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
        number = rng.randint(10, 9999)
        name = f"{token}-{suffix}{number}"
        if name not in registry.entities:
            return registry.register_entity(name)
    raise RuntimeError("Failed to synthesize unique machine name")


def synthesize_product_name(rng: random.Random, registry: NameRegistry, max_tries: int = 500) -> str:
    for attempt in range(max_tries):
        name = f"{choice(rng, ORG_PREFIXES)} {choice(rng, PRODUCT_TOKENS)} {rng.randint(1, 99)}"
        if attempt > 30 or name in registry.entities:
            name = f"{name}-{rng.randint(100, 9999)}"
        if name not in registry.entities:
            return registry.register_entity(name)
    raise RuntimeError("Failed to synthesize unique product name")


def synthesize_identifier(
    rng: random.Random,
    registry: NameRegistry,
    prefix: str,
    length: int = 8,
    max_tries: int = 300,
) -> str:
    alphabet = string.ascii_uppercase + string.digits
    for _ in range(max_tries):
        body = "".join(rng.choice(alphabet) for _ in range(length))
        # Avoid purely numeric IDs that look like real SSN/MRN patterns.
        if body.isdigit():
            continue
        identifier = f"{prefix}-{body}"
        if identifier not in registry.identifiers:
            return registry.register_id(identifier)
    raise RuntimeError(f"Failed to synthesize unique identifier with prefix {prefix}")


def rarity_group_for_token(token: str) -> str:
    """Heuristic rarity band from token shape (used for matching strata)."""
    compact = token.replace(" ", "").replace("-", "")
    if len(compact) >= 14 or any(ch.isdigit() for ch in compact):
        return "rare"
    if len(compact) >= 9:
        return "uncommon"
    return "common"


def bulk_unique_check(values: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate value in bulk check: {value}")
        seen.add(value)


def demo_name_stream(seed: int, n: int = 20) -> list[str]:
    """Small helper used in tests to show collision-free generation."""
    rng = make_rng("demo-names", seed)
    registry = NameRegistry.empty()
    names = []
    generators = (
        synthesize_person_name,
        synthesize_org_name,
        synthesize_place_name,
        synthesize_machine_name,
        synthesize_product_name,
    )
    for i, gen in zip(range(n), itertools.cycle(generators)):
        names.append(gen(rng, registry))
    return names
