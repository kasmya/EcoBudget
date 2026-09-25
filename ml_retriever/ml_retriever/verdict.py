"""Verdict-aware scoring for yes_no and comparison tasks.

The Phase 6 `judge_structured` scores these types on whether the required VALUES
appear in the answer, not on the actual yes/no or which-is-bigger verdict. That
overstates task success (it passes a bare "$799" for "is X pricier than Y?", and
even passes a wrong verdict, while failing a correct bare "no"). This module adds:

  * a deterministic VERDICT the system can emit from the two retrieved values
    (compare magnitudes in the direction the question asks), and
  * a verdict-aware judge that checks the emitted verdict against the ground
    truth computed from the corpus values.

Numeric attributes only (price, weight, camera MP, height, range, battery, ...).
Descriptive comparisons (chip, best_for, location, trip length given as a word
range) have no well-defined magnitude verdict and are left to fact-coverage.
Pure/model-free, so it is unit-tested directly.
"""
from __future__ import annotations

import re
from typing import Optional

GREATER = {"higher", "greater", "more", "larger", "heavier", "bigger", "taller",
           "longer", "ahead", "faster", "pricier", "costlier"}
LESS = {"lower", "less", "lighter", "smaller", "cheaper", "shorter", "slower",
        "fewer", "closer"}


def strip_names(text: str, names) -> str:
    """Remove entity names (e.g. 'iPhone 16', 'Pixel 9') from prose before
    parsing a magnitude, so the entity's model number is not mistaken for the
    value (the bug: 'The iPhone 16 weighs 170 grams' must parse 170, not 16)."""
    for n in names or ():
        if n:
            text = re.sub(re.escape(n), " ", text, flags=re.IGNORECASE)
    return text


def parse_magnitude(text: str, exclude_names=None) -> Optional[float]:
    """First numeric magnitude in a value string ('$1,299'->1299, '170 grams'->170,
    '48MP'->48, '632 meters'->632). Word-numbers / ranges ('two to four') -> None.
    `exclude_names` are stripped first so entity model numbers are not picked up."""
    if not text:
        return None
    text = strip_names(text, exclude_names)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    return float(m.group().replace(",", "")) if m else None


def all_magnitudes(text: str, exclude_names=None) -> list[float]:
    """Every numeric magnitude, in order (for parsing a model answer that
    concatenates two values, e.g. '$799, $799' -> [799, 799]). Entity names are
    stripped first so their model numbers are not counted."""
    text = strip_names(text or "", exclude_names)
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def question_direction(question: str) -> Optional[str]:
    """'greater' | 'less' | None, from the comparative word in the question."""
    toks = set(re.findall(r"[a-z]+", (question or "").lower()))
    if toks & GREATER:
        return "greater"
    if toks & LESS:
        return "less"
    return None


def _verdict(m1: float, m2: float, direction: str) -> str:
    """yes/no for 'is entity1 {direction} than entity2?' (strict; equal -> no)."""
    if direction == "greater":
        return "yes" if m1 > m2 else "no"
    return "yes" if m1 < m2 else "no"


def yes_no_ground_truth(question: str, val1: str, val2: str,
                        ent1: str = "", ent2: str = "") -> Optional[str]:
    """Ground-truth yes/no from the two corpus values and the question direction.
    Entity names are excluded so their model numbers are not parsed as values."""
    d = question_direction(question)
    m1, m2 = parse_magnitude(val1, [ent1]), parse_magnitude(val2, [ent2])
    if d is None or m1 is None or m2 is None:
        return None
    return _verdict(m1, m2, d)


def yes_no_predicted(question: str, answer: str, ent1: str = "", ent2: str = "") -> Optional[str]:
    """The verdict implied by the values the model actually emitted (first two
    magnitudes in the answer, in requirement order)."""
    d = question_direction(question)
    mags = all_magnitudes(answer, [ent1, ent2])
    if d is None or len(mags) < 2:
        return None
    return _verdict(mags[0], mags[1], d)


def comparison_ground_truth(question: str, ent1: str, val1: str,
                            ent2: str, val2: str) -> Optional[str]:
    """For 'which is {adj}: X or Y?', the entity that wins; None if not a
    which-question or the values are not numeric."""
    d = question_direction(question)
    m1, m2 = parse_magnitude(val1, [ent1]), parse_magnitude(val2, [ent2])
    if d is None or m1 is None or m2 is None or m1 == m2:
        return None
    win_first = (m1 > m2) if d == "greater" else (m1 < m2)
    return ent1 if win_first else ent2
