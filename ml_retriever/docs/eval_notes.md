# Evaluation notes

## Decomposer (Phase 2) — attribute snap-to-vocab policy

**What it is.** `TaskDecomposer(attribute_vocab=...)` snaps each generated
attribute to the nearest slug in the corpus's closed 35-attribute vocabulary
(`ml_retriever/decomposer.py: snap_attribute`). This is constrained decoding
over a known finite label set, not a repair of wrong decisions.

**Scope & threshold.**
- **Attribute-only.** Entities are **never** snapped — a wrong entity is a real
  failure the metric must show.
- Snap first takes an exact match on the normalized form (lowercase, drop glue
  words {of,the,a,an}, token-sort); otherwise the nearest slug by Levenshtein
  distance, accepted only if within `0.4 * max(len(pred_norm), len(slug_norm))`.
  Far-off predictions are left unchanged rather than force-snapped.

**Why it's honest here (verification, base+LoRA on the 36-example val split).**
Failure analysis of the 22 missed requirements (of 60) shows the errors are
overwhelmingly *spelling corruption of the correct attribute* (e.g.
`display_refresh_rate` → `displays_recover_rate`) on the second entity of
comparison questions — not entity-type-as-attribute confusion (that was ~1
outlier). Verification that snap does not rescue wrong intent:
- Conceptual errors (genuinely different attribute, e.g. `shoe`/`light_pair`
  for `weight`): **6/6 remain wrong after snap.**
- Dropped-entity requirements: **6/6 remain dropped after snap** (snap cannot
  create a missing requirement).
- 5 cases where snap mapped a garble to the gold slug were confirmed to be
  severe corruptions of the *correct* attribute (`displays_recover_rate` →
  `display_refresh_rate`, `noises_cancelment` → `noise_cancellation`): the
  question asks about that attribute, the garble shares the slug's word-roots,
  and the same attribute was emitted correctly for the pair's first entity.
  These are legitimate constrained-decoding fixes. **No wrong-intent false
  rescue was found.**

**Metrics (val, base+LoRA, config: flan-t5-base, LoRA r=32 α=64 q,v,k,o,
20 epochs, 300 train pairs).**
- exact-match, un-snapped: **0.472**
- exact-match, snapped: **0.75**
- per-requirement accuracy (intent), un-snapped: **0.63**
- entity_f1 ≈ 0.89, attribute_f1 ≈ 0.87 (snapped)

## Phase 3 retrieval — known issue: entity dominance / attribute collision

For the requirement `(Oberoi Rajvilas Jaipur, amenities)`, the correct
amenities passage is **not in the bi-encoder top-5**: the same hotel's `rating`
passage ("5-star hotel…", sim 0.745) and `price_per_night` (0.694) outrank it,
and the only amenities passage in the top-5 is the *wrong hotel's* (Ibis, 0.543).
The MiniLM query "Oberoi Rajvilas Jaipur amenities" is dominated by the entity
name, so it can't localize the attribute within the entity. The cross-encoder
re-rank does **not** fix it (top-1 stays `rating`). This is the same
entity-dominance effect seen in Phase 4 sufficiency scoring.

Impact: the pipeline fed the T38 list task a `rating` passage, so the answerer
returned "5-star" — a retrieval failure, NOT an answerer failure (given the
correct evidence, both small and base enumerate the amenities correctly).

Proposed Phase 3 fix (not yet implemented): entity-aware retrieval — when a
requirement names an entity, restrict/boost candidates to passages of that
entity, then rank by attribute similarity. Offline we have entity metadata;
a live fetcher would fetch per-entity anyway. Tracked, not fixed here.

**Gate decision.** Snapped exact-match 0.75 is accepted as meeting the ≥0.65
decomposer gate, justified by the verification above: for a closed attribute
vocabulary, snapping is principled constrained decoding, and the residual
genuinely-wrong (conceptual + dropped) rate is ~11/60 ≈ 18% of requirements,
concentrated on second-entity comparison generation. The reported pipeline uses
the snapped decomposer.

## Answerer / judge (Phase 6, fork A)

**Gold format.** All comparison/yes_no/multi_part/list tasks are judged against
hand-curated `required_facts` (canonical values the source passages contain,
`scripts/curate_gold.py`) — NOT the `expected_answer` verdict string, which is
retained display-only. Training targets and both eval metrics use this one gold,
so train == eval.

**Judge formatting normalizer** (`judge._normalize`, symmetric on gold & pred):
- lowercase, collapse whitespace
- strip `$` and thousands commas: `$1,099` == `1099`
- number-unit spacing: `120Hz` == `120 Hz`
- trailing `.0` decimals: `799.00` == `799`, `10.0 ounces` == `10 ounces`
- drop leading articles: `the iPhone 15` == `iPhone 15`

Does NOT cover: synonyms/paraphrase (`oz` ≠ `ounces`), unit conversion, semantic
equivalence. Match threshold stays 1.0 (every required fact must match after
normalization).

**Metrics.** `judge_success` (binary: all facts matched) and `fact_f1`
(continuous per-fact hit rate = fraction of required_facts matched), reported
separately — the 0.5 gate is on `judge_success`, `fact_f1` shows how close.
