# Verdict-aware re-measurement (frozen test)

Yes/no and 'which is X' comparison tasks are scored on the CORRECT VERDICT (computed from the two corpus values and the question direction, checked against the verdict implied by the values the system emitted), not on required-fact presence. Descriptive/non-numeric cases fall back to fact-coverage and are marked.

| type | n | fact-coverage (old) | verdict accuracy (new) |
|---|---|---|---|
| yes_no | 64 | 1.000 | 64/64 = 1.000 |
| comparison | 28 | 0.821 | 1/3 = 0.333 |
| single_fact | 16 | 1.000 | n/a (fact-coverage) |
| multi_part | 16 | 0.500 | n/a (fact-coverage) |
| procedure | 4 | 0.000 | n/a (fact-coverage) |

**Headline shift.** yes_no fact-coverage 1.000 -> verdict accuracy 1.000 on 64 numeric yes/no tasks. comparison verdict accuracy 1/3.

**Honest overall success (verdict where defined, else fact-coverage): 0.859** over 128 tasks. (The old fact-coverage-only headline was 0.895.)

Sample verdict checks (question | emitted values | verdict):

- WRONG: 'Which building is taller: the Burj Khalifa or the Empire Sta' -> '828 meters (2,717 feet),'
- OK: 'Which country has a larger population: India or Japan?' -> 'roughly 1.4 billion peop'
- WRONG: 'Which EV has the longer range: the Tesla Model 3 or the Niss' -> 'roughly 309-363 miles de'
- OK: 'Does the iPhone 16 have a higher price than the OnePlus 12?' -> '$799, $799'
- OK: "Is iPhone 16's price higher than OnePlus 12's?" -> '$799, $799'
- OK: 'Does iPhone 16 have a greater price than OnePlus 12?' -> '$799, $799'
- OK: 'Is iPhone 16 ahead of OnePlus 12 on price?' -> '$799, $799'
- OK: 'Does the iPhone 16 have a higher weight than the Google Pixe' -> '170 grams, 198 grams'
- OK: "Is iPhone 16's weight higher than Google Pixel 9's?" -> '170 grams, 198 grams'
- OK: 'Does iPhone 16 have a greater weight than Google Pixel 9?' -> '170 grams, 198 grams'
- OK: 'Is iPhone 16 ahead of Google Pixel 9 on weight?' -> '170 grams, 198 grams'
- OK: 'Does the OnePlus 12 have a higher weight than the Samsung Ga' -> '220 grams, 232 grams'
