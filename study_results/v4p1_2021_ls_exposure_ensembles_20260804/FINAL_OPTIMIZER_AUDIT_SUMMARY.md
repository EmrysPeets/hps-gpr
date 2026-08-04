# Final optimizer audit and bound-support conclusion

The post-repair audit passes.  All 600 predeclared nominal scan tasks are
complete, the reviewed table contains all 6,600 exact
truth--scenario--toy--mass--factor rows, and the attempt ledger records all
7,263 valid actual-fit candidates.  Exactly one candidate is selected for each
reviewed row.  No row is interpolated and no expected-limit band is
constructed.

Two targeted repair rounds supplied independent, salted actual fits at only
the flagged exact mass points.  The final selection uses 6,433 nominal rows and
167 repair rows (155 from round 1 and 12 from round 2).  No nested-likelihood
optimizer miss, exact initialization lock, or unresolved initialization state
remains.  The 16,500 nested comparisons comprise 13,285 allowed domain gains
and 3,215 likelihood plateaus.

## Bound-support conclusion

For the two 100%-exposure projections (`2021_1pct_x100` and
`2021_10pct_x10`) and both functional-form truth lanes, factor 15 still has
upper-bound occupancy:

| truth lane | projected sample | factor-15 at-bound / near-bound rows |
|---|---|---:|
| generalized-gamma threshold | 10% x 10 | 4 / 13 of 110 |
| generalized-gamma threshold | 1% x 100 | 14 / 19 of 110 |
| signal-like power exponential | 10% x 10 | 77 / 82 of 110 |
| signal-like power exponential | 1% x 100 | 38 / 48 of 110 |

Factors 20 and 25 have zero at-bound and zero near-bound rows in all four
projection groups.  Every one of the 440 exact factor-20 to factor-25
comparisons is a likelihood plateau within the predeclared numerical tolerance.
The largest positive LML changes from factor 20 to 25 are 0.001655, 0.001793,
0.000075, and 0.000027, respectively, and the factor-25 optimized normalized
length scales remain below 20.

Thus factor 20 is the smallest provisionally supported upper bound for the
requested 2021 100% projection pilot.  This is a hyperparameter-support and
optimizer-stability result, not a coverage, exclusion, discovery, or
expected-band result.

The native 2021 1% ensemble is a separate caveat: at factor 20 it retains
30/41 at-/near-bound rows in the generalized-gamma lane and 12/20 in the
signal-like lane, while factor 25 has none.  A single bound frozen across all
exposures would therefore require factor 25 in this pilot; factor 20 should not
be presented as a universal all-exposure choice.

## Review products and hashes

The nominal collector products are preserved.  The final reviewed table is
also published as a collector-compatible complete pair:

- `derived/scan_reviewed_rows_complete.csv`
  SHA-256 `16390d0e932609d9c5b8a7ed59b4848e7f596a9e8888b558323583857f9b5ebc`
- `derived/scan_reviewed_collection_complete.json`
  SHA-256 `1e288215670ab47adc76b6102342deddef025d0ee462ec8d5cf9708ea7e13ac0`

The complete-pair report has `kind=scan`, `partial=false`,
`incomplete_tasks=0`, an exact absolute output path, and the matching reviewed
CSV hash.  Its CSV is byte-identical to
`derived/scan_optimizer_reviewed_actual_rows.csv`.

Primary audit products:

- reviewed actual rows:
  `16390d0e932609d9c5b8a7ed59b4848e7f596a9e8888b558323583857f9b5ebc`
- actual-fit selection ledger:
  `01dc56bb40996379449980e9318d7b312843e11930a1ca74e66c9790b8d1d7f7`
- audit summary:
  `5333efd06f3e703a03c3eab9bba6d0ae1ea129046fb63b4ab62e1249a3e75178`
- nested-LML audit:
  `b8ce97a77d247b42274156b9c84e8ddd1ab54249f14174515b5cb2a165fd92c1`
- bound occupancy:
  `4a8368f6f2d0bdf8bdca70dd1a8d8e11e39eaaf41540720a5be358b17d242a29`
- empty final repair manifest:
  `961b55d97efb482c1a4e7ae20b11bcb247974e369fa124f798e35a7770ecf6c8`
- empty rejected-attempt ledger:
  `1afe47a2273d80555b1d91809882f2ad63214704a309245a04a452cc9320b9df`

The full study-local test suite passes: 27 tests.
