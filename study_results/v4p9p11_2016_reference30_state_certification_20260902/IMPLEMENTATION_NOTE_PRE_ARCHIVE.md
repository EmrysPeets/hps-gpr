# Pre-archive implementation correction

The first `archive` invocation stopped before creating any archive certificate.
`observed_gp_states_k12_reviewed.csv` does not carry `ls_lo` or `n_train`; those
geometry fields reside in each row's hash-verified `selected_source` CSV.  The
reader initially requested the absent summary columns and pandas raised a
`ValueError` before the first state certificate was evaluated.

No archive-classification, robust-repeat, signal, p-value, or limit artifact
was produced.  The runner was corrected to read `ls_lo`, `ls_hi`, and
`n_train` from the selected source named by each reviewed row.  Selection
rules, thresholds, inputs, and the frozen protocol/spec are unchanged.  The
support-30 low-control phase is repeated after the correction so every
downstream decision points to one execution-script hash.

Before the successful archive phase, the independent statistics reviewer also
required three provenance-only hardenings: robust selected states pass their
recorded optimizer LML into fixed-state closure; historical repair-source
counts are kept separate from warning-free-repeat fields; and deterministic
polish requires the absolute LML change to be within tolerance.  These edits
likewise change no data, threshold, candidate, or selection rule.  The control
phase is repeated once more under the final script hash.

Pre-correction script SHA-256:
`b8a7312a86d5f25eb64ba9da46834a0abd151cad97e968d93c9b3205699219b8`.

Pre-correction passing control-decision SHA-256:
`5eaf43d5b9155e20c868f43ef466a0de74fc5d6f9a05a310f8b77911c4743c7f`.
