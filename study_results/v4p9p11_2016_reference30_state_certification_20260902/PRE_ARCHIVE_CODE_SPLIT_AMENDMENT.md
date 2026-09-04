# Prospective code-split amendment before archive/state evaluation

This amendment was frozen after the support-30 low-control result was known but
before any archived-state certificate or robust production fit succeeded.  It
does not alter an input, fit, threshold, candidate, or decision rule.

The first and canonical control execution used script SHA-256
`b8a7312a86d5f25eb64ba9da46834a0abd151cad97e968d93c9b3205699219b8`.
That exact source is preserved as `run_control_frozen.py`.  Its canonical
decision is preserved byte-for-byte as
`derived/control_adequacy/control_decision_initial_frozen.json`, SHA-256
`5eaf43d5b9155e20c868f43ef466a0de74fc5d6f9a05a310f8b77911c4743c7f`.
The attempt and selected-cell ledgers have SHA-256 values
`8c6a3d9ef68ea36bfeb4f53e77f7770b8ddcbed422587655a4acc96d52231ac5`
and
`2515e88bdf0d372119b4a07b62e295c66adff2ab0e5c826c2c6607e7bef0edb0`.

The original monolithic runner was subsequently edited only to correct the
downstream archive reader and harden downstream certification.  Re-executions
of the control code produced byte-identical attempt and selected-cell ledgers,
although their timestamp-bearing decision JSON files differed.  Those later
decision JSON files are noncanonical and are not used downstream.

To remove the script-hash race, downstream execution is split into
`run_downstream_certification.py`.  Its CLI exposes only `archive` and
`robust`; it verifies the first control decision, ledgers, and preserved
control-script hash.  An AST-normalized comparison of the complete control
execution closure (17 functions/classes: hashing, input/card validation,
rebinning, resolution/bounds, fitting, prediction/covariance scoring,
repeat selection, and `run_control`) is identical between the preserved
control script and downstream script, with common SHA-256
`519c27e7fd71c81b41763127a777f5fa2b337f877fd4351f67534858b5011622`.

The only attempted archive invocation before this amendment stopped in pandas
column validation because two geometry fields were requested from the wrong
CSV.  It created no archive certificate or decision, selected no state, and
computed no signal or inference quantity.  The corrected downstream reader
obtains those fields from each row's already-declared, hash-verified source.

This is provenance/implementation hardening only.  The frozen protocol SHA
`bf3253ec...` and spec SHA `4c1c8355...` remain unchanged.
