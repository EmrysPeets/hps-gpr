# Saved reference metadata finalization

The complete combined 74 MeV attempt-2 recovery finished generation, all fits,
CLs inversion, and validation before its final reference-ledger assembly raised
`KeyError: 'batch_id'`. The saved scalar-check list contains 72 extended
audit rows using the legacy labelled schema and 3,700 batch-reference rows using
the batch-id schema. The final assembly incorrectly indexed both as batch rows.

`finalize_reference_metadata.py` applies the unchanged frozen recovery audit to
the batch-reference rows and preserves all 3,772 rows in the numerical result.
Exactly one fallback must match exactly one successful row by batch id, method,
toy index, and window-count SHA-256. All original convergence and r/q agreement
gates remain required. The supplemental execution audit and original collector
still check the entire saved numerical result, including the legacy rows.

The finalized result is written to a separate directory. The failed run,
unverified result, traceback, numerical ledger, and all frozen inference sources
are preserved. The derivative contract freezes those inputs, this protocol, and
the finalizer. A verification function requires every original numerical/result
field to equal the saved unverified result exactly; only finalization and
reference-ledger metadata may be added. No toy is generated, fit repeated,
endpoint changed, check relaxed, or observation selected by this operation.
