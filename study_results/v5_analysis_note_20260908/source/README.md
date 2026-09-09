# Source status for the Harvard final-combinations derivative

`writing_sample.tex` is copied from the accepted reader-facing derivative and now
imports the replacement Section 6. The source is intentionally not buildable until
the reviewed numerical workflow supplies `../derived/generated_selected_results.tex`
and the four figures listed in the parent `README.md`.

Run Tectonic from this directory only after those generated inputs are frozen. The
strict `\includegraphics` and `\input` calls are intended to fail on a missing asset;
do not substitute placeholder numbers or historical result figures.
