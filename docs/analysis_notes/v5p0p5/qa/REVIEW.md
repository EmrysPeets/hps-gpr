# v5.0.5 delivery checks

The final note has 253 pages. All pages were rendered, and the revised text,
tables, figures and appendix pages were inspected at higher resolution.
Figure 1 retains the original published vector curves. Figure 2 uses the
archived numerical inputs with revised layout and colors. The small toy
catalogues were enlarged to full portrait pages after the first layout check.

No unresolved references, undefined labels, duplicated labels or overfull
boxes remain. The 242 inherited result files are byte-identical to the parent.
All 180 new binning-fit coordinates converged with positive expectations.
The nominal 76 MeV combined signed root is 0.165569, reproducing the archived
0.166 after rounding.

A separate Tectonic 0.15.0 build in the Git worktree reproduced the text of
every page and the pixels of every page rendered at scale 0.6. PDF byte hashes
differ between builds because of metadata; the delivered PDF is identified by
the package manifest. Tectonic reports newer PDF input headers and an internal
bibliography rerun warning; these did not leave unresolved content or change
the independent build's text or rendered pages.

The source build needs Tectonic and its normal TeX resources. Regenerating the
Python figures additionally needs numpy, scipy, pandas, matplotlib, PyMuPDF and
shapely. Cached Python modules, font caches and platform-specific dependency
binaries are excluded from the portable delivery; install these libraries in
the target Python environment. Exact font rendering depends on installed fonts.

These are document and bounded-diagnostic checks. They do not add stress-field
or global-tail calibration to the new binning choices or alter the historical
qualification status of the observed results.
