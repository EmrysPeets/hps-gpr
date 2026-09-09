# Analysis note v5.0.1 source

`main.tex` is the entry point for the complete 175-page v5.0.1 review draft,
including the revised abstract. `writing_sample.tex` is an inherited historical
entry point and is not the v5.0.1 note.

Keep this directory together with the sibling `figures/` and `derived/`
directories: the note uses relative paths to those bundled assets. The
bibliography is `hps_gpr_analysis_note.bib` in this directory.

The verified local build runs from this directory using Tectonic (XeTeX).
From the package directory, run `bash scripts/build_note.sh`; the result is
`qa/build/main.pdf`. A separate document-only rebuild passed with identical
page text and dimensions. See the package README and `qa/portable_build.json`.

For transfer to Overleaf, use the complete source package and select this
`main.tex` entry point. Preserve the relative directory structure. The supplied
PDF records the locally verified build; an Overleaf-hosted build has not been
run as part of this release.
