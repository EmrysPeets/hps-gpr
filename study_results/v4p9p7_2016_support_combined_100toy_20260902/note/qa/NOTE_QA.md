# v4.9.7 analysis-note QA

Status: **pass**.

The canonical note is `../HPS_GPR_Analysis_Note_v4p9p7.pdf`, a 237-page,
letter-size PDF with SHA-256
`cc1a80878d915ad4ed8f2438c2fd5b613d7fae3ffc0793891a898188a91084a1`.
The mirror under `output/pdf/` is byte-identical.

## Semantic and build checks

- Extracted text contains no Unicode replacement character, unresolved `??`,
  `TODO`, `TBD`, or `[[` marker.
- The log contains no undefined reference, fatal TeX error, or overfull box.
- The package-level `lineno.sty` UTF-8 warning and three underfull boxes are inherited
  from the v4.9.6 source; the extracted PDF contains no replacement character.
- A source-only rebuild in a fresh temporary directory succeeded. It has the same 237
  pages and byte-identical extracted text (SHA-256
  `b9d0ae76e6b4c248f371ce6f1ad045178c2328171a68d71aceeb929d57d944b6`).
  PDF container bytes differ because the producer writes build metadata.

## Visual checks

- Poppler rendered all 237 pages at 90 dpi; ten contact sheets cover pages 1--237.
- Every contact sheet was visually inspected for blank pages, clipping, overlap,
  broken graphics, and gross layout regressions.
- Pages 1, 3, 11, 17, 71--82, 92--95, and 235--237 were also rendered at 180 dpi
  and inspected. These cover the title/abstract, contents, current change-log entry,
  scope, all new v4.9.7 support and signal-audit material, the historical v4.2 and
  v4.9.5 labels, the downstream gate/conclusions, and the bibliography end.
- The new Phase-1 table and all five new v4.9.7 figures are legible. No visible
  clipping or overlap was found.
- The page-94/page-95 break now begins page 95 with the complete statement that the
  v4.9.7 combined upper limit and 100-toy combined band are absent by construction.

## Interpretation checks

The note states the terminal no-edge outcome and the enforced absence of Phase 2,
the 65 MeV holdout, a support-specific full-2016 observed scan, a v4.9.7 combined
upper limit, and 100-toy combined bands. It keeps the accepted v4.2 combined result
and v4.9.5 2021-only result as distinct historical states. The 2021 comparison is
presented as a local-asymptotic robustness diagnostic, not as global significance,
coverage, or proof that the feature is signal or background.
