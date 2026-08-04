# Primary references and section architecture for the v4.1 follow-up

This is a research draft for the analysis-note update. It does not modify the
analysis-note worktree.

## Bottom line

The literature supports the following statements:

1. For an RBF/squared-exponential kernel, the length scale is the input-space
   separation over which the covariance decays. It is a model hyperparameter,
   not by itself a measured detector, signal, or background-physics scale.
2. A lower or upper hyperparameter bound defines the optimizer search domain.
   An optimum on a bound is evidence that the constraint is active; it is not
   evidence that the bound has been physically measured.
3. More training data can make the log-marginal-likelihood surface sharper and
   more informative about a length scale. There is no general result that the
   fitted length scale must increase monotonically with luminosity. The
   exposure dependence must be measured with a production-matched ensemble.
4. A statistically valid luminosity study scales an expected Poisson intensity,
   then fluctuates it. It must not multiply a realized observed histogram.
5. Hyperparameter closure, signal bias, expected sensitivity, interval
   coverage, and the observed physics result answer different questions. A
   single plot or ensemble should not be promoted across those categories
   without an explicit construction that supports both uses.

## Claim-to-reference map

| Claim in the note | Primary support | Scope caution |
| --- | --- | --- |
| RBF length scale controls covariance versus input separation; GP kernels encode assumptions about smoothness/correlation | `RasmussenWilliams2006`, especially Chapters 4 and 5; implementation-specific details in `ScikitLearnGPRDocs16` | Do not call the fitted length scale a physical correlation length without a separately justified mapping. |
| Hyperparameters are commonly selected by maximizing log marginal likelihood; the objective can have multiple local maxima | `RasmussenWilliams2006`, Sec. 5.4; `ScikitLearnGPRDocs16` | Repeated starts and a branch ledger address numerical local optima, not model misspecification. |
| More data can sharpen the marginal-likelihood landscape | `RasmussenWilliams2006`, Sec. 5.4 and Fig. 5.3 discussion | This does not prove that the optimum moves upward with luminosity. |
| HEP GP background models require kernel/model-selection and bias studies | `FrateGP2017`, `Gandrakota2023` | These papers motivate validation; they do not calibrate the HPS implementation or card. |
| Background-model flexibility and model choice should be tested for bias and coverage with toys | `Dauncey2015DiscreteProfiling` | Discrete profiling is not the HPS method; cite it for the validation principle, not as an equivalence. |
| Exposure-scaled event counts are Poisson draws with a scaled mean; nested exposure samples can be built from independent Poisson increments | `Kingman1993PoissonProcesses`; binned HEP likelihood context in `HistFactory` and `Junk1999` | If the fitted truth intensity is held fixed, the ensemble is conditional on that truth model. |
| Frequentist coverage is a repeated-sampling property of the complete interval procedure | `Neyman1937`, `FeldmanCousins1998` | Coverage of an extraction interval is not coverage of a later CLs upper limit. |
| The ensemble definition must say whether nuisance parameters are fixed or varied | `Conrad2003PoissonSystematics` | Mixing fixed-nuisance and varied-nuisance ensembles changes the coverage question. |
| Asimov data provide median expected sensitivity; this is distinct from finite-sample coverage | `Cowan2011` | An Asimov curve is neither an expected-limit band nor a coverage test. |
| CLs confidence levels and expected confidence levels for low-count Poisson searches | `Junk1999`, `Read2002` | CLs is modified frequentist; direct procedure-specific coverage still needs to be checked. |
| Toy-calculator implementation of frequentist tests and intervals | `Moneta2010RooStats` | Cite only if implementation context is useful; the HPS code does not become RooStats by analogy. |

## Exact prose recommended for the methodology

### Interpreting the length scale and its bounds

```tex
For the RBF covariance, $\ell$ is the characteristic separation in the GP input
coordinate over which the prior covariance decays. It therefore controls the scale
of variations admitted by the background model
\cite[Chs.~4--5]{RasmussenWilliams2006}; it is not, without an additional mapping, a
measurement of a detector or background-physics correlation length. The lower and
upper values supplied to the numerical optimizer define its admissible search
domain. Consequently, an optimum at $\ell_{\max}$ establishes that the constraint
is active for that fit, but does not establish that the physical background has a
correlation scale equal to the boundary.

The log marginal likelihood can have multiple local maxima, so the production
implementation uses repeated optimizer starts and retains the largest acceptable
log-marginal-likelihood branch
\cite{RasmussenWilliams2006,ScikitLearnGPRDocs16}. Rasmussen and Williams also show
that increasing the number of training observations can sharpen the
log-marginal-likelihood dependence on the characteristic length scale. This is a
motivation to revalidate the admissible range at higher exposure, not a theorem that
the fitted optimum must increase monotonically with luminosity.
```

### Statistically valid exposure scaling

```tex
The exposure study scales an expected event intensity rather than the realized
observed counts. If $\lambda_i^{10\%}$ denotes the validated 2021 10\% expected
count in bin $i$, an exposure factor $s$ defines
\[
  \lambda_i(s)=s\lambda_i^{10\%},
  \qquad
  N_i^{(t,s)}\sim\operatorname{Pois}\!\left[\lambda_i(s)\right].
\]
This is the binned form of Poisson-process exposure scaling
\cite{Kingman1993PoissonProcesses,HistFactory}. The construction
$N_i(s)=sN_i^{\mathrm{obs}}$ is not used: it propagates one observed fluctuation
deterministically and is not a pseudoexperiment from the scaled-exposure sampling
model.

For paired comparisons at
$0=s_0<s_1<\cdots<s_J$, independent increments
\[
  \Delta N_{ij}^{(t)}
  \sim\operatorname{Pois}\!\left[(s_j-s_{j-1})\lambda_i^{10\%}\right],
  \qquad
  N_i^{(t,s_j)}=\sum_{r=1}^{j}\Delta N_{ir}^{(t)}
\]
give the correct Poisson marginal at every exposure while correlating the toy
realizations in a controlled way. Every pseudo-spectrum is then passed through the
full production GP refit and signal-extraction chain.
```

### Conditional versus unconditional toy ensembles

```tex
The ensemble definition is part of the result. Holding the smooth truth intensity
and nuisance parameters fixed tests repeated sampling conditional on that truth.
Drawing nuisance parameters or truth-family parameters for each toy asks a different
question and must be reported separately
\cite{Conrad2003PoissonSystematics}. The primary hyperparameter study should
therefore report at least a GP self-closure ensemble and one or more independent
functional-form truth ensembles. Agreement in self-closure alone does not establish
robustness to background-model misspecification
\cite{FrateGP2017,Gandrakota2023,Dauncey2015DiscreteProfiling}.
```

## Recommended section architecture

### Section 4: Statistical model and numerical parameterization

1. **Kernel semantics and optimizer domain**
   - Define the RBF covariance and the log-mass coordinate.
   - State what $\ell$, $\ell_{\min}$, and $\ell_{\max}$ mean.
   - State explicitly that bound occupancy is a numerical diagnostic.
   - Cite `RasmussenWilliams2006` and the version-matched scikit-learn
     documentation.
2. **Observed v4 range diagnostic**
   - Report the controlled factors and plateau rule.
   - Label factor 12 as a post-v4 observed/asymptotic candidate.
   - State that observed limit direction and local $p_0$ did not define the
     plateau rule.
3. **Planned exposure-scaled generator**
   - Specify $\lambda_i(s)$ and Poisson draws.
   - Specify paired independent increments or paired common-random-number
     identifiers.
   - Freeze support, rebinning, exclusion geometry, and truth-family ledger.

### Section 5: Validation program, split into four noninterchangeable products

#### 5.1 Hyperparameter closure

Question: does the full refit recover a stable, non-boundary optimizer state as
exposure changes?

Required outputs:

- $\ell_{\mathrm{opt}}/\sigma_x$ and
  $\ell_{\mathrm{opt}}/\ell_{\max}$ distributions;
- upper- and lower-bound occupancy;
- log marginal likelihood and branch-repeat stability;
- optimizer failure rate;
- summaries versus exposure, mass, and truth family.

This stage can select a numerical range using a predeclared rule such as the
first nonbinding setting followed by a stable next-setting plateau. It does not
produce a limit or significance.

#### 5.2 Signal-bias and signal-absorption closure

Question: after a candidate range is fixed, does the background refit preserve
the null and recover injected signals?

Required outputs:

- zero-injection spurious-signal mean;
- injected-amplitude bias and RMS;
- pull mean and width;
- $\Delta Z$ or equivalent significance-calibration residual;
- injection recovery versus mass and strength;
- truth-family dependence.

Use independent smooth truth families as well as GP self-closure. Cite
`FrateGP2017`, `Gandrakota2023`, and `Dauncey2015DiscreteProfiling`.

#### 5.3 Expected sensitivity

Question: for the now-frozen card, what limit scale is expected under the
background-only model?

Keep two outputs distinct:

- an Asimov median sensitivity, citing `Cowan2011`;
- finite-toy median and central expected-limit intervals, only if and when the
  user authorizes limit bands.

Use new seeds after the hyperparameter-selection stage, or predeclare a
selection/evaluation split. Do not choose the length-scale range from whichever
expected or observed limit is strongest.

#### 5.4 Direct coverage

Question: how often does the reported interval or upper-limit procedure contain
the fixed injected truth under repeated sampling?

Define for each mass and injected strength

```tex
\widehat C(\mu,m)=
\frac{1}{N_{\mathrm{toy}}}
\sum_{t=1}^{N_{\mathrm{toy}}}
\mathbf{1}\!\left[\mu\in I_t(m)\right].
```

Report binomial Monte Carlo uncertainty on $\widehat C$. Apply the complete
procedure to every toy, including any data-driven branch repair or model/range
selection that is intended to remain part of the final procedure. Separate
fixed-nuisance from varied-nuisance ensemble definitions. Cite
`Neyman1937`, `FeldmanCousins1998`, and
`Conrad2003PoissonSystematics`.

### Section 6: Candidate observed/asymptotic diagnostics

- Show the exact observed v4.1 limit and fixed-mass asymptotic $p_0$.
- Show the matched v4 comparison.
- State that the analytic Sidak reference is conditional on a fixed card and
  does not account for the post-v4 range study.
- Do not show expected bands, empirical limit-tail quantities, or a
  toy-calibrated global significance.

### Section 7: Physics result gate

Only after Sections 5.1--5.4 pass should the note promote:

- the final frozen kernel card;
- observed and expected exclusion products;
- a coverage-qualified interpretation;
- any toy-calibrated scan-wise discovery probability.

Until then, use “observed/asymptotic candidate” and keep all physics-limit and
significance statements conditional.

## Suggested evidence ledger

| Product | Generator | Full GP refit? | Signal injected? | Selects card? | May support physics result? |
| --- | --- | ---: | ---: | ---: | ---: |
| Hyperparameter self-closure | GP truth | yes | optional | yes, by predeclared range rule | no |
| Hyperparameter misspecification closure | independent smooth truths | yes | optional | yes, by same rule | no |
| Signal-bias/absorption closure | GP and independent truths | yes | yes | no, after freeze | only as a gate |
| Asimov expected sensitivity | deterministic expectation | n/a | no | no | planning/median only |
| Expected-limit ensemble | background-only pseudoexperiments | as defined by final procedure | no | no | expected bands after authorization |
| Direct coverage | fixed injected truth with complete procedure | yes | yes | no | yes, as calibration evidence |
| Observed result | data | reviewed/frozen | unknown | no | yes only after gates |

## BibTeX: replace or enrich existing entries

The note already has `RasmussenWilliams2006`, `Gandrakota2023`,
`FrateGP2017`, `Cowan2011`, `Junk1999`, `Read2002`, and `HistFactory`.
The following are compatible replacements for the first two entries and add
missing primary metadata.

```bibtex
@book{RasmussenWilliams2006,
  author = {Rasmussen, Carl Edward and Williams, Christopher K. I.},
  title = {Gaussian Processes for Machine Learning},
  publisher = {MIT Press},
  address = {Cambridge, Massachusetts},
  series = {Adaptive Computation and Machine Learning},
  year = {2006},
  isbn = {978-0-262-18253-9},
  url = {https://gaussianprocess.org/gpml/}
}

@article{Gandrakota2023,
  author = {Gandrakota, Abhijith and Lath, Amitabh and Morozov, Alexandre V. and Murthy, Sindhu},
  title = {Model Selection and Signal Extraction Using Gaussian Process Regression},
  journal = {JHEP},
  volume = {02},
  pages = {230},
  year = {2023},
  doi = {10.1007/JHEP02(2023)230},
  eprint = {2202.05856},
  archivePrefix = {arXiv},
  primaryClass = {hep-ex},
  url = {https://arxiv.org/abs/2202.05856}
}
```

## BibTeX: recommended new entries

```bibtex
@article{Neyman1937,
  author = {Neyman, Jerzy},
  title = {Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability},
  journal = {Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences},
  volume = {236},
  number = {767},
  pages = {333--380},
  year = {1937},
  doi = {10.1098/rsta.1937.0005}
}

@article{FeldmanCousins1998,
  author = {Feldman, Gary J. and Cousins, Robert D.},
  title = {Unified Approach to the Classical Statistical Analysis of Small Signals},
  journal = {Phys. Rev. D},
  volume = {57},
  pages = {3873--3889},
  year = {1998},
  doi = {10.1103/PhysRevD.57.3873},
  eprint = {physics/9711021},
  archivePrefix = {arXiv},
  primaryClass = {physics.data-an},
  url = {https://arxiv.org/abs/physics/9711021}
}

@article{Conrad2003PoissonSystematics,
  author = {Conrad, Jan and Botner, Olga and Hallgren, Allan and Perez de los Heros, Carlos},
  title = {Including Systematic Uncertainties in Confidence Interval Construction for Poisson Statistics},
  journal = {Phys. Rev. D},
  volume = {67},
  pages = {012002},
  year = {2003},
  doi = {10.1103/PhysRevD.67.012002},
  eprint = {hep-ex/0202013},
  archivePrefix = {arXiv},
  primaryClass = {hep-ex},
  url = {https://arxiv.org/abs/hep-ex/0202013}
}

@article{Dauncey2015DiscreteProfiling,
  author = {Dauncey, P. D. and Kenzie, M. and Wardle, N. and Davies, G. J.},
  title = {Handling Uncertainties in Background Shapes: The Discrete Profiling Method},
  journal = {JINST},
  volume = {10},
  number = {04},
  pages = {P04015},
  year = {2015},
  doi = {10.1088/1748-0221/10/04/P04015},
  eprint = {1408.6865},
  archivePrefix = {arXiv},
  primaryClass = {physics.data-an},
  url = {https://arxiv.org/abs/1408.6865}
}

@book{Kingman1993PoissonProcesses,
  author = {Kingman, J. F. C.},
  title = {Poisson Processes},
  publisher = {Oxford University Press},
  address = {Oxford},
  series = {Oxford Studies in Probability},
  volume = {3},
  year = {1993},
  isbn = {978-0-19-853693-2},
  doi = {10.1093/oso/9780198536932.001.0001}
}

@inproceedings{Moneta2010RooStats,
  author = {Moneta, Lorenzo and Belasco, Kevin and Cranmer, Kyle and Kreiss, Sven and Lazzaro, Alfio and Piparo, Danilo and Schott, Gregory and Verkerke, Wouter and Wolf, Matthias},
  title = {The {RooStats} Project},
  booktitle = {Proceedings of the 13th International Workshop on Advanced Computing and Analysis Techniques in Physics Research},
  series = {Proceedings of Science},
  volume = {ACAT2010},
  pages = {057},
  year = {2010},
  doi = {10.22323/1.093.0057},
  eprint = {1009.1003},
  archivePrefix = {arXiv},
  primaryClass = {physics.data-an},
  url = {https://arxiv.org/abs/1009.1003}
}

@misc{ScikitLearnGPRDocs16,
  author = {{scikit-learn developers}},
  title = {Gaussian Processes: scikit-learn 1.6.1 Documentation},
  year = {2025},
  note = {Version 1.6.1; accessed August 4, 2026},
  url = {https://scikit-learn.org/1.6/modules/gaussian_process.html}
}
```

## Optional practical kernel reference

This is useful for plain-language interpretation of length scales, but the
canonical citation should remain Rasmussen and Williams.

```bibtex
@phdthesis{Duvenaud2014,
  author = {Duvenaud, David},
  title = {Automatic Model Construction with Gaussian Processes},
  school = {University of Cambridge},
  year = {2014},
  url = {https://www.repository.cam.ac.uk/handle/1810/247281}
}
```

## Citation-placement recommendations

- Cite `RasmussenWilliams2006` immediately after the first interpretation of
  $\ell$, not only in the historical introduction.
- Cite `ScikitLearnGPRDocs16` only for implementation facts: LML optimization,
  bounds, log-transformed hyperparameters, and repeated optimizer starts.
- Cite `Kingman1993PoissonProcesses` at the scaled-mean and independent-increment
  equations.
- Cite `Conrad2003PoissonSystematics` where the note distinguishes conditional
  fixed-truth toys from varied-nuisance ensembles.
- Cite `Dauncey2015DiscreteProfiling` in the validation section, not as support
  for selecting factor 12 from observed limits.
- Cite `Cowan2011` for Asimov median sensitivity and asymptotic test statistics,
  but not for direct finite-sample coverage.
- Cite `FeldmanCousins1998` and `Neyman1937` for the definition of repeated-
  sampling coverage. Continue to cite `Junk1999` and `Read2002` for CLs itself.

## Statements to avoid

- “The tenfold luminosity increase requires a larger length scale.”
- “Factor 12 is the measured physical correlation length.”
- “Zero boundary occupancy proves calibration.”
- “The Asimov curve demonstrates coverage.”
- “Hyperparameter toys are expected-limit bands.”
- “A fixed-card Sidak correction accounts for the factor scan.”
- “Scaling the 10% observed histogram by ten is a 100% pseudoexperiment.”

Prefer:

> The full-2016 v4 fits exposed an active numerical ceiling. Factor 12 is the
> first nonbinding observed/asymptotic candidate followed by a stable plateau.
> Production-matched exposure-scaled pseudoexperiments will determine how the
> distribution of the refitted length scale changes with event statistics and
> whether the candidate range remains nonbinding across independent truth
> families.
