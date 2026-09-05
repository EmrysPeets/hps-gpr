# 2021 10% background profiling comparison

Frozen before numerical evaluation, 5 September 2026. This isolated diagnostic
compares profiling assumptions, without changing a released result or selecting
a model from a favorable observed limit. No new toys or unblinding.

## Common inputs

Use the attested v4.9.12 runtime, reviewed 2021 kernel states, archived integer
histogram, 36-300 MeV GP support, nominal signal resolution, moving +/-2.25 sigma
training exclusion and fitted window, and all 201 masses from 50 to 250 MeV.
Keep the released full-template yield per electron-channel epsilon squared
fixed. Apply the expanded note's electron/muon branching correction to every
displayed coupling curve; retain uncorrected values in the CSV.

## Three primary curves

1. Released Poisson likelihood with count-space Gaussian background constraint:
   lambda = b + L theta + A w, penalty theta.T theta / 2. b and C are the
   log-GP posterior count moments; L L.T is the actual conditioned covariance.
2. Direct latent log-GP profile: lambda = exp(g + R theta) + A w, with the same
   quadratic constraint and R R.T equal to the posterior log covariance. This
   retains the non-Gaussian positive background distribution rather than its
   count-space Gaussian approximation. g is the posterior log mean; exp(g) is
   the nominal/median background, while exp(g + diag(K)/2) is its count mean.
   Use exp(g) at theta=0 for the alternative's background-only Asimov spectrum.
   Record this small mean/median distinction, not silently re-center it.
3. Fixed background: lambda = b + A w, b fixed to the released sideband GP mean,
   with no background nuisance parameters. This is a conditional reference
   that treats the estimated mean as known, not an improved uncertainty model.

Use exact binned Poisson likelihoods, signed fits for diagnostics, nonnegative
physical signal strength for upper limits, and the same bounded piecewise
Cowan tilde-q CLs mapping at 90% confidence. The alternative is a penalized
likelihood in latent Gaussian coordinates; it does not insert a density
Jacobian for a change of nuisance coordinates. It is not Bayesian integration.

## Numerical and source checks

Reconstruct the original count moments and training LML. Match the native
histogram content and all frozen input hashes. Condition log covariance with
the same deterministic relative diagonal-loading sequence as the release;
record all loads and their induced change in count covariance. Never clip
eigenvalues. Fail on nonpositive expectations, failed fits, likelihood-nesting
violations, nonmonotone CLs traces beyond 5e-5, or roots missing CLs=0.1 by 2e-6.
Use a centered Poisson deviance to avoid cancellation of large log likelihoods,
scaled signal coordinates and analytic derivatives. Independently solve the
Gaussian-constraint likelihood with the same code as a numerical control, and
record its difference from the release rather than overwriting it. Check
derivatives by finite differences and the fixed model against an independent
one-dimensional Poisson likelihood solution. Preserve every mass point.

Display the already-selected 65, 71, 78 and 182 MeV diagnostics. Plot profiled
quantities only where a profile was performed. Use a common GP-mean residual
baseline throughout, rather than joining two different subtractions. Error
bars are counting-only, and GP bands are conditional marginal background
uncertainties, not independent residual errors or significance bands.

## Literature and limits of interpretation

- Frate et al., https://arxiv.org/abs/1709.05681, Eq. 4 and Sec. III:
  GP-constrained intensity models, positive log-intensity models, and explicit
  limitations of interpreting kernels/regularization as frequentist constraints.
- Cowan et al., https://arxiv.org/abs/1007.1727, Sec. 3.7:
  nuisance profiling, bounded statistics and the piecewise asymptotic mapping.
- HPS Collaboration, https://arxiv.org/abs/2212.10629, Sec. IV.1:
  a separate established option is a simultaneous Poisson signal plus positive
  exponential-polynomial background fit. It changes the functional model and
  support as well as profiling and is not the controlled comparison here.
- Dauncey et al., https://arxiv.org/abs/1408.6865:
  discrete profiling addresses uncertainty among candidate background forms.

These citations motivate model families, not this dataset's coverage. Both GP
profiles retain the same moving-mask vulnerability demonstrated in v4.9.12.5.
Agreement cannot validate sideband contamination, the kernel family, or the
partially unblinded model-selection history. The new limits remain conditional
asymptotic diagnostics, not a replacement exclusion. Use one nice +10 process,
one numerical-library thread, and preserve all parent artifacts byte-for-byte.
