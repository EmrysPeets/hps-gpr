# Independent fit audit, 12 September 2026

**No material numerical or physical-implementation bug found.** Reviewed `scripts/fit_models.py`, the inherited context and solver modules, the protocol, all 24 saved model rows, and the 459-cell individual shape grid. Replayed the saved objectives without running the authoring script or changing its outputs. Statistical tail calibration is outside this audit.

## Fixed experiment and normalization

All alternatives use the same counts, GP means and GP uncertainties at every saved bin. The union masks contain 102/84/24 bins for 2015/2016/2021. The code constructs each union once, excludes it from GP training, and holds its background mean, covariance factor and kernel fixed during profiling. The 2015 kernel retains the inherited endpoint handling.

The MC92 density conversion is computed once from fractional overlaps with native histogram bins inside plus/minus 1.64 MC sigma at 92 MeV. It does not change with fitted mass, width or beta. This implements the declared local diagnostic; it is not a fresh physical production calculation at each fitted mass. Relative to historical scaled92 density conversion, the factors are 0.9671643/0.9966972/1.0003000.

Gaussian templates use exact bin-integrated CDF differences, normalize over the full saved spectrum, then restrict to the fixed mask. They are not renormalized inside the fit window: the changing captured fraction is retained. Checked representative mass/width combinations give fractions below unity, approximately 0.9863--0.999996.

## Derivatives, objectives and bounds

Independent central differences checked both mass and width-fraction template derivatives at endpoints and a nonstationary interior point. Maximum relative discrepancy was 4.2e-9. The normalized-template derivative includes the derivative of its full-support normalization. The outer beta derivative correctly uses log(E/2.3), and the width gradient includes the derivative of the assumed quadratic penalty. Profiling the inner convex amplitude/background fit permits these envelope derivatives.

Re-evaluating all 24 models at their saved parameters reproduces NLL within 1.8e-13. Saved Q values match 2(NLL_null-NLL_fit) within 3.1e-14; sqrtQ agrees within floating-point precision. The shared null is NLL=124.58231158105455. Saved expectations and fitted backgrounds are positive. The author's maximum inner score is 8.52e-8 and outer projected score 6.19e-7. These are convergence checks, not a proof of every nonconvex global optimum; multiple starts and grid-based seeds provide additional support.

The 459 shape-grid cells are unique and complete for three datasets, 17 means, and nine widths. Means remain in 90--94 MeV, width fractions in 0--1, and beta in -6--6. The widths interpolate between fixed MC92 and scaled92 endpoints, never below MC. The reported energy/independent-mass/profile-width solution has a 2021 lower-width boundary; that must remain visible in interpretation. Grid penalized Q agrees with max(0,Q-t^2) within 1.8e-15.

## Nesting and the width penalty

Within a common width/mean/penalty family, the rate alternatives satisfy common -> free-energy -> independent amplitudes. Likewise fixed -> shared -> independent means enlarge the allowed mean domain within a family. The independently fitted amplitude model also checks the expected near-saturation of the two-parameter energy-rate description.

**The unpenalized fully scaled control is not an ordinary nested point of the penalized width objective.** Setting all three t values to one in the profiled model adds 1.5 NLL, or three units of Q penalty; the fully scaled control omits that penalty. The MC t=0 control is the zero-penalty nested endpoint. State the change in assumed objective when comparing scaled controls with MC-centered profiling. The penalty is a diagnostic assumption, not a measured calibration likelihood.

## Historical 92 MeV versus the fixed union

The original 92 MeV calculation is reproduced exactly: sqrtQ=2.520256220, common epsilon-squared-equivalent=3.096134969e-6, and common-to-separate penalty=11.552480746, on 86/68/17 bins. Enlarging to the union while retaining historical density gives sqrtQ=2.453153002. Freezing the MC92 density then gives the new baseline sqrtQ=2.446642493. Thus the legacy/new baseline difference mostly reflects the changed mask and GP conditioning, with a smaller conversion effect; it is not caused by mass/width profiling.

`consults/audit_history.json` retains these controls, the parent free-energy historical comparison, and hashes of the audited inputs. Absolute NLL values from historical and union experiments are not interchangeable. The main fixed-union roots 2.4466, 4.1802, 4.5002 and 4.5978 are correctly reproduced as objective summaries.

Keep the proxy-amplitude and conditional-root labels: `p_naive_1d` is not a calibrated p-value for the expanded mass/width/energy alternatives. Per-campaign means are bounded descriptive offsets, not independent physical particle masses or measured calibration shifts. `signal_events` counts the signal inside the fixed fitted bins; curvature errors condition on the fitted shape/rate parameters. The inherited 2015 extension and background qualifications remain applicable.
