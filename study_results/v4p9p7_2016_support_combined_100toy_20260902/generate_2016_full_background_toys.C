#include "../../root_macros/funcform/funcform_common.h"

namespace {
TF1* make_shift_sigpowexp_tail_2016(TH1* h, double fit_min,
                                    double fit_max, double meanw,
                                    double rmsw) {
  TF1* f = ff_make_shift_sigpowexp_tail(h, fit_min, fit_max, meanw, rmsw);
  if (f == nullptr) {
    return nullptr;
  }
  f->SetParameter(1, 5.0);
  f->SetParameter(2, std::max(1e-3, rmsw / 8.0));
  f->SetParameter(3, std::max(0.0, fit_min - 0.006));
  f->SetParameter(4, std::max(0.0, fit_min - 0.003));
  f->SetParameter(6, 0.8);
  f->SetParameter(7, -1.5);
  f->SetParLimits(1, 0.0, 40.0);
  f->SetParLimits(2, 1e-4, 0.4);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.020),
                  std::min(fit_min + 0.010, fit_max - 1e-3));
  f->SetParLimits(4, std::max(0.0, fit_min - 0.010),
                  std::min(fit_min + 0.050, fit_max - 1e-3));
  f->SetParLimits(5, 1e-3, 0.02);
  f->SetParLimits(6, -10.0, 10.0);
  f->SetParLimits(7, -10.0, 10.0);
  return f;
}
}  // namespace

void generate_2016_full_background_toys(
    const char* outfile =
        "study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/"
        "2016_10pct_shape_x10_background_toys_100.root",
    int n_toys = 100) {
  FuncFormJobConfig job;
  job.dataset_key = "2016";
  job.dataset_label = "HPS 2016 10% shape at full-exposure count scale";
  job.input_file =
      "study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/"
      "source_2016_10pct.root";
  job.hist_name = "h_Minv_General_Final_1";
  job.output_root = outfile;
  job.note_plot_stem =
      "study_results/v4p9p7_2016_support_combined_100toy_20260902/figures/"
      "source_2016_10pct_functional_truth_x10";
  job.fit_min = 0.030;
  job.fit_max = 0.210;
  job.toy_support_min = 0.0;
  job.toy_support_max = 0.300;
  job.scan_min = 0.030;
  job.scan_max = 0.210;
  job.n_toys = n_toys;
  job.toy_lumi_scale = 10.0;
  job.primary_target_chi2ndf = 2.0;
  job.validation_max_rel_diff_full = 0.05;
  job.validation_max_rel_diff_scan = 0.05;
  job.validation_max_abs_sideband_frac_diff = 0.02;
  job.allow_bernstein_primary_fallback = false;
  job.bernstein_tag = "fBern5";
  job.fit_min_scan = {0.030, 0.032, 0.034, 0.035, 0.040};

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fShiftSigPowTail", "shifted sigmoid*power*exp + tail",
                  true, true, make_shift_sigpowexp_tail_2016});
  defs.push_back({"fGenGammaThresh", "thresholded gen-gamma", true, true,
                  ff_make_gengamma_thresh});

  ff_run_job(job, defs);
}
