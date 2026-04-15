#include "funcform_common.h"

namespace {
TF1* ff_make_shift_sigpowexp_tail_2015(TH1* h, double fit_min, double fit_max, double meanw, double rmsw) {
  TF1* f = ff_make_shift_sigpowexp_tail(h, fit_min, fit_max, meanw, rmsw);
  if (f == nullptr) {
    return nullptr;
  }
  f->SetParameter(1, 4.5);
  f->SetParameter(2, std::max(0.005, std::max(1e-3, rmsw / 7.0)));
  f->SetParameter(3, std::max(0.0, fit_min - 0.010));
  f->SetParameter(4, std::max(0.0, fit_min - 0.004));
  f->SetParameter(5, 0.0035);
  f->SetParameter(6, 1.5);
  f->SetParameter(7, -0.5);
  f->SetParLimits(1, 0.0, 40.0);
  f->SetParLimits(2, 1e-4, 0.4);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.030), std::min(fit_min + 0.005, fit_max - 1e-3));
  f->SetParLimits(4, std::max(0.0, fit_min - 0.015), std::min(fit_min + 0.020, fit_max - 1e-3));
  f->SetParLimits(5, 1e-3, 0.02);
  f->SetParLimits(6, -10.0, 10.0);
  f->SetParLimits(7, -10.0, 10.0);
  return f;
}
}

void make_func_data_output_2015(
    const char* outfile = "outputs/funcform_toys/funcform_2015_dataset_mod_toys.root",
    int n_toys = 100) {
  FuncFormJobConfig job;
  job.dataset_key = "2015";
  job.dataset_label = "HPS 2015 Functional-Form Fits";
  job.input_file = "/Users/emryspeets/research_plots/2015_data/invariant_mass_0pt5mm_full.root";
  job.hist_name = "invariant_mass";
  job.output_root = outfile;
  job.note_plot_stem = "hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2015";
  job.fit_min = 0.014;
  job.fit_max = 0.135;
  job.toy_support_min = 0.0;
  job.toy_support_max = 0.150;
  job.scan_min = 0.020;
  job.scan_max = 0.130;
  job.n_toys = n_toys;
  job.primary_target_chi2ndf = 4.0;
  job.validation_max_rel_diff_full = 0.05;
  job.validation_max_rel_diff_scan = 0.05;
  job.validation_max_abs_sideband_frac_diff = 0.02;
  job.allow_bernstein_primary_fallback = false;
  job.bernstein_tag = "fBern5";
  job.fit_min_scan = {0.014, 0.015, 0.018, 0.020};

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fShiftSigPowTail", "shifted sigmoid*power*exp + tail", true, true, ff_make_shift_sigpowexp_tail_2015});
  defs.push_back({"fShiftSigPow", "shifted sigmoid*power*exp", true, true, ff_make_shift_sigpowexp});
  defs.push_back({"fGenGammaShift", "shifted gen-gamma", false, true, ff_make_gengamma_shift});
  defs.push_back({"fSigPowExpQ", "sigmoid*power*exp + raw expquad", false, false, ff_make_sigpowexp_expquad});
  defs.push_back({"fEndpoint", "endpoint-aware sigmoid*power*exp", false, false, ff_make_sigpowexp_endpoint});

  ff_run_job(job, defs);
}
