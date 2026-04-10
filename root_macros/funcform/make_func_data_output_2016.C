#include "funcform_common.h"

namespace {
TF1* ff_make_shift_sigpowexp_tail_2016(TH1* h, double fit_min, double fit_max, double meanw, double rmsw) {
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
  f->SetParLimits(3, std::max(0.0, fit_min - 0.020), std::min(fit_min + 0.010, fit_max - 1e-3));
  f->SetParLimits(4, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.050, fit_max - 1e-3));
  f->SetParLimits(5, 1e-3, 0.02);
  f->SetParLimits(6, -10.0, 10.0);
  f->SetParLimits(7, -10.0, 10.0);
  return f;
}
}

void make_func_data_output_2016(
    const char* outfile = "outputs/funcform_toys/funcform_2016_toys.root",
    int n_toys = 100) {
  FuncFormJobConfig job;
  job.dataset_key = "2016";
  job.dataset_label = "HPS 2016 Functional-Form Fits";
  job.input_file = "/Users/emryspeets/research_plots/2016_IMD/EventSelection_Data_10Percent.root";
  job.hist_name = "h_Minv_General_Final_1";
  job.output_root = outfile;
  job.note_plot_stem = "hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2016";
  job.fit_min = 0.035;
  job.fit_max = 0.210;
  job.n_toys = n_toys;
  job.primary_target_chi2ndf = 2.0;
  job.allow_bernstein_primary_fallback = false;
  job.bernstein_tag = "fBern5";
  job.fit_min_scan = {0.040, 0.035};

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fShiftSigPowTail", "shifted sigmoid*power*exp + tail", true, true, ff_make_shift_sigpowexp_tail_2016});
  defs.push_back({"fGenGammaThresh", "thresholded gen-gamma", true, true, ff_make_gengamma_thresh});
  defs.push_back({"fShiftSigPow", "shifted sigmoid*power*exp", false, true, ff_make_shift_sigpowexp});
  defs.push_back({"fSigPowExpQ", "sigmoid*power*exp + raw expquad", false, false, ff_make_sigpowexp_expquad});
  defs.push_back({"fLogPolyThresh", "thresholded log-polynomial", false, false, ff_make_logpoly_thresh});
  defs.push_back({"fBern5", "positive Bernstein fallback", false, false, ff_make_bern5});

  ff_run_job(job, defs);
}
