#include "funcform_common.h"

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
  job.fit_min = 0.030;
  job.fit_max = 0.210;
  job.n_toys = n_toys;
  job.primary_target_chi2ndf = 2.0;
  job.allow_bernstein_primary_fallback = false;
  job.bernstein_tag = "fBern5";

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fGenGammaThresh", "thresholded generalized-gamma", true, true, ff_make_gengamma_thresh});
  defs.push_back({"fLogPolyThresh", "thresholded log-polynomial", true, true, ff_make_logpoly_thresh});
  defs.push_back({"fSigPowExpQ", "sigmoid * power * exp(linear + quadratic)", false, true, ff_make_sigpowexp_expquad});
  defs.push_back({"fBern5", "positive Bernstein fallback", false, true, ff_make_bern5});

  ff_run_job(job, defs);
}
