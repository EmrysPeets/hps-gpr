#include "funcform_common.h"

void make_func_data_output_2021(
    const char* outfile = "outputs/funcform_toys/funcform_2021_toys.root",
    int n_toys = 100) {
  FuncFormJobConfig job;
  job.dataset_key = "2021";
  job.dataset_label = "HPS 2021 Functional-Form Fits";
  job.input_file = "/Users/emryspeets/root_files/tc_1pct/preselection_invM_psumlt2p8_hists.root";
  job.hist_name = "preselection/h_invM_8000";
  job.output_root = outfile;
  job.note_plot_stem = "hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2021";
  job.fit_min = 0.030;
  job.fit_max = 0.240;
  job.n_toys = n_toys;
  job.primary_target_chi2ndf = 2.0;
  job.allow_bernstein_primary_fallback = false;
  job.bernstein_tag = "fBern5";

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fSigPowExpQ", "sigmoid * power * exp(-x/theta + c_{1}x + c_{2}x^{2})", true, true, ff_make_sigpowexp_expquad});
  defs.push_back({"fSigPow", "sigmoid * x^{a} * exp(-x/theta)", true, true, ff_make_sigpowexp});
  defs.push_back({"fBern5", "positive Bernstein fallback", false, true, ff_make_bern5});

  ff_run_job(job, defs);
}
