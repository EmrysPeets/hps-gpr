#include "funcform_common.h"

void make_func_data_output_2015(
    const char* outfile = "outputs/funcform_toys/funcform_2015_toys.root",
    int n_toys = 100) {
  FuncFormJobConfig job;
  job.dataset_key = "2015";
  job.dataset_label = "HPS 2015 Functional-Form Fits";
  job.input_file = "/Users/emryspeets/research_plots/2015_data/invariant_mass_0pt5mm_full.root";
  job.hist_name = "invariant_mass";
  job.output_root = outfile;
  job.note_plot_stem = "hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2015";
  job.fit_min = 0.015;
  job.fit_max = 0.140;
  job.n_toys = n_toys;
  job.primary_target_chi2ndf = 2.0;
  job.allow_bernstein_primary_fallback = true;
  job.bernstein_tag = "fBern5";

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fEndpoint", "sigmoid * x^{a} * exp(-x/theta) * (1 - x/x_{max})^{b}", true, true, ff_make_sigpowexp_endpoint});
  defs.push_back({"fGenGammaShift", "shifted generalized-gamma", true, true, ff_make_gengamma_shift});
  defs.push_back({"fBern5", "positive Bernstein fallback", false, true, ff_make_bern5});

  ff_run_job(job, defs);
}
