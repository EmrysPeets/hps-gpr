#include "funcform_common.h"

void make_func_data_output_2021(
    const char* outfile = "outputs/funcform_toys/funcform_2021_dataset_mod_toys.root",
    int n_toys = 100) {
  FuncFormJobConfig job;
  job.dataset_key = "2021";
  job.dataset_label = "HPS 2021 Functional-Form Fits";
  job.input_file = "/Users/emryspeets/root_files/tc_1pct/preselection_invM_psumlt2p8_hists.root";
  job.hist_name = "preselection/h_invM_8000";
  job.output_root = outfile;
  job.note_plot_stem = "hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2021";
  job.fit_min = 0.030;
  job.fit_max = 0.250;
  job.scan_min = 0.030;
  job.scan_max = 0.250;
  job.n_toys = n_toys;
  job.primary_target_chi2ndf = 2.0;
  job.validation_max_rel_diff_full = 0.05;
  job.validation_max_rel_diff_scan = 0.05;
  job.validation_max_abs_sideband_frac_diff = 0.02;
  job.allow_bernstein_primary_fallback = false;
  job.bernstein_tag = "fBern5";
  job.fit_min_scan = {0.030, 0.032, 0.035};

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fSigPowExpQ", "sigmoid*power*exp + raw expquad", true, true, ff_make_sigpowexp_expquad});
  defs.push_back({"fShiftSigPowTail", "shifted sigmoid*power*exp + tail", true, true, ff_make_shift_sigpowexp_tail});
  defs.push_back({"fShiftSigPow", "shifted sigmoid*power*exp", false, true, ff_make_shift_sigpowexp});
  defs.push_back({"fSigPow", "sigmoid*x^{a}*exp(-x/theta)", false, false, ff_make_sigpowexp});
  defs.push_back({"fBern5", "positive Bernstein fallback", false, false, ff_make_bern5});

  ff_run_job(job, defs);
}
