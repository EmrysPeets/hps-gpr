#include "../../root_macros/funcform/funcform_common.h"

// Build smooth, independent functional-form seed spectra on the exact
// 40--300 MeV production training support.  Pseudoexperiments are drawn by the
// Python study driver so that toy IDs can be paired across exposure scenarios.
void make_production_support_seed(
    const char* dataset_label,
    const char* input_file,
    const char* output_root,
    const char* plot_stem) {
  FuncFormJobConfig job;
  job.dataset_key = "2021";
  job.dataset_label = dataset_label;
  job.input_file = input_file;
  job.hist_name = "preselection/h_invM_8000";
  job.output_root = output_root;
  job.note_plot_stem = plot_stem;
  job.fit_min = 0.040;
  job.fit_max = 0.300;
  job.toy_support_min = 0.040;
  job.toy_support_max = 0.300;
  job.scan_min = 0.050;
  job.scan_max = 0.250;
  job.n_toys = 0;
  job.toy_lumi_scale = 1.0;
  job.primary_target_chi2ndf = 2.0;
  job.validation_max_rel_diff_full = 0.05;
  job.validation_max_rel_diff_scan = 0.05;
  job.validation_max_abs_sideband_frac_diff = 0.02;
  job.allow_bernstein_primary_fallback = true;
  job.bernstein_tag = "fBern5";
  job.fit_min_scan = {0.040, 0.045, 0.050, 0.055, 0.060};

  std::vector<FuncFormCandidateDef> defs;
  defs.push_back({"fSigPowExpQ", "sigmoid*power*exp + raw expquad", true, true,
                  ff_make_sigpowexp_expquad});
  defs.push_back({"fGenGammaThresh", "thresholded gen-gamma", true, true,
                  ff_make_gengamma_thresh});
  defs.push_back({"fGenGammaShift", "shifted gen-gamma", false, true,
                  ff_make_gengamma_shift});
  defs.push_back({"fEndpoint", "endpoint-aware sigmoid*power*exp", false, true,
                  ff_make_sigpowexp_endpoint});
  defs.push_back({"fBern5", "positive Bernstein fallback", false, true,
                  ff_make_bern5});

  ff_run_job(job, defs);
}

void make_all_production_support_seeds() {
  make_production_support_seed(
      "HPS 2021 1% functional-form seed",
      "/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root",
      "study_results/v4p1_2021_ls_exposure_ensembles_20260804/inputs/"
      "funcform_seed_2021_1pct_support040_300.root",
      "study_results/v4p1_2021_ls_exposure_ensembles_20260804/plots/"
      "funcform_seed_2021_1pct_support040_300");
  make_production_support_seed(
      "HPS 2021 10% functional-form seed",
      "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root",
      "study_results/v4p1_2021_ls_exposure_ensembles_20260804/inputs/"
      "funcform_seed_2021_10pct_support040_300.root",
      "study_results/v4p1_2021_ls_exposure_ensembles_20260804/plots/"
      "funcform_seed_2021_10pct_support040_300");
}

void make_production_support_seeds() {
  make_all_production_support_seeds();
}
