#ifndef HPS_GPR_FUNCFORM_COMMON_H
#define HPS_GPR_FUNCFORM_COMMON_H

#include "TCanvas.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TH1.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TMath.h"
#include "TNamed.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct FuncFormChi2Info {
  double chi2_pearson{0.0};
  double chi2_neyman{0.0};
  int nbin_used{0};
  int ndf_sel{0};
  double pearson_chi2ndf{-1.0};
  double neyman_chi2ndf{-1.0};
};

struct FuncFormCandidateDef {
  std::string tag;
  std::string label;
  bool preferred_primary{false};
  bool enabled{true};
  TF1* (*factory)(TH1*, double, double, double, double);
};

struct FuncFormFitSummary {
  std::string tag;
  std::string label;
  TF1* func{nullptr};
  bool ok{false};
  int ndf_root{0};
  double chi2ndf_root{-1.0};
  FuncFormChi2Info eval;
  double fit_min{0.0};
  double fit_max{0.0};
  int trial_index{-1};
  std::string trial_label;
};

struct FuncFormJobConfig {
  std::string dataset_key;
  std::string dataset_label;
  std::string input_file;
  std::string hist_name;
  std::string output_root;
  std::string note_plot_stem;
  double fit_min{0.0};
  double fit_max{0.0};
  int n_toys{100};
  double primary_target_chi2ndf{2.0};
  bool allow_bernstein_primary_fallback{false};
  std::string bernstein_tag{"fBern5"};
  std::vector<double> fit_min_scan;
};

inline std::string ff_json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

inline void ff_window_moments(const TH1* h, double xmin, double xmax,
                              double& mean, double& var, long long& nwin) {
  double sumw = 0.0;
  double sumx = 0.0;
  double sumx2 = 0.0;
  const int nb = h->GetNbinsX();
  int b1 = h->GetXaxis()->FindBin(xmin + 1e-12);
  int b2 = h->GetXaxis()->FindBin(xmax - 1e-12);
  b1 = std::max(1, b1);
  b2 = std::min(nb, b2);
  for (int i = b1; i <= b2; ++i) {
    const double x = h->GetXaxis()->GetBinCenter(i);
    const double w = h->GetBinContent(i);
    sumw += w;
    sumx += w * x;
    sumx2 += w * x * x;
  }
  nwin = static_cast<long long>(std::llround(sumw));
  if (sumw > 0.0) {
    mean = sumx / sumw;
    var = std::max(0.0, sumx2 / sumw - mean * mean);
  } else {
    mean = 0.08;
    var = 0.03 * 0.03;
  }
}

inline FuncFormChi2Info ff_eval_binavg_chi2(const TH1* h, TF1* f,
                                            double xmin, double xmax) {
  FuncFormChi2Info out;
  const int nb = h->GetNbinsX();
  int b1 = h->GetXaxis()->FindBin(xmin + 1e-12);
  int b2 = h->GetXaxis()->FindBin(xmax - 1e-12);
  b1 = std::max(1, b1);
  b2 = std::min(nb, b2);

  for (int i = b1; i <= b2; ++i) {
    const double obs = h->GetBinContent(i);
    const double x1 = h->GetXaxis()->GetBinLowEdge(i);
    const double x2 = h->GetXaxis()->GetBinUpEdge(i);
    const double bw = x2 - x1;
    if (bw <= 0.0) {
      continue;
    }

    const double integ = f->Integral(x1, x2);
    if (!std::isfinite(integ)) {
      continue;
    }
    double expv = integ / bw;
    if (!std::isfinite(expv)) {
      continue;
    }
    expv = std::max(expv, 1e-12);

    const double diff = obs - expv;
    out.chi2_pearson += (diff * diff) / expv;
    if (obs > 1e-12) {
      out.chi2_neyman += (diff * diff) / obs;
    }
    out.nbin_used++;
  }

  out.ndf_sel = std::max(1, out.nbin_used - f->GetNpar());
  if (out.nbin_used > 0) {
    out.pearson_chi2ndf = out.chi2_pearson / out.ndf_sel;
    out.neyman_chi2ndf = out.chi2_neyman / out.ndf_sel;
  }
  return out;
}

inline double ff_sigmoid(double x, double xt, double w) {
  if (w <= 0.0) {
    return 0.0;
  }
  return 1.0 / (1.0 + TMath::Exp(-(x - xt) / w));
}

inline double ff_clip_exp(double arg) {
  if (arg > 80.0) {
    return TMath::Exp(80.0);
  }
  if (arg < -80.0) {
    return TMath::Exp(-80.0);
  }
  return TMath::Exp(arg);
}

inline double ff_clamp_to_limits(TF1* f, int idx, double value) {
  double lo = 0.0;
  double hi = 0.0;
  f->GetParLimits(idx, lo, hi);
  if (hi > lo) {
    const double pad = std::max(1e-9, 1e-6 * (hi - lo));
    return std::min(std::max(value, lo + pad), hi - pad);
  }
  return value;
}

inline int ff_find_param_index(TF1* f, const char* name) {
  if (f == nullptr || name == nullptr) {
    return -1;
  }
  for (int i = 0; i < f->GetNpar(); ++i) {
    const char* pname = f->GetParName(i);
    if (pname != nullptr && std::string(pname) == name) {
      return i;
    }
  }
  return -1;
}

inline void ff_set_param_named(TF1* f, const char* name, double value) {
  const int idx = ff_find_param_index(f, name);
  if (idx < 0) {
    return;
  }
  f->SetParameter(idx, ff_clamp_to_limits(f, idx, value));
}

inline double ff_seed_value_named(TF1* f, const char* name, double fallback) {
  const int idx = ff_find_param_index(f, name);
  return (idx >= 0) ? f->GetParameter(idx) : fallback;
}

inline double ff_model_sigpowexp(double* xx, double* p) {
  const double x = xx[0];
  if (x <= 0.0) {
    return 0.0;
  }
  const double A = p[0];
  const double a = p[1];
  const double theta = p[2];
  const double xt = p[3];
  const double w = p[4];
  if (theta <= 0.0 || w <= 0.0) {
    return 0.0;
  }
  return A * ff_sigmoid(x, xt, w) * TMath::Power(x, a) * TMath::Exp(-x / theta);
}

inline double ff_model_sigpowexp_expquad(double* xx, double* p) {
  const double x = xx[0];
  if (x <= 0.0) {
    return 0.0;
  }
  const double A = p[0];
  const double a = p[1];
  const double theta = p[2];
  const double xt = p[3];
  const double w = p[4];
  const double c1 = p[5];
  const double c2 = p[6];
  if (theta <= 0.0 || w <= 0.0) {
    return 0.0;
  }
  return A * ff_sigmoid(x, xt, w) * TMath::Power(x, a) * TMath::Exp(-x / theta)
         * ff_clip_exp(c1 * x + c2 * x * x);
}

inline double ff_model_sigpowexp_endpoint(double* xx, double* p) {
  const double x = xx[0];
  if (x <= 0.0) {
    return 0.0;
  }
  const double A = p[0];
  const double a = p[1];
  const double theta = p[2];
  const double xt = p[3];
  const double w = p[4];
  const double xmax = p[5];
  const double b = p[6];
  if (theta <= 0.0 || w <= 0.0 || xmax <= x) {
    return 0.0;
  }
  const double tail = 1.0 - x / xmax;
  if (tail <= 0.0) {
    return 0.0;
  }
  return A * ff_sigmoid(x, xt, w) * TMath::Power(x, a) * TMath::Exp(-x / theta)
         * TMath::Power(std::max(tail, 1e-9), b);
}

inline double ff_model_shift_sigpowexp(double* xx, double* p) {
  const double x = xx[0];
  if (x <= 0.0) {
    return 0.0;
  }
  const double A = p[0];
  const double a = p[1];
  const double theta = p[2];
  const double x0 = p[3];
  const double xt = p[4];
  const double w = p[5];
  const double z = x - x0;
  if (z <= 0.0 || theta <= 0.0 || w <= 0.0) {
    return 0.0;
  }
  const double safe_z = std::max(z, 1e-9);
  return A * ff_sigmoid(x, xt, w) * TMath::Power(safe_z, a) * TMath::Exp(-safe_z / theta);
}

inline double ff_model_shift_sigpowexp_tail(double* xx, double* p) {
  const double x = xx[0];
  if (x <= 0.0) {
    return 0.0;
  }
  const double A = p[0];
  const double a = p[1];
  const double theta = p[2];
  const double x0 = p[3];
  const double xt = p[4];
  const double w = p[5];
  const double c1 = p[6];
  const double c2 = p[7];
  const double xmid = p[8];
  const double xscale = p[9];
  const double z = x - x0;
  if (z <= 0.0 || theta <= 0.0 || w <= 0.0 || xscale <= 0.0) {
    return 0.0;
  }
  const double safe_z = std::max(z, 1e-9);
  const double u = (x - xmid) / xscale;
  return A * ff_sigmoid(x, xt, w) * TMath::Power(safe_z, a) * TMath::Exp(-safe_z / theta)
         * ff_clip_exp(c1 * u + c2 * u * u);
}

inline double ff_model_gengamma_shift(double* xx, double* p) {
  const double x = xx[0];
  const double A = p[0];
  const double a = p[1];
  const double lambda = p[2];
  const double power = p[3];
  const double x0 = p[4];
  const double z = x - x0;
  if (z <= 0.0 || lambda <= 0.0 || power <= 0.0) {
    return 0.0;
  }
  const double safe_z = std::max(z, 1e-9);
  return A * TMath::Power(safe_z, a) * TMath::Exp(-TMath::Power(safe_z / lambda, power));
}

inline double ff_model_gengamma_thresh(double* xx, double* p) {
  const double x = xx[0];
  const double A = p[0];
  const double a = p[1];
  const double lambda = p[2];
  const double power = p[3];
  const double x0 = p[4];
  const double xt = p[5];
  const double w = p[6];
  const double z = x - x0;
  if (z <= 0.0 || lambda <= 0.0 || power <= 0.0 || w <= 0.0) {
    return 0.0;
  }
  const double safe_z = std::max(z, 1e-9);
  return A * ff_sigmoid(x, xt, w) * TMath::Power(safe_z, a)
         * TMath::Exp(-TMath::Power(safe_z / lambda, power));
}

inline double ff_model_logpoly_thresh(double* xx, double* p) {
  const double x = xx[0];
  if (x <= 0.0) {
    return 0.0;
  }
  const double A = p[0];
  const double c1 = p[1];
  const double c2 = p[2];
  const double xt = p[3];
  const double w = p[4];
  if (w <= 0.0) {
    return 0.0;
  }
  const double lx = std::log(x);
  return A * ff_sigmoid(x, xt, w) * ff_clip_exp(c1 * lx + c2 * lx * lx);
}

inline double ff_model_bern5(double* xx, double* p) {
  const double x = xx[0];
  const double A = p[0];
  const double xmin = p[7];
  const double xmax = p[8];
  if (x < xmin || x > xmax || xmax <= xmin) {
    return 0.0;
  }

  double q[6];
  for (int i = 0; i < 6; ++i) {
    q[i] = p[1 + i];
  }
  double maxq = q[0];
  for (int i = 1; i < 6; ++i) {
    maxq = std::max(maxq, q[i]);
  }
  double sumexp = 0.0;
  for (int i = 0; i < 6; ++i) {
    sumexp += TMath::Exp(q[i] - maxq);
  }
  if (sumexp <= 0.0) {
    return 0.0;
  }

  double wgt[6];
  for (int i = 0; i < 6; ++i) {
    wgt[i] = TMath::Exp(q[i] - maxq) / sumexp;
  }

  const double z = (x - xmin) / (xmax - xmin);
  const double one_minus = 1.0 - z;
  const double b0 = TMath::Power(one_minus, 5);
  const double b1 = 5.0 * z * TMath::Power(one_minus, 4);
  const double b2 = 10.0 * TMath::Power(z, 2) * TMath::Power(one_minus, 3);
  const double b3 = 10.0 * TMath::Power(z, 3) * TMath::Power(one_minus, 2);
  const double b4 = 5.0 * TMath::Power(z, 4) * one_minus;
  const double b5 = TMath::Power(z, 5);
  return A * (wgt[0] * b0 + wgt[1] * b1 + wgt[2] * b2 + wgt[3] * b3 + wgt[4] * b4 + wgt[5] * b5);
}

inline TF1* ff_make_sigpowexp(TH1* h, double fit_min, double fit_max, double, double rmsw) {
  TF1* f = new TF1("fSigPow", ff_model_sigpowexp, fit_min, fit_max, 5);
  f->SetParNames("A", "a", "theta", "xt", "w");
  f->SetNpx(8000);
  f->SetParameters(h->GetMaximum(), 6.0, std::max(1e-3, rmsw / 8.0), std::max(0.0, fit_min - 0.002), 0.003);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 2.0, 12.0);
  f->SetParLimits(2, 0.004, 0.20);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.010, fit_max - 1e-3));
  f->SetParLimits(4, 0.0015, 0.010);
  return f;
}

inline TF1* ff_make_sigpowexp_expquad(TH1* h, double fit_min, double fit_max, double, double rmsw) {
  TF1* f = new TF1("fSigPowExpQ", ff_model_sigpowexp_expquad, fit_min, fit_max, 7);
  f->SetParNames("A", "a", "theta", "xt", "w", "c1", "c2");
  f->SetNpx(8000);
  f->SetParameters(h->GetMaximum(), 6.0, std::max(1e-3, rmsw / 8.0), std::max(0.0, fit_min - 0.002), 0.003, 0.0, 0.0);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 0.0, 40.0);
  f->SetParLimits(2, 1e-4, 0.4);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.05, fit_max - 1e-3));
  f->SetParLimits(4, 1e-3, 0.02);
  f->SetParLimits(5, -50.0, 50.0);
  f->SetParLimits(6, -2000.0, 2000.0);
  return f;
}

inline TF1* ff_make_sigpowexp_endpoint(TH1* h, double fit_min, double fit_max, double, double rmsw) {
  TF1* f = new TF1("fEndpoint", ff_model_sigpowexp_endpoint, fit_min, fit_max, 7);
  f->SetParNames("A", "a", "theta", "xt", "w", "xmax", "b");
  f->SetNpx(8000);
  f->SetParameters(h->GetMaximum(), 5.0, std::max(1e-3, rmsw / 6.0), std::max(0.0, fit_min - 0.002), 0.002, fit_max * 1.03, 1.5);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 0.0, 40.0);
  f->SetParLimits(2, 1e-4, 0.4);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.05, fit_max - 1e-3));
  f->SetParLimits(4, 1e-3, 0.02);
  f->SetParLimits(5, fit_max * 1.001, fit_max * 2.5);
  f->SetParLimits(6, 0.0, 20.0);
  return f;
}

inline TF1* ff_make_shift_sigpowexp(TH1* h, double fit_min, double fit_max, double, double rmsw) {
  TF1* f = new TF1("fShiftSigPow", ff_model_shift_sigpowexp, fit_min, fit_max, 6);
  f->SetParNames("A", "a", "theta", "x0", "xt", "w");
  f->SetNpx(8000);
  f->SetParameters(
      h->GetMaximum(),
      6.0,
      std::max(0.004, std::max(1e-3, rmsw / 10.0)),
      std::max(0.0, fit_min - 0.008),
      std::max(0.0, fit_min - 0.004),
      0.003);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 2.0, 12.0);
  f->SetParLimits(2, 0.004, 0.20);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.020), std::min(fit_min + 0.005, fit_max - 1e-3));
  f->SetParLimits(4, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.010, fit_max - 1e-3));
  f->SetParLimits(5, 0.0015, 0.010);
  return f;
}

inline TF1* ff_make_shift_sigpowexp_tail(TH1* h, double fit_min, double fit_max, double, double rmsw) {
  TF1* f = new TF1("fShiftSigPowTail", ff_model_shift_sigpowexp_tail, fit_min, fit_max, 10);
  f->SetParNames("A", "a", "theta", "x0", "xt", "w", "c1", "c2", "xmid", "xscale");
  f->SetNpx(8000);
  const double xmid = 0.5 * (fit_min + fit_max);
  const double xscale = std::max(1e-3, 0.5 * (fit_max - fit_min));
  f->SetParameters(
      h->GetMaximum(),
      6.0,
      std::max(0.004, std::max(1e-3, rmsw / 10.0)),
      std::max(0.0, fit_min - 0.008),
      std::max(0.0, fit_min - 0.004),
      0.003,
      1.0,
      -0.8,
      xmid,
      xscale);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 2.0, 12.0);
  f->SetParLimits(2, 0.004, 0.20);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.020), std::min(fit_min + 0.005, fit_max - 1e-3));
  f->SetParLimits(4, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.010, fit_max - 1e-3));
  f->SetParLimits(5, 0.0015, 0.010);
  f->SetParLimits(6, -3.0, 3.0);
  f->SetParLimits(7, -3.0, 1.0);
  f->FixParameter(8, xmid);
  f->FixParameter(9, xscale);
  return f;
}

inline TF1* ff_make_gengamma_shift(TH1* h, double fit_min, double fit_max, double meanw, double rmsw) {
  TF1* f = new TF1("fGenGammaShift", ff_model_gengamma_shift, fit_min, fit_max, 5);
  f->SetParNames("A", "a", "lambda", "power", "x0");
  f->SetNpx(8000);
  f->SetParameters(h->GetMaximum(), 1.5, std::max(1e-3, rmsw / 3.0), 1.2, std::max(0.0, fit_min - 0.01));
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 0.5, 20.0);
  f->SetParLimits(2, 1e-4, std::max(0.02, 2.0 * (fit_max - fit_min)));
  f->SetParLimits(3, 0.2, 10.0);
  f->SetParLimits(4, std::max(0.0, fit_min - 0.03), std::min(meanw, fit_min + 0.02));
  return f;
}

inline TF1* ff_make_gengamma_thresh(TH1* h, double fit_min, double fit_max, double meanw, double rmsw) {
  TF1* f = new TF1("fGenGammaThresh", ff_model_gengamma_thresh, fit_min, fit_max, 7);
  f->SetParNames("A", "a", "lambda", "power", "x0", "xt", "w");
  f->SetNpx(8000);
  f->SetParameters(h->GetMaximum(), 1.2, std::max(1e-3, rmsw / 3.0), 1.2, std::max(0.0, fit_min - 0.01), std::max(0.0, fit_min - 0.002), 0.003);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, 0.5, 20.0);
  f->SetParLimits(2, 1e-4, std::max(0.02, 2.0 * (fit_max - fit_min)));
  f->SetParLimits(3, 0.2, 10.0);
  f->SetParLimits(4, std::max(0.0, fit_min - 0.03), std::min(meanw, fit_min + 0.02));
  f->SetParLimits(5, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.05, fit_max - 1e-3));
  f->SetParLimits(6, 1e-3, 0.02);
  return f;
}

inline TF1* ff_make_logpoly_thresh(TH1* h, double fit_min, double fit_max, double, double) {
  TF1* f = new TF1("fLogPolyThresh", ff_model_logpoly_thresh, fit_min, fit_max, 5);
  f->SetParNames("A", "c1", "c2", "xt", "w");
  f->SetNpx(8000);
  f->SetParameters(h->GetMaximum(), -2.0, -0.3, std::max(0.0, fit_min - 0.002), 0.003);
  f->SetParLimits(0, 1e-9, 1e18);
  f->SetParLimits(1, -30.0, 30.0);
  f->SetParLimits(2, -10.0, 10.0);
  f->SetParLimits(3, std::max(0.0, fit_min - 0.010), std::min(fit_min + 0.05, fit_max - 1e-3));
  f->SetParLimits(4, 1e-3, 0.02);
  return f;
}

inline TF1* ff_make_bern5(TH1* h, double fit_min, double fit_max, double, double) {
  TF1* f = new TF1("fBern5", ff_model_bern5, fit_min, fit_max, 9);
  f->SetParNames("A", "q1", "q2", "q3", "q4", "q5", "q6", "xminF", "xmaxF");
  f->SetNpx(8000);
  const double aseed = std::max(1.0, h->GetMaximum());
  f->SetParameters(aseed, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, fit_min, fit_max);
  f->FixParameter(7, fit_min);
  f->FixParameter(8, fit_max);
  f->SetParLimits(0, 1e-9, 1e18);
  return f;
}

inline std::vector<double> ff_fit_min_scan_values(const FuncFormJobConfig& job) {
  if (!job.fit_min_scan.empty()) {
    return job.fit_min_scan;
  }
  return std::vector<double>{job.fit_min};
}

inline int ff_seed_trial_count(const std::string& tag) {
  if (tag == "fBern5") {
    return 1;
  }
  if (tag == "fLogPolyThresh") {
    return 2;
  }
  if (tag == "fShiftSigPowTail") {
    return 4;
  }
  if (tag == "fSigPowExpQ") {
    return 1;
  }
  return 4;
}

inline std::string ff_trial_label(int trial_index) {
  switch (trial_index) {
    case 0: return "default";
    case 1: return "early-turnon";
    case 2: return "broad-tail";
    case 3: return "late-turnon";
    default: return "trial";
  }
}

inline void ff_apply_trial_seed(
    TF1* f,
    const std::string& tag,
    int trial_index,
    double fit_min,
    double fit_max,
    double meanw,
    double rmsw) {
  if (f == nullptr || trial_index <= 0) {
    return;
  }
  const double span = std::max(1e-3, fit_max - fit_min);
  const double theta_short = std::max(0.003, std::max(1e-3, rmsw / 10.0));
  const double theta_mid = std::max(0.006, std::max(1e-3, rmsw / 7.0));
  const double theta_long = std::max(theta_mid, std::min(0.35, 0.45 * span));
  const double x0_early = std::max(0.0, fit_min - std::min(0.015, 0.25 * span));
  const double x0_mid = std::max(0.0, fit_min - std::min(0.008, 0.15 * span));
  const double xt_early = std::max(0.0, fit_min - std::min(0.006, 0.10 * span));
  const double xt_mid = std::max(0.0, fit_min - std::min(0.002, 0.03 * span));
  const double xt_late = std::min(fit_max - 1e-3, fit_min + std::min(0.008, 0.10 * span));
  const double lambda_short = std::max(0.003, std::max(1e-3, rmsw / 3.5));
  const double lambda_mid = std::max(lambda_short, std::min(0.25, 0.30 * span));
  const double lambda_long = std::max(lambda_mid, std::min(0.30, 0.45 * span));
  if (trial_index == 1) {
    ff_set_param_named(f, "a", 4.0);
    ff_set_param_named(f, "theta", theta_mid);
    ff_set_param_named(f, "x0", x0_early);
    ff_set_param_named(f, "xt", xt_early);
    ff_set_param_named(f, "w", 0.0025);
    ff_set_param_named(f, "lambda", lambda_short);
    ff_set_param_named(f, "power", 1.0);
    ff_set_param_named(f, "c1", 1.5);
    ff_set_param_named(f, "c2", -1.0);
    return;
  }
  if (trial_index == 2) {
    ff_set_param_named(f, "a", 5.0);
    ff_set_param_named(f, "theta", theta_long);
    ff_set_param_named(f, "x0", x0_mid);
    ff_set_param_named(f, "xt", xt_mid);
    ff_set_param_named(f, "w", 0.0045);
    ff_set_param_named(f, "lambda", lambda_mid);
    ff_set_param_named(f, "power", 1.5);
    ff_set_param_named(f, "c1", 1.5);
    ff_set_param_named(f, "c2", -0.5);
    return;
  }
  if (trial_index == 3) {
    ff_set_param_named(f, "a", 2.5);
    ff_set_param_named(f, "theta", theta_mid);
    ff_set_param_named(f, "x0", x0_early);
    ff_set_param_named(f, "xt", xt_late);
    ff_set_param_named(f, "w", 0.0060);
    ff_set_param_named(f, "lambda", lambda_long);
    ff_set_param_named(f, "power", 2.0);
    ff_set_param_named(f, "c1", 1.2);
    ff_set_param_named(f, "c2", -2.2);
    return;
  }
  if (tag == "fLogPolyThresh") {
    ff_set_param_named(f, "c1", ff_seed_value_named(f, "c1", -2.0));
    ff_set_param_named(f, "c2", ff_seed_value_named(f, "c2", -0.3));
  }
  (void)meanw;
}

inline FuncFormFitSummary ff_fit_candidate(TH1* h, const FuncFormCandidateDef& def,
                                           const FuncFormJobConfig& job) {
  FuncFormFitSummary best;
  best.tag = def.tag;
  best.label = def.label;
  double best_metric = std::numeric_limits<double>::infinity();

  const std::vector<double> fit_mins = ff_fit_min_scan_values(job);
  for (double fit_min : fit_mins) {
    const double fit_max = job.fit_max;
    double meanw = 0.0;
    double varw = 0.0;
    long long nwin = 0;
    ff_window_moments(h, fit_min, fit_max, meanw, varw, nwin);
    const double rmsw = std::sqrt(std::max(0.0, varw));
    const int trial_count = ff_seed_trial_count(def.tag);
    for (int trial = 0; trial < trial_count; ++trial) {
      std::cout << "[funcform][" << def.tag << "] try fit_min=" << fit_min
                << " trial=" << ff_trial_label(trial) << std::endl;
      TF1* func = def.factory(h, fit_min, fit_max, meanw, rmsw);
      if (func == nullptr) {
        continue;
      }
      ff_apply_trial_seed(func, def.tag, trial, fit_min, fit_max, meanw, rmsw);
      const char* fit_opt = "RQSNLI";
      TFitResultPtr r = h->Fit(func, fit_opt);

      FuncFormFitSummary trial_fit;
      trial_fit.tag = def.tag;
      trial_fit.label = def.label;
      trial_fit.func = func;
      trial_fit.ok = (static_cast<int>(r) == 0) && r->IsValid();
      trial_fit.ndf_root = (r && r->Ndf() > 0) ? r->Ndf() : 0;
      trial_fit.chi2ndf_root = (trial_fit.ndf_root > 0) ? (r->Chi2() / trial_fit.ndf_root) : -1.0;
      trial_fit.eval = ff_eval_binavg_chi2(h, func, fit_min, fit_max);
      trial_fit.fit_min = fit_min;
      trial_fit.fit_max = fit_max;
      trial_fit.trial_index = trial;
      trial_fit.trial_label = ff_trial_label(trial);

      const double metric = trial_fit.eval.pearson_chi2ndf;
      if (metric < best_metric) {
        if (best.func != nullptr) {
          delete best.func;
          best.func = nullptr;
        }
        best = trial_fit;
        best_metric = metric;
      } else {
        delete trial_fit.func;
        trial_fit.func = nullptr;
      }
    }
  }

  std::cout << "[funcform][" << def.tag << "] "
            << (best.ok ? "fit OK" : "fit PROBLEM")
            << "  fit_min=" << best.fit_min
            << "  trial=" << best.trial_label
            << "  ROOT chi2/ndf=" << best.chi2ndf_root
            << "  Pearson(eval)=" << best.eval.pearson_chi2ndf
            << "  Neyman(eval)=" << best.eval.neyman_chi2ndf << "\n";
  return best;
}

inline double ff_metric(const FuncFormFitSummary& fit) {
  if (!std::isfinite(fit.eval.pearson_chi2ndf) || fit.eval.pearson_chi2ndf < 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return fit.eval.pearson_chi2ndf;
}

inline bool ff_is_usable(const FuncFormFitSummary& fit) {
  if (fit.func == nullptr) {
    return false;
  }
  if (!std::isfinite(ff_metric(fit))) {
    return false;
  }
  for (int i = 0; i < fit.func->GetNpar(); ++i) {
    if (!std::isfinite(fit.func->GetParameter(i))) {
      return false;
    }
  }
  return true;
}

inline int ff_find_best_index(const std::vector<FuncFormFitSummary>& fits,
                              const std::vector<std::string>& tags) {
  int best = -1;
  double best_metric = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < fits.size(); ++i) {
    if (!ff_is_usable(fits[i])) {
      continue;
    }
    if (!tags.empty() && std::find(tags.begin(), tags.end(), fits[i].tag) == tags.end()) {
      continue;
    }
    const double metric = ff_metric(fits[i]);
    if (metric < best_metric) {
      best_metric = metric;
      best = static_cast<int>(i);
    }
  }
  return best;
}

inline int ff_choose_primary_index(const std::vector<FuncFormFitSummary>& fits,
                                   const FuncFormJobConfig& job,
                                   const std::vector<FuncFormCandidateDef>& defs) {
  for (const auto& def : defs) {
    if (!def.preferred_primary) {
      continue;
    }
    for (std::size_t i = 0; i < fits.size(); ++i) {
      if (fits[i].tag == def.tag && ff_is_usable(fits[i])
          && ff_metric(fits[i]) <= job.primary_target_chi2ndf) {
        return static_cast<int>(i);
      }
    }
  }
  std::vector<std::string> preferred_tags;
  for (const auto& def : defs) {
    if (def.preferred_primary) {
      preferred_tags.push_back(def.tag);
    }
  }

  const int best_preferred = ff_find_best_index(fits, preferred_tags);
  const int bern_idx = ff_find_best_index(fits, std::vector<std::string>{job.bernstein_tag});
  if (job.allow_bernstein_primary_fallback && bern_idx >= 0) {
    if (best_preferred < 0) {
      return bern_idx;
    }
    if (ff_metric(fits[best_preferred]) > job.primary_target_chi2ndf
        && ff_metric(fits[bern_idx]) < ff_metric(fits[best_preferred])) {
      return bern_idx;
    }
  }
  if (best_preferred >= 0) {
    return best_preferred;
  }
  if (bern_idx >= 0) {
    return bern_idx;
  }
  return ff_find_best_index(fits, std::vector<std::string>{});
}

inline void ff_save_png_pdf(TCanvas* c, const std::string& stem) {
  const std::size_t slash = stem.find_last_of("/\\");
  if (slash != std::string::npos) {
    gSystem->mkdir(stem.substr(0, slash).c_str(), true);
  }
  c->SaveAs((stem + ".png").c_str());
  c->SaveAs((stem + ".pdf").c_str());
}

inline void ff_write_fit_metadata(TFile* fout,
                                  const std::vector<FuncFormFitSummary>& fits,
                                  int primary_idx,
                                  const FuncFormJobConfig& job) {
  TDirectory* meta_dir = fout->mkdir("fit_metadata");
  TDirectory::TContext ctx(meta_dir);

  std::ostringstream all_json;
  all_json << "{\n"
           << "  \"dataset\": \"" << ff_json_escape(job.dataset_key) << "\",\n"
           << "  \"fit_min_GeV\": " << std::setprecision(8) << job.fit_min << ",\n"
           << "  \"fit_max_GeV\": " << std::setprecision(8) << job.fit_max << ",\n"
           << "  \"primary_function\": \"" << ff_json_escape(primary_idx >= 0 ? fits[primary_idx].tag : "") << "\",\n"
           << "  \"fits\": [\n";

  for (std::size_t i = 0; i < fits.size(); ++i) {
    const auto& fit = fits[i];
    std::ostringstream one;
    one << "{\n"
        << "  \"tag\": \"" << ff_json_escape(fit.tag) << "\",\n"
        << "  \"label\": \"" << ff_json_escape(fit.label) << "\",\n"
        << "  \"is_primary\": " << (static_cast<int>(i) == primary_idx ? "true" : "false") << ",\n"
        << "  \"fit_ok\": " << (fit.ok ? "true" : "false") << ",\n"
        << "  \"fit_min_GeV\": " << fit.fit_min << ",\n"
        << "  \"fit_max_GeV\": " << fit.fit_max << ",\n"
        << "  \"trial_index\": " << fit.trial_index << ",\n"
        << "  \"trial_label\": \"" << ff_json_escape(fit.trial_label) << "\",\n"
        << "  \"root_chi2ndf\": " << fit.chi2ndf_root << ",\n"
        << "  \"pearson_chi2ndf\": " << fit.eval.pearson_chi2ndf << ",\n"
        << "  \"neyman_chi2ndf\": " << fit.eval.neyman_chi2ndf << ",\n"
        << "  \"nbin_used\": " << fit.eval.nbin_used << ",\n"
        << "  \"ndf_sel\": " << fit.eval.ndf_sel << "\n"
        << "}";
    TNamed named((fit.tag + "_metadata").c_str(), one.str().c_str());
    named.Write();

    all_json << "    " << one.str();
    if (i + 1 < fits.size()) {
      all_json << ",";
    }
    all_json << "\n";
  }
  all_json << "  ]\n}\n";

  TNamed summary("fit_summary_json", all_json.str().c_str());
  summary.Write();
  TNamed primary("primary_function_tag", primary_idx >= 0 ? fits[primary_idx].tag.c_str() : "");
  primary.Write();
}

inline void ff_make_toys_for_fit(TFile* fout, const FuncFormFitSummary& fit,
                                 TH1* h_in, double fit_min, double fit_max, int n_toys) {
  if (!ff_is_usable(fit)) {
    return;
  }

  const int nb = h_in->GetNbinsX();
  const double hxmin = h_in->GetXaxis()->GetXmin();
  const double hxmax = h_in->GetXaxis()->GetXmax();
  const long long nfill = static_cast<long long>(
      std::llround(h_in->Integral(h_in->FindBin(fit.fit_min + 1e-9), h_in->FindBin(fit.fit_max - 1e-9))));

  fit.func->SetRange(fit.fit_min, fit.fit_max);
  if (gROOT->GetListOfFunctions()->FindObject(fit.func->GetName()) == nullptr) {
    gROOT->GetListOfFunctions()->Add(fit.func);
  }
  TDirectory* d = fout->mkdir(fit.tag.c_str());
  TDirectory::TContext ctx(d);
  for (int itoy = 0; itoy < n_toys; ++itoy) {
    const std::string hname = fit.tag + "_toy_" + std::to_string(itoy);
    TH1D htoy(hname.c_str(), hname.c_str(), nb, hxmin, hxmax);
    htoy.Sumw2(false);
    htoy.FillRandom(fit.func->GetName(), std::max<long long>(1, nfill));
    htoy.Write();
  }
}

inline std::string ff_short_legend_label(const std::string& label) {
  if (label == "shifted sigmoid*power*exp + tail") {
    return "shifted sigm*pow*exp + tail";
  }
  if (label == "shifted sigmoid*power*exp") {
    return "shifted sigm*pow*exp";
  }
  if (label == "sigmoid*power*exp + raw expquad") {
    return "sigm*pow*exp + expquad";
  }
  if (label == "sigmoid*x^{a}*exp(-x/theta)") {
    return "sigmoid*x^{a}*exp(-x/#theta)";
  }
  if (label == "thresholded gen-gamma") {
    return "thresholded gen-gamma";
  }
  if (label == "thresholded log-polynomial") {
    return "thresholded log-poly";
  }
  if (label == "positive Bernstein fallback") {
    return "Bernstein fallback";
  }
  if (label == "endpoint-aware sigmoid*power*exp") {
    return "endpoint-aware sigm*pow*exp";
  }
  return label;
}

inline void ff_make_overlay_plot(TH1* h_in,
                                 const std::vector<FuncFormFitSummary>& fits,
                                 int primary_idx,
                                 const FuncFormJobConfig& job) {
  gStyle->SetOptStat(0);
  TCanvas c("c_funcform", "c_funcform", 1550, 820);
  c.SetLeftMargin(0.10);
  c.SetBottomMargin(0.11);
  c.SetTopMargin(0.08);
  c.SetRightMargin(0.36);
  c.SetLogy();

  TH1* hdraw = dynamic_cast<TH1*>(h_in->Clone("hdraw_funcform"));
  hdraw->SetDirectory(nullptr);
  double plot_min = job.fit_min;
  const std::vector<double> fit_mins = ff_fit_min_scan_values(job);
  if (!fit_mins.empty()) {
    plot_min = *std::min_element(fit_mins.begin(), fit_mins.end());
  }
  hdraw->GetXaxis()->SetRangeUser(plot_min, job.fit_max);
  hdraw->SetMarkerStyle(20);
  hdraw->SetMarkerSize(0.8);
  hdraw->SetLineColor(kBlack);
  hdraw->SetTitle((job.dataset_label + ";m_{e^{+}e^{-}} [GeV];Events / bin").c_str());
  hdraw->GetXaxis()->SetTitleOffset(1.0);
  hdraw->GetYaxis()->SetTitleOffset(1.2);

  double ymax = 0.0;
  double ymin = std::numeric_limits<double>::infinity();
  const int b1 = hdraw->GetXaxis()->FindBin(plot_min + 1e-12);
  const int b2 = hdraw->GetXaxis()->FindBin(job.fit_max - 1e-12);
  for (int i = b1; i <= b2; ++i) {
    const double y = hdraw->GetBinContent(i);
    if (y > 0.0) {
      ymax = std::max(ymax, y);
      ymin = std::min(ymin, y);
    }
  }
  if (!std::isfinite(ymin)) {
    ymin = 1.0;
  }
  hdraw->SetMinimum(std::max(0.5, 0.5 * ymin));
  hdraw->SetMaximum(std::max(10.0, 2.0 * ymax));
  hdraw->Draw("E1");

  const int colors[] = {kBlue + 1, kRed + 1, kGreen + 2, kOrange + 7, kMagenta + 2};
  TLegend leg(0.66, 0.12, 0.985, 0.92);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.SetMargin(0.22);
  leg.SetTextSize(0.022);
  leg.SetEntrySeparation(0.18);
  leg.AddEntry(hdraw, "Input histogram", "lep");

  int color_idx = 0;
  for (std::size_t i = 0; i < fits.size(); ++i) {
    if (fits[i].func == nullptr) {
      continue;
    }
    fits[i].func->SetRange(fits[i].fit_min, fits[i].fit_max);
    fits[i].func->SetLineColor(colors[color_idx % (sizeof(colors) / sizeof(colors[0]))]);
    fits[i].func->SetLineWidth(static_cast<int>(i) == primary_idx ? 4 : 2);
    fits[i].func->Draw("SAME");
    std::ostringstream entry;
    entry << "#splitline{" << ff_short_legend_label(fits[i].label) << "}{#chi^{2}/ndof = "
          << std::fixed << std::setprecision(2) << ff_metric(fits[i]) << "}";
    leg.AddEntry(fits[i].func, entry.str().c_str(), "l");
    color_idx++;
  }
  leg.Draw();
  c.RedrawAxis();
  ff_save_png_pdf(&c, job.note_plot_stem);
  delete hdraw;
}

inline void ff_run_job(const FuncFormJobConfig& job,
                       const std::vector<FuncFormCandidateDef>& defs) {
  gROOT->SetBatch(true);
  TH1::AddDirectory(kFALSE);

  TFile* fin = TFile::Open(job.input_file.c_str(), "READ");
  if (fin == nullptr || fin->IsZombie()) {
    std::cerr << "Cannot open " << job.input_file << "\n";
    return;
  }
  TH1* h0 = dynamic_cast<TH1*>(fin->Get(job.hist_name.c_str()));
  if (h0 == nullptr) {
    std::cerr << "Missing histogram " << job.hist_name << " in " << job.input_file << "\n";
    fin->Close();
    delete fin;
    return;
  }
  h0->SetDirectory(nullptr);
  fin->Close();
  delete fin;

  std::vector<FuncFormFitSummary> fits;
  for (const auto& def : defs) {
    if (!def.enabled) {
      continue;
    }
    fits.push_back(ff_fit_candidate(h0, def, job));
    if (fits.back().func != nullptr) {
      fits.back().func->SetName(def.tag.c_str());
      fits.back().func->SetTitle(def.label.c_str());
    }
  }

  const int primary_idx = ff_choose_primary_index(fits, job, defs);
  if (primary_idx >= 0) {
    std::cout << "[funcform][" << job.dataset_key << "] primary function = "
              << fits[primary_idx].tag << "  metric=" << ff_metric(fits[primary_idx]) << "\n";
  } else {
    std::cout << "[funcform][" << job.dataset_key << "] no valid primary function found\n";
  }

  const std::size_t slash = job.output_root.find_last_of("/\\");
  if (slash != std::string::npos) {
    gSystem->mkdir(job.output_root.substr(0, slash).c_str(), true);
  }
  TFile* fout = TFile::Open(job.output_root.c_str(), "RECREATE");
  if (fout == nullptr || fout->IsZombie()) {
    std::cerr << "Cannot create " << job.output_root << "\n";
    delete h0;
    return;
  }

  TH1D* h_in = dynamic_cast<TH1D*>(h0->Clone("input_hist"));
  if (h_in == nullptr) {
    h_in = new TH1D("input_hist", h0->GetTitle(), h0->GetNbinsX(), h0->GetXaxis()->GetXmin(), h0->GetXaxis()->GetXmax());
    for (int i = 1; i <= h0->GetNbinsX(); ++i) {
      h_in->SetBinContent(i, h0->GetBinContent(i));
      h_in->SetBinError(i, std::sqrt(std::max(0.0, h0->GetBinContent(i))));
    }
  }
  h_in->SetDirectory(fout);
  h_in->Write();

  TDirectory* fit_dir = fout->mkdir("fit_functions");
  {
    TDirectory::TContext ctx(fit_dir);
    for (const auto& fit : fits) {
      if (fit.func != nullptr) {
        fit.func->Write((fit.tag + "_fit").c_str());
      }
    }
  }

  ff_write_fit_metadata(fout, fits, primary_idx, job);
  for (std::size_t i = 0; i < fits.size(); ++i) {
    if (!ff_is_usable(fits[i])) {
      continue;
    }
    ff_make_toys_for_fit(fout, fits[i], h0, job.fit_min, job.fit_max, job.n_toys);
  }

  fout->Write();
  fout->Close();

  ff_make_overlay_plot(h0, fits, primary_idx, job);

  delete h0;
  for (auto& fit : fits) {
    delete fit.func;
    fit.func = nullptr;
  }
  delete fout;
}

#endif
