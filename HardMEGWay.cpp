#include "TFile.h"
#include "TH1D.h"
#include "TH2.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TString.h"
#include "TSystem.h"
#include "TAxis.h"

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>

//==============================================================
// Convenience helpers
// Formatting and file-name helpers do not change the fit result.
//==============================================================
static std::string formatP_6(double P)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << P;
    return oss.str();
}

static std::string formatP_showpos_6(double P)
{
    std::ostringstream oss;
    oss << std::showpos << std::fixed << std::setprecision(6) << P;
    return oss.str();
}

static std::string formatP_1_showpos(double P)
{
    std::ostringstream oss;
    oss << std::showpos << std::fixed << std::setprecision(1) << P;
    return oss.str();
}

static std::string formatP_tag_1(double P)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << P;
    std::string s = oss.str();
    for (char &c : s) {
        if (c == '.') c = 'p';
    }
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

//==============================================================
// Important part: input histograms
// Load the measured A_FB(x) histogram used in the fit.
//==============================================================
TH1D* loadSymmetryAfbVsX(const std::string& afbDir, double Pvalue)
{
    const std::string filename =
        afbDir + "/Afb_vs_x_P_" + formatP_tag_1(Pvalue) + ".root";

    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: cannot open file: " << filename << "\n";
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("Afb_vs_x"));
    if (!h) {
        const std::string altName = "Afb_vs_x_P_" + formatP_tag_1(Pvalue);
        h = dynamic_cast<TH1D*>(f->Get(altName.c_str()));
    }
    if (!h) {
        std::cerr << "ERROR: histogram 'Afb_vs_x' not found in: " << filename << "\n";
        f->Close();
        delete f;
        return nullptr;
    }

    TH1D* hClone = dynamic_cast<TH1D*>(h->Clone(
        Form("Afb_vs_x_clone_P_%s", formatP_1_showpos(Pvalue).c_str())));
    hClone->SetDirectory(nullptr);

    f->Close();
    delete f;
    return hClone;
}

//==============================================================
// Important part: reference template
// Load the precomputed <cos(theta)>(x) template used by the model fit.
//==============================================================
TH1D* loadAvgCosVsX(const std::string& avgCosDir, double Pvalue)
{
    const std::string filename =
        avgCosDir + "/AvgCostheta_vs_x_P_" + formatP_tag_1(Pvalue) + ".root";

    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: cannot open avg-cos file: " << filename << "\n";
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("AvgCosTheta_vs_x"));
    if (!h) {
        const std::string altName = "hAvgCosTheta_vs_x_P_" + formatP_tag_1(Pvalue);
        h = dynamic_cast<TH1D*>(f->Get(altName.c_str()));
    }
    if (!h) {
        std::cerr << "ERROR: histogram 'AvgCosTheta_vs_x' not found in: " << filename << "\n";
        f->Close();
        delete f;
        return nullptr;
    }

    TH1D* hClone = dynamic_cast<TH1D*>(h->Clone(
        Form("AvgCos_vs_x_clone_P_%s", formatP_1_showpos(Pvalue).c_str())));
    hClone->SetDirectory(nullptr);

    f->Close();
    delete f;
    return hClone;
}

//==============================================================
// Important part: fit model
// A_FB(x) = K * x * P * <cos(theta)>(x)
//==============================================================
static TH1D* gAvgCos = nullptr;

double AfbModel(double* xx, double* par)
{
    if (!gAvgCos) return 0.0;

    const double E = xx[0];
    const int bin = gAvgCos->FindBin(E);
    if (bin < 1 || bin > gAvgCos->GetNbinsX()) return 0.0;
    const double avgCos = gAvgCos->GetBinContent(bin);

    // par[0] = P and par[1] = K
    return par[1] * E * par[0] * avgCos;

}

static bool SubtractBaselineHistogram(TH1D* target,
                                      const TH1D* baseline,
                                      const char* label)
{
    if (!target || !baseline) {
        return false;
    }
    if (target->GetNbinsX() != baseline->GetNbinsX()) {
        std::cerr << "WARN: cannot subtract " << label
                  << " baseline because the bin counts differ.\n";
        return false;
    }

    target->Add(baseline, -1.0);
    return true;
}

void FitP_from_Avgcos()
{
    const std::string baseAfbDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3-layer afbvsx";
    const std::string baseAvgCosDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/Reference graph/Avg costheta";
    const std::string baseOutDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3 layer MEG";

    const std::string afbDir = baseAfbDir;
    const std::string avgCosDir = baseAvgCosDir;
    const std::string outDir = baseOutDir;
    std::vector<double> pTrueVals;
    std::vector<double> pRatioVals;
    std::vector<double> pRatioErrVals;
    bool kCalibrated = false;
    double Kglobal = 1.5;
    double KglobalErr = 0.0;

    gSystem->mkdir(outDir.c_str(), true);

    TH1D* hAfbZero = loadSymmetryAfbVsX(afbDir, 0.0);
    TH1D* hAvgCosZero = loadAvgCosVsX(avgCosDir, 0.0);
    if (!hAfbZero) {
        std::cerr << "WARN: P=0 Afb baseline not available; no Afb subtraction will be applied.\n";
    }
    if (!hAvgCosZero) {
        std::cerr << "WARN: P=0 avg cos(theta) baseline not available; no avg-cos subtraction will be applied.\n";
    }

    //==========================================================
    // Important part: baseline subtraction and P fit
    // This loop does the baseline subtraction, K calibration, and final P fit.
    //==========================================================
    for (int i = 0; i <= 20; ++i) {
        double Ptrue = 1.0 - 0.1 * i;
        if (std::abs(Ptrue) <= 1e-12) continue;
        TH1D* hAfb = loadSymmetryAfbVsX(afbDir, Ptrue);
        if (!hAfb) continue;

        const std::string tag = formatP_tag_1(Ptrue);
        gAvgCos = loadAvgCosVsX(avgCosDir, Ptrue);
        if (!gAvgCos) {
            std::cerr << "ERROR: avg cos(theta) histogram not available for P=" << Ptrue << "\n";
            delete hAfb;
            continue;
        }

        SubtractBaselineHistogram(hAfb, hAfbZero, "Afb");
        SubtractBaselineHistogram(gAvgCos, hAvgCosZero, "avg cos(theta)");
        TH1D* hAvgCos = dynamic_cast<TH1D*>(gAvgCos->Clone(
            Form("AvgCos_for_plot_P_%s", formatP_1_showpos(Ptrue).c_str())));
        hAvgCos->SetDirectory(nullptr);
        hAvgCos->SetTitle(Form("<cos#theta>(x)-<cos#theta>_{P=0}(x), P_{true}=%+.1f; x=E/E_{end}; <cos#theta>", Ptrue));
        hAvgCos->SetMarkerStyle(20);
        hAvgCos->SetMarkerSize(0.8);
        hAvgCos->SetLineWidth(2);

        //======================================================
        // Convenience and graph output
        // This plotting block is for inspection and presentation.
        // It does not change the extracted fit parameters.
        // Parts of this graph/output setup were organised with LLM help.
        //======================================================
        TCanvas cAvgCos(Form("cAvgCos_%s", tag.c_str()), "<cos#theta> vs x", 900, 650);
        hAvgCos->Draw("P");

        const std::string pShow = formatP_1_showpos(Ptrue);
        const std::string outAvgCosPdf = outDir + "/AvgCosVsX_P_" + pShow + ".pdf";
        cAvgCos.SaveAs(outAvgCosPdf.c_str());

        // Important part: convert the baseline-subtracted template into
        // response per unit polarization before fitting.
        gAvgCos->Scale(1.0 / Ptrue);
        
        const double xMin = hAfb->GetXaxis()->GetXmin();
        const double xMax = hAfb->GetXaxis()->GetXmax();
        const double xMinFit = 0.4;
        const double fitLo = std::max(xMin, xMinFit);

        // Important part: calibrate the shared K factor once with P fixed to +1.
        if (!kCalibrated) {
            TF1 fK("fK", AfbModel, fitLo, xMax, 2);
            fK.SetParameter(0, 1.0);
            fK.FixParameter(0, 1.0);
            fK.SetParameter(1, 1.5);
            const int fitStatusK = hAfb->Fit(&fK, "WQR");
            if (fitStatusK != 0) {
                std::cerr << "WARN: K calibration fit failed for P=" << Ptrue
                          << " status=" << fitStatusK << "\n";
            }
            Kglobal = fK.GetParameter(1);
            KglobalErr = fK.GetParError(1);
            kCalibrated = true;
            std::cout << "Global K calibrated once: " << Kglobal
                      << " +/- " << KglobalErr
                      << " (from Ptrue=" << Ptrue << ")\n";
        }

        // Important part: fit P with the calibrated K held fixed.
        TF1 fP("fP", AfbModel, fitLo, xMax, 2);
        fP.FixParameter(1, Kglobal);
        fP.SetParameter(0, 0.0);
        int fitStatus = hAfb->Fit(&fP, "QR");
        if (fitStatus != 0) {
            std::cerr << "WARN: fit failed for P=" << Ptrue << " status=" << fitStatus << "\n";
        }

        const double Pfit = fP.GetParameter(0);
        const double PfitErr = fP.GetParError(0);
        pTrueVals.push_back(Ptrue);
        pRatioVals.push_back(Pfit / Ptrue);
        pRatioErrVals.push_back(PfitErr / std::abs(Ptrue));

        hAfb->SetMarkerStyle(20);
        hAfb->SetMarkerSize(0.8);
        hAfb->SetLineWidth(1);
        fP.SetLineWidth(2);
        fP.SetLineColor(kRed + 1);

        TCanvas cAfb(Form("cAfb_%s", tag.c_str()), "Afb vs x with fit", 900, 650);
        hAfb->SetTitle(Form("A_{FB}(x)-A_{FB,P=0}(x) fit, P_{true}=%+.1f; x=E/E_{end}; A_{FB}", Ptrue));
        hAfb->Draw("E1");
        fP.Draw("SAME");
        TLegend leg(0.52, 0.73, 0.88, 0.88);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.AddEntry(hAfb, Form("A_{FB}(x), P_{true}=%+.1f", Ptrue), "lep");
        leg.AddEntry(&fP, Form("fit: P_{fit}=%+.3f #pm %.3f, K=%.3f", Pfit, PfitErr, Kglobal), "l");
        leg.Draw();

        const std::string outPdf = outDir + "/AfbVsX_FitP_P_" + pShow + ".pdf";
        const std::string outRoot = outDir + "/AfbVsX_FitP_P_" + pShow + ".root";
        cAfb.SaveAs(outPdf.c_str());

        TFile fOut(outRoot.c_str(), "RECREATE");
        hAvgCos->Write("AvgCos_vs_x");
        hAfb->Write("Afb_vs_x");
        fP.Write("fit_AfbModel");
        fOut.Close();

        delete hAvgCos;
        delete gAvgCos;
        gAvgCos = nullptr;
        delete hAfb;
    }

    delete hAfbZero;
    delete hAvgCosZero;

    if (!pTrueVals.empty()) {
        //======================================================
        // Convenience and graph output
        // Summary graph for the fit quality across P values.
        // This is presentation-only and does not affect the fit.
        // Parts of this graph/output setup were organised with LLM help.
        //======================================================
        TGraphErrors gRatio((int)pTrueVals.size());
        gRatio.SetName("g_pfit_over_ptrue_vs_ptrue");
        gRatio.SetTitle("P_{fit}/P_{true} vs P_{true}; P_{true}; P_{fit}/P_{true}");
        gRatio.SetMarkerStyle(20);
        for (int i = 0; i < (int)pTrueVals.size(); ++i) {
            gRatio.SetPoint(i, pTrueVals[i], pRatioVals[i]);
            gRatio.SetPointError(i, 0.0, pRatioErrVals[i]);
        }

        TCanvas cRatio("c_pfit_over_ptrue_vs_ptrue", "Pfit/Ptrue vs Ptrue", 900, 650);
        gRatio.Draw("APE");
        cRatio.SaveAs(Form("%s/PfitOverPtrueVsPtrue.pdf", outDir.c_str()));
    }
}


//==============================================================
// Convenience only
// Program entry point.
//==============================================================
int main()
{
    FitP_from_Avgcos();
    return 0;
}
