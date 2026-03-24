#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TLine.h"
#include "TRandom3.h"
#include "TSystem.h"
#include "TF1.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <functional>
#include <fstream>
#include <unordered_map>
#include <vector>

//==============================================================
// Convenience helpers
// Formatting, path handling, and small utilities do not change the toy result.
//==============================================================
static std::string FormatPTag(double P)
{
    std::ostringstream tag;
    tag << std::fixed << std::setprecision(1) << P;
    std::string s = tag.str();
    for (char &c : s) {
        if (c == '.') c = 'p';
    }
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

static std::vector<long long> BuildNList(long long nMin, long long nMax, int nSteps)
{
    std::vector<long long> nVals;
    if (nSteps < 2) return nVals;
    nVals.reserve(nSteps);
    for (int i = 0; i < nSteps; ++i) {
        const double nVal =
            (double)nMin + (double)i * (double)(nMax - nMin) / (double)(nSteps - 1);
        nVals.push_back((long long)llround(nVal));
    }
    return nVals;
}

static std::vector<double> BuildPolarizationList(bool scanAllP, double singleP)
{
    std::vector<double> pVals;
    if (scanAllP) {
        pVals.reserve(20);
        for (int i = -10; i <= 10; ++i) {
            if (i == 0) continue;
            pVals.push_back(0.1 * (double)i);
        }
    } else {
        pVals.push_back(singleP);
    }
    return pVals;
}

static std::string DefaultAvgCosRefTagForP(double P)
{
    return FormatPTag(P);
}

static bool ParsePTagToValue(const std::string& pTag, double& p)
{
    if (pTag.empty()) return false;
    std::string s = pTag;
    if (s[0] == 'm') s[0] = '-';
    for (char& c : s) {
        if (c == 'p') c = '.';
    }
    try {
        p = std::stod(s);
    } catch (...) {
        return false;
    }
    return std::isfinite(p);
}

//==============================================================
// Important part: fit model
// A_FB(x) = K * x * P * <cos(theta)>(x)
//==============================================================
static TH1D* gAvgCos = nullptr;
static TH1D* gAfbZeroBias = nullptr;

double AfbModel(double* xx, double* par)
{
    if (!gAvgCos) return 0.0;
    const double E = xx[0];
    const int bin = gAvgCos->FindBin(E);
    if (bin < 1 || bin > gAvgCos->GetNbinsX()) return 0.0;
    const double avgCos = gAvgCos->GetBinContent(bin);
    // par[0] = P, par[1] = K
    return par[1] * E * par[0] * avgCos;
}

static TH1D* BuildAfbFromFB(const TH1D* hF,
                            const TH1D* hB,
                            const std::string& name,
                            TRandom3& rng,
                            double scale)
{
    TH1D* hAfb = (TH1D*)hF->Clone(name.c_str());
    hAfb->Reset("ICES");
    hAfb->SetTitle("A_{FB}(x); x=E/E_{end}; A_{FB}");
    hAfb->SetDirectory(nullptr);

    const int nBins = hF->GetNbinsX();

    //==========================================================
    // Important part: Poisson toy generation
    // F ~ Poisson(scale * hF_bin)
    // B ~ Poisson(scale * hB_bin)
    // The total N fluctuates naturally rather than being fixed exactly.
    //==========================================================
    for (int ix = 1; ix <= nBins; ++ix) {
        const double muF = scale * hF->GetBinContent(ix);
        const double muB = scale * hB->GetBinContent(ix);

        const unsigned int F = (muF > 0.0) ? (unsigned int)rng.Poisson(muF) : 0U;
        const unsigned int B = (muB > 0.0) ? (unsigned int)rng.Poisson(muB) : 0U;

        const double N = (double)(F + B);
        double A = 0.0;
        double eA = 0.0;
        if (N > 0.0) {
            A = ((double)F - (double)B) / N;
            const double FB = (double)F * (double)B;
            if (FB > 0.0) {
                eA = 2.0 * std::sqrt(FB) / std::pow(N, 1.5);
            }
        }

        hAfb->SetBinContent(ix, A);
        hAfb->SetBinError(ix, eA);
    }

    hAfb->SetStats(0);
    return hAfb;
}

static bool FitPFromAfb(TH1D* hAfb, double kFixed, double& pFit)
{
    // Important part: extract P from the toy A_FB(x) histogram with K fixed.
    const double xMinRaw = hAfb->GetXaxis()->GetXmin();
    const double xMax = hAfb->GetXaxis()->GetXmax();
    const double xMinFit = 0.4;
    const double xMin = (xMinRaw > xMinFit) ? xMinRaw : xMinFit;

    TF1 fP("fP", AfbModel, xMin, xMax, 2);
    fP.FixParameter(1, kFixed);
    fP.SetParameter(0, 0.0);
    const int fitStatus = hAfb->Fit(&fP, "QRN");
    if (fitStatus != 0) return false;

    pFit = fP.GetParameter(0);
    if (!std::isfinite(pFit)) return false;
    return true;
}

static TH1D* LoadH1(TFile* f, const std::string& name)
{
    if (!f) return nullptr;
    TH1D* h = dynamic_cast<TH1D*>(f->Get(name.c_str()));
    if (!h) return nullptr;
    TH1D* clone = dynamic_cast<TH1D*>(h->Clone());
    if (clone) clone->SetDirectory(nullptr);
    return clone;
}

static TH1D* LoadAvgCosTemplate(const std::string& avgCosFile,
                                const std::string& histTag)
{
    TFile* f = TFile::Open(avgCosFile.c_str(), "READ");
    if (!f || f->IsZombie()) {
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("AvgCosTheta_vs_x"));
    if (!h) {
        h = dynamic_cast<TH1D*>(f->Get(("hAvgCosTheta_vs_x_P_" + histTag).c_str()));
    }

    TH1D* out = nullptr;
    if (h) {
        out = dynamic_cast<TH1D*>(h->Clone(("AvgCos_template_" + histTag).c_str()));
        if (out) out->SetDirectory(nullptr);
    }

    f->Close();
    delete f;
    return out;
}

static TH1D* LoadAfbTemplate(const std::string& afbFile)
{
    TFile* f = TFile::Open(afbFile.c_str(), "READ");
    if (!f || f->IsZombie()) {
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("Afb_vs_x"));
    TH1D* out = nullptr;
    if (h) {
        out = dynamic_cast<TH1D*>(h->Clone("Afb_zero_bias_template"));
        if (out) out->SetDirectory(nullptr);
    }

    f->Close();
    delete f;
    return out;
}

static void SubtractAfbZeroBias(TH1D* hAfb)
{
    if (!hAfb || !gAfbZeroBias) return;
    if (hAfb->GetNbinsX() != gAfbZeroBias->GetNbinsX()) return;
    hAfb->Add(gAfbZeroBias, -1.0);
}

static std::string ResolveInputFile(const std::vector<std::string>& dirs,
                                    const std::string& filename)
{
    for (const std::string& dir : dirs) {
        const std::string path = dir + "/" + filename;
        std::ifstream f(path);
        if (f.good()) {
            return path;
        }
    }

    if (!dirs.empty()) return dirs.front() + "/" + filename;
    return filename;
}

//==============================================================
// Convenience only
// Program entry point and run configuration.
//==============================================================
int main(int argc, char** argv)
{
    std::string baseAfbDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3-layer afbvsx";
    std::string baseAvgCosDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/cos refer";
    std::string afbZeroDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3-layer afbvsx";
    std::string outDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3 layer MEG/CPU toys";

    std::string refDir;
    bool refDirProvided = false;
    double Ptrue = 1.0;
    bool scanAllP = true;
    std::string avgCosRefTag = DefaultAvgCosRefTagForP(Ptrue);
    bool avgCosRefTagProvided = false;
    double kFixedMeg = 8.224;
    long long nMin = 1000000LL;
    long long nMax = 1000000000LL;
    int nSteps = 1000;
    int nToys = 500;
    unsigned long long seed = 123456789ULL;

    //==========================================================
    // Convenience only
    // Command-line parsing and usage text do not change the toy physics.
    //==========================================================
    auto PrintUsage = [&](const char* prog) {
        std::cout
            << "Usage: " << prog
            << " [--ref-dir <dir>] [--base-afb-dir <dir>] [--base-avgcos-dir <dir>] [--afb-zero-dir <dir>]\n"
            << "        [--avgcos-ref-tag <tag>] [--k-fixed <K>] [--out <dir>] [--p <Ptrue>] [--nmin <N>] [--nmax <N>] [--nsteps <steps>]\n"
            << "        [--ntoys <N>] [--seed <seed>] [--scan-p]\n"
            << "If --ref-dir is not set, uses:\n"
            << "  Afb:  <base-afb-dir>/Afb_vs_x_P_<tag>.root\n"
            << "  <cos>: <base-avgcos-dir>/AvgCostheta_vs_x_P_<avgcos-ref-tag>.root\n"
            << "  P=0 bias: <afb-zero-dir>/Afb_vs_x_P_0p0.root\n"
            << "  Defaults use raw Afb count templates and the no-offset avg-cos outputs.\n"
            << "Default mode scans P_true from -1.0 to 1.0 (step 0.1), excluding 0.0.\n"
            << "Use --p <Ptrue> to run a single polarization.\n";
    };

    struct OptHandler {
        bool needsValue;
        std::function<void(const char*)> fn;
    };

    std::unordered_map<std::string, OptHandler> opts;
    opts["--ref-dir"] = {true, [&](const char* v) {
        refDir = v;
        refDirProvided = true;
    }};
    opts["--base-afb-dir"] = {true, [&](const char* v) {
        baseAfbDir = v;
    }};
    opts["--base-avgcos-dir"] = {true, [&](const char* v) {
        baseAvgCosDir = v;
    }};
    opts["--afb-zero-dir"] = {true, [&](const char* v) {
        afbZeroDir = v;
    }};
    opts["--base-mc-dir"] = {true, [&](const char* v) {
        baseAvgCosDir = v;
    }};
    opts["--out"] = {true, [&](const char* v) {
        outDir = v;
    }};
    opts["--p"] = {true, [&](const char* v) {
        Ptrue = std::stod(v);
        scanAllP = false;
    }};
    opts["--scan-p"] = {false, [&](const char*) {
        scanAllP = true;
    }};
    opts["--avgcos-ref-tag"] = {true, [&](const char* v) {
        avgCosRefTag = v;
        avgCosRefTagProvided = true;
    }};
    opts["--k-fixed"] = {true, [&](const char* v) {
        kFixedMeg = std::stod(v);
    }};
    opts["--nmin"] = {true, [&](const char* v) {
        nMin = std::stoll(v);
    }};
    opts["--nmax"] = {true, [&](const char* v) {
        nMax = std::stoll(v);
    }};
    opts["--nsteps"] = {true, [&](const char* v) {
        nSteps = std::stoi(v);
    }};
    opts["--ntoys"] = {true, [&](const char* v) {
        nToys = std::stoi(v);
    }};
    opts["--seed"] = {true, [&](const char* v) {
        seed = std::stoull(v);
    }};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }

        auto it = opts.find(a);
        if (it == opts.end()) {
            std::cerr << "ERROR: unknown option: " << a << "\n";
            PrintUsage(argv[0]);
            return 1;
        }

        if (it->second.needsValue) {
            if (i + 1 >= argc) {
                std::cerr << "ERROR: missing value for " << a << "\n";
                PrintUsage(argv[0]);
                return 1;
            }
            const char* v = argv[++i];
            it->second.fn(v);
        } else {
            it->second.fn(nullptr);
        }
    }

    if (nToys <= 0) {
        std::cerr << "ERROR: --ntoys must be > 0.\n";
        return 1;
    }
    if (!scanAllP && std::abs(Ptrue) <= 1e-12) {
        std::cerr << "ERROR: --p 0 is not supported in this workflow.\n"
                  << "       Relative 1% target uses |Ptrue|, so Ptrue=0 is not defined.\n";
        return 1;
    }

    if (scanAllP) {
        gSystem->mkdir(outDir.c_str(), true);
        const std::string bigSummaryPath = outDir + "/toy_allP_big_summary.csv";
        const std::string compactSummaryPath = outDir + "/toy_scan_1pct_summary.csv";
        std::ofstream bigSummary(bigSummaryPath);
        std::ofstream compactSummary(compactSummaryPath);
        if (!bigSummary) {
            std::cerr << "ERROR: cannot write summary file: " << bigSummaryPath << "\n";
            return 1;
        }
        if (!compactSummary) {
            std::cerr << "ERROR: cannot write compact summary file: " << compactSummaryPath << "\n";
            return 1;
        }
        bigSummary
            << "status,Ptrue,sigma_target_1pct,N,mean_p,sigma_p,bias_p,toys_used,fit_failures,pass_at_N,"
            << "n_required_1pct_fit,sigma_last,pass_1pct,k_fixed,avgcos_ref_tag,afb_file,avgcos_file\n";
        compactSummary
            << "status,Ptrue,sigma_target_1pct,n_required_1pct_fit,sigma_last,pass_1pct\n";

        const std::vector<double> pVals = BuildPolarizationList(true, Ptrue);
        for (size_t i = 0; i < pVals.size(); ++i) {
            const double pRun = pVals[i];
            const std::string pTag = FormatPTag(pRun);
            const std::string avgCosRefTagThisRun =
                avgCosRefTagProvided ? avgCosRefTag : DefaultAvgCosRefTagForP(pRun);
            const std::string outSubdir = outDir + "/P_" + pTag;
            const unsigned long long seedRun = seed + (unsigned long long)(100003 * i);

            std::ostringstream cmd;
            cmd << "\"" << argv[0] << "\" ";
            if (refDirProvided) {
                cmd << "--ref-dir \"" << refDir << "\" ";
            } else {
                cmd << "--base-afb-dir \"" << baseAfbDir << "\" ";
                cmd << "--base-avgcos-dir \"" << baseAvgCosDir << "\" ";
            }
            cmd << "--avgcos-ref-tag \"" << avgCosRefTagThisRun << "\" ";
            cmd << "--k-fixed " << std::setprecision(8) << kFixedMeg << " ";
            cmd << "--out \"" << outSubdir << "\" ";
            cmd << "--p " << std::fixed << std::setprecision(1) << pRun << " ";
            cmd << "--nmin " << nMin << " --nmax " << nMax << " --nsteps " << nSteps << " ";
            cmd << "--ntoys " << nToys << " --seed " << seedRun << " ";
            cmd << "--afb-zero-dir \"" << afbZeroDir << "\"";

            std::cout << "=== scan P=" << std::fixed << std::setprecision(1) << pRun << " ===\n";
            const int rc = std::system(cmd.str().c_str());
            const std::string detailPath = outSubdir + "/toy_big_summary_singleP.csv";
            const std::string onePctPath = outSubdir + "/toy_1pct_summary.txt";

            if (rc != 0) {
                bigSummary << "RUN_FAILED," << pRun << ",NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,0,"
                           << kFixedMeg << "," << avgCosRefTagThisRun << ",\"\",\"\"\n";
                compactSummary << "RUN_FAILED," << pRun << ",0.01,NaN,NaN,0\n";
                continue;
            }

            std::ifstream in(detailPath);
            if (!in) {
                bigSummary << "DETAIL_MISSING," << pRun << ",NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,0,"
                           << kFixedMeg << "," << avgCosRefTagThisRun << ",\"\",\"\"\n";
                compactSummary << "DETAIL_MISSING," << pRun << ",0.01,NaN,NaN,0\n";
                continue;
            }

            std::string line;
            bool isHeader = true;
            while (std::getline(in, line)) {
                if (isHeader) { isHeader = false; continue; }
                if (line.empty()) continue;
                bigSummary << line << "\n";
            }

            std::ifstream onePctIn(onePctPath);
            if (!onePctIn) {
                compactSummary << "ONEPCT_MISSING," << pRun << ",0.01,NaN,NaN,0\n";
                continue;
            }

            std::map<std::string, std::string> onePctValues;
            while (std::getline(onePctIn, line)) {
                const size_t eq = line.find('=');
                if (eq == std::string::npos) continue;
                const std::string key = line.substr(0, eq);
                const std::string value = line.substr(eq + 1);
                onePctValues[key] = value;
            }

            compactSummary
                << "OK,"
                << onePctValues["Ptrue"] << ","
                << onePctValues["sigma_target_1pct"] << ","
                << onePctValues["N_required_1pct_fit"] << ","
                << onePctValues["sigma_last"] << ","
                << onePctValues["pass_1pct"] << "\n";
        }

        bigSummary.close();
        compactSummary.close();
        std::cout << "Wrote big summary: " << bigSummaryPath << "\n";
        std::cout << "Wrote compact 1% summary: " << compactSummaryPath << "\n";
        return 0;
    }

    if (!avgCosRefTagProvided) {
        avgCosRefTag = DefaultAvgCosRefTagForP(Ptrue);
    }

    //==========================================================
    // Important part: input templates
    // Load the forward/backward count templates and the avg-cos reference.
    //==========================================================
    gSystem->mkdir(outDir.c_str(), true); // ensure output directory exists early

    const std::string tag = FormatPTag(Ptrue); // format P tag
    const std::string afbFilename = "Afb_vs_x_P_" + tag + ".root"; // Afb filename
    const std::string avgCosFilename = "AvgCostheta_vs_x_P_" + avgCosRefTag + ".root"; // fixed reference avg-cos filename
    const std::string afbZeroFilename = "Afb_vs_x_P_0p0.root"; // raw P=0 Afb filename

    std::vector<std::string> afbDirs; // candidate Afb directories
    std::vector<std::string> avgCosDirs; // candidate avg-cos directories
    if (refDirProvided) { // if explicit ref dir
        afbDirs.push_back(refDir); // use explicit directory as provided
        avgCosDirs.push_back(refDir); // use explicit directory as provided
    } else { // use base dirs
        afbDirs.push_back(baseAfbDir); // Afb base dir
        avgCosDirs.push_back(baseAvgCosDir); // avg-cos directory
    } // end directory candidate setup

    const std::string afbFile = ResolveInputFile(afbDirs, afbFilename); // resolve Afb file path
    const std::string avgCosFile = ResolveInputFile(avgCosDirs, avgCosFilename); // resolve avg-cos file path
    const std::string afbZeroFile = ResolveInputFile({afbZeroDir}, afbZeroFilename); // resolve raw P=0 Afb bias path

    TFile* fAfb = TFile::Open(afbFile.c_str(), "READ"); // open Afb file
    if (!fAfb || fAfb->IsZombie()) { // check open success
        std::cerr << "ERROR: cannot open Afb file: " << afbFile << "\n"; // error
        return 1; // exit error
    } // end Afb open check

    TH1D* hF = LoadH1(fAfb, "F_vs_x"); // load forward counts
    TH1D* hB = LoadH1(fAfb, "B_vs_x"); // load backward counts
    fAfb->Close(); // close Afb file
    delete fAfb; // delete file object
    if (!hF || !hB) { // check hist availability
        std::cerr << "ERROR: missing F_vs_x or B_vs_x in: " << afbFile << "\n" // error
                  << "       Rebuild Afb files with updated MontecarloAFBvsx.\n"; // hint
        return 1; // exit error
    } // end F/B check

    gAfbZeroBias = LoadAfbTemplate(afbZeroFile);
    if (!gAfbZeroBias) {
        std::cerr << "WARN: failed to load raw P=0 Afb baseline from: " << afbZeroFile
                  << "\n      Proceeding without post-build Afb subtraction.\n";
    }

    { // input templates: F_ref(x), B_ref(x)
        //======================================================
        // Convenience and graph output
        // Diagnostic plot of the input templates.
        // This is presentation-only and does not change the toy result.
        // Parts of this graph/output setup were organised with LLM help.
        //======================================================
        TCanvas cInputFB("c_input_fb", "F and B vs x", 900, 650);
        hF->SetLineWidth(2);
        hB->SetLineWidth(2);
        hF->SetLineColor(kBlue + 1);
        hB->SetLineColor(kRed + 1);
        hF->SetTitle("Input templates; x; counts");
        hF->Draw("HIST");
        hB->Draw("HIST SAME");

        TLegend leg(0.65, 0.75, 0.88, 0.88);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.AddEntry(hF, "F_{ref}(x)", "l");
        leg.AddEntry(hB, "B_{ref}(x)", "l");
        leg.Draw();

        cInputFB.SaveAs((outDir + "/input_F_B_vs_x.pdf").c_str());
    }

    TH1D* hAref = (TH1D*)hF->Clone("Afb_ref");
    hAref->Reset("ICES");
    for (int i = 1; i <= hF->GetNbinsX(); ++i) {
        const double F = hF->GetBinContent(i);
        const double B = hB->GetBinContent(i);
        const double N = F + B;
        if (N > 0.0) hAref->SetBinContent(i, (F - B) / N);
    }
    SubtractAfbZeroBias(hAref);

    {
        //======================================================
        // Convenience and graph output
        // Diagnostic plot of the reference A_FB(x).
        // This is presentation-only and does not change the toy result.
        // Parts of this graph/output setup were organised with LLM help.
        //======================================================
        TCanvas cAref("c_afb_ref", "Afb ref", 900, 650);
        hAref->SetStats(0);
        hAref->SetLineWidth(2);
        hAref->SetTitle("A_{FB}^{ref}(x); x; A_{FB}");
        hAref->Draw("HIST");
        cAref.SaveAs((outDir + "/Afb_ref_vs_x.pdf").c_str());
    }

    const double Nref = hF->Integral(1, hF->GetNbinsX()) + // sum forward
                        hB->Integral(1, hB->GetNbinsX()); // sum backward
    if (Nref <= 0.0) { // guard zero reference
        std::cerr << "ERROR: reference N is zero.\n"; // error
        return 1; // exit error
    } // end Nref check

    gAvgCos = LoadAvgCosTemplate(avgCosFile, avgCosRefTag); // load fixed reference avg-cos template
    std::string avgCosUsedTag = avgCosRefTag; // track which template was loaded
    if (!gAvgCos) { // guard template
        std::cerr << "ERROR: failed to load avg cos(theta) template from: " << avgCosFile << "\n"; // error
        delete hAref;
        return 1; // exit error
    } // end template check
    double avgCosTemplateP = std::numeric_limits<double>::quiet_NaN();
    if (ParsePTagToValue(avgCosUsedTag, avgCosTemplateP) && std::abs(avgCosTemplateP) > 1e-12) {
        // Important part: remove the polarization already built into the
        // reference <cos(theta)>(x) file so the template is per unit polarization.
        gAvgCos->Scale(1.0 / avgCosTemplateP);
    } else {
        std::cerr << "WARN: could not infer template polarization from avg-cos tag '"
                  << avgCosUsedTag << "'; leaving avg-cos unscaled.\n";
    }
    std::cout << "Using avg cos(theta) reference tag: " << avgCosUsedTag << "\n";

    { // acceptance-weighted angular information <cos(theta)>(x)
        //======================================================
        // Convenience and graph output
        // Diagnostic plot of the fixed <cos(theta)>(x) template.
        // This is presentation-only and does not change the toy result.
        // Parts of this graph/output setup were organised with LLM help.
        //======================================================
        TCanvas cCos("c_input_avgcos", "<cos> vs x", 900, 650);
        gAvgCos->SetStats(0);
        gAvgCos->SetLineWidth(2);
        gAvgCos->SetTitle(Form("Reference <cos#theta>(x), tag=%s; x; <cos#theta>",
                               avgCosUsedTag.c_str()));
        gAvgCos->Draw("HIST");
        cCos.SaveAs((outDir + "/input_avgCos_vs_x.pdf").c_str());
    }

    const double Kglobal = kFixedMeg;
    std::cout << "Using fixed K from 10^9 MEG fit: " << Kglobal << "\n";

    //==========================================================
    // Important part: toy scan over N
    // Build toys across the N scan and measure the fitted-P spread.
    //==========================================================
    const std::vector<long long> nVals = BuildNList(nMin, nMax, nSteps); // build scan list
    if (nVals.empty()) {
        std::cerr << "ERROR: nVals is empty. Use --nsteps >= 2.\n";
        delete hF;
        delete hB;
        delete gAvgCos;
        gAvgCos = nullptr;
        return 1;
    }
    const long long nFirst = nVals.front(); // first N in scan
    const long long nMid = nVals[nVals.size() / 2]; // middle N in scan
    const long long nLast = nVals.back(); // last N in scan
    std::vector<double> nScan; // N values for graph
    std::vector<double> meanVals; // mean fitted P values
    std::vector<double> sigmaVals; // sigma values
    std::vector<double> biasVals; // bias values
    std::vector<int> toysUsedVals; // successful toy fits per N
    std::vector<int> fitFailureVals; // fit failures per N

    TRandom3 rng(seed); // RNG instance

    for (long long N : nVals) { // loop target N
        const double scale = (double)N / Nref; // scale factor relative to ref

        std::vector<double> pFits; // store fitted P values
        pFits.reserve(nToys); // reserve size
        int fitFailures = 0; // count failed fits

        for (int it = 0; it < nToys; ++it) { // loop toys
            TH1D* hAfb = BuildAfbFromFB(hF, hB, "Afb_toy", rng, scale); // build toy Afb
            SubtractAfbZeroBias(hAfb); // remove the raw P=0 bias after building the toy
            double pFit = std::numeric_limits<double>::quiet_NaN(); // fit result
            if (FitPFromAfb(hAfb, Kglobal, pFit)) {
                pFits.push_back(pFit); // store fit
            } else {
                ++fitFailures; // track rejected toys
            }
            delete hAfb; // cleanup toy hist 
            
        } // end toy loop

        if (pFits.empty()) {
            std::cerr << "WARN: all toy fits failed for N=" << N << "; skipping this scan point.\n";
            continue;
        }

        double mean = 0.0; // mean storage
        double mean2 = 0.0; // mean square storage
        for (double v : pFits) { // loop over fits
            mean += v; // sum
            mean2 += v * v; // sum squares
        } // end fit loop
        mean /= (double)pFits.size(); // compute mean
        mean2 /= (double)pFits.size(); // compute mean square
        double var = mean2 - mean * mean; // variance
        if (var < 0.0) var = 0.0; // guard negative due to precision
        const double sigma = std::sqrt(var); // standard deviation

        nScan.push_back((double)N); // store N
        meanVals.push_back(mean); // store mean fitted P
        sigmaVals.push_back(sigma); // store sigma
        biasVals.push_back(mean - Ptrue); // store bias
        toysUsedVals.push_back((int)pFits.size()); // toys used
        fitFailureVals.push_back(fitFailures); // failed fits

        std::cout << "N=" << N // print N
                  << " meanP=" << mean // print mean
                  << " sigmaP=" << sigma // print sigma
                  << " bias=" << (mean - Ptrue) // print bias
                  << " toysUsed=" << pFits.size()
                  << " fitFailures=" << fitFailures << "\n"; // print toy summary

        if (N == nFirst || N == nMid || N == nLast) { // P-hat distributions at representative N
            //==================================================
            // Convenience and graph output
            // Representative P-hat distributions for a few N values.
            // This is presentation-only and does not change the toy result.
            // Parts of this graph/output setup were organised with LLM help.
            //==================================================
            TH1D hPhat(Form("hP_%lld", N),
                       Form("#hat{P} distribution (N=%lld);#hat{P};Entries", N),
                       80, Ptrue - 0.5, Ptrue + 0.5);
            for (double v : pFits) hPhat.Fill(v);

            TCanvas cPhat(Form("c_phat_%lld", N), "Phat", 900, 650);
            hPhat.SetLineWidth(2);
            hPhat.Draw("HIST");
            TLine lTrue(Ptrue, 0.0, Ptrue, hPhat.GetMaximum());
            lTrue.SetLineStyle(2);
            lTrue.SetLineWidth(2);
            lTrue.Draw("SAME");
            cPhat.SaveAs((outDir + "/Phat_hist_N_" + std::to_string(N) + ".pdf").c_str());
        }
    } // end N loop

    if (nScan.empty()) {
        std::cerr << "ERROR: no valid N scan points were produced.\n";
        delete hF;
        delete hB;
        delete hAref;
        delete gAvgCos;
        gAvgCos = nullptr;
        return 1;
    }

    const double sigmaTarget1 = 0.01;
    const double sigmaTarget5 = 0.05;

    std::cout << "Sigma target (1% of |Ptrue|) = " << sigmaTarget1 << "\n";

    if (!nScan.empty() && nScan.size() == meanVals.size() && meanVals.size() == sigmaVals.size()) {
        //======================================================
        // Convenience and graph output
        // Summary plot of <P_hat> with its spread across N.
        // This is presentation-only and does not change the toy result.
        // Parts of this graph/output setup were organised with LLM help.
        //======================================================
        TGraphErrors gMeanErr( // <P> +/- sigma vs N
            (int)nScan.size(),
            meanVals.data(),
            nScan.data(),
            sigmaVals.data(),
            nullptr
        );
        gMeanErr.SetTitle("N vs <#hat{P}> #pm #sigma;<#hat{P}>;N_{accepted}");
        gMeanErr.SetMarkerStyle(20);
        gMeanErr.SetMarkerSize(0.8);

        TCanvas cMeanErr("c_mean_err", "<P> vs N", 900, 650);
        cMeanErr.SetLogy();
        gMeanErr.Draw("AP");

        TLine lineTrueMean(Ptrue, nScan.front(), Ptrue, nScan.back());
        lineTrueMean.SetLineStyle(2);
        lineTrueMean.SetLineWidth(2);
        lineTrueMean.Draw("SAME");

        cMeanErr.SaveAs((outDir + "/meanP_pm_sigma_vs_N.pdf").c_str());
    }

    TGraph gSigma((int)nScan.size(), nScan.data(), sigmaVals.data()); // graph for sigma
    TF1 fScale("fScale", "[0]/sqrt(x)", nScan.front(), nScan.back());
    fScale.SetParameter(0, 1.0); // initial guess

    bool kFitValid = false;
    double k = std::numeric_limits<double>::quiet_NaN();
    if (nScan.size() >= 2) {
        gSigma.Fit(&fScale, "Q0"); // dont drawing the fitted line
        k = fScale.GetParameter(0);
        kFitValid = std::isfinite(k) && (k > 0.0);
    }
    auto N_from_sigma = [&](double thr) -> double {
        if (thr <= 0.0 || !kFitValid) return 0.0;
        return (k / thr) * (k / thr);
    };
    fScale.SetLineWidth(2);

    const double nAt1_fit = N_from_sigma(sigmaTarget1);
    const double nAt5_fit = N_from_sigma(sigmaTarget5);

    std::cout << "Fit k = " << k << "\n";
    std::cout << "N_required(1%) (fit) = " << (long long)llround(nAt1_fit) << "\n";

    const double sigmaLast = sigmaVals.empty() ? std::numeric_limits<double>::quiet_NaN()
                                               : sigmaVals.back();
    const double nMaxScanned = nScan.back();
    const int pass1pct =
        (std::isfinite(sigmaLast) && sigmaLast <= sigmaTarget1) ? 1 : 0;

    const std::string onePctSummaryPath = outDir + "/toy_1pct_summary.txt";
    {
        std::ofstream onePct(onePctSummaryPath);
        if (onePct) {
            onePct << std::setprecision(17);
            onePct << "Ptrue=" << Ptrue << "\n";
            onePct << "sigma_target_1pct=" << sigmaTarget1 << "\n";
            onePct << "N_required_1pct_fit=" << nAt1_fit << "\n";
            onePct << "sigma_last=" << sigmaLast << "\n";
            onePct << "pass_1pct=" << pass1pct << "\n";
        }
    }

    const std::string singleSummaryPath = outDir + "/toy_big_summary_singleP.csv";
    {
        std::ofstream detail(singleSummaryPath);
        if (detail) {
            detail
                << "status,Ptrue,sigma_target_1pct,N,mean_p,sigma_p,bias_p,toys_used,fit_failures,pass_at_N,"
                << "n_required_1pct_fit,sigma_last,pass_1pct,k_fixed,avgcos_ref_tag,afb_file,avgcos_file\n";
            for (size_t i = 0; i < nScan.size(); ++i) {
                const long long nVal = (long long)llround(nScan[i]);
                const int passAtN = (sigmaVals[i] <= sigmaTarget1) ? 1 : 0;
                detail << "OK,"
                       << std::fixed << std::setprecision(8) << Ptrue << ","
                       << sigmaTarget1 << ","
                       << nVal << ","
                       << meanVals[i] << ","
                       << sigmaVals[i] << ","
                       << biasVals[i] << ","
                       << toysUsedVals[i] << ","
                       << fitFailureVals[i] << ","
                       << passAtN << ","
                       << nAt1_fit << ","
                       << sigmaLast << ","
                       << pass1pct << ","
                       << Kglobal << ","
                       << avgCosUsedTag << ","
                       << "\"" << afbFile << "\","
                       << "\"" << avgCosFile << "\"\n";
            }
        }
    }

    std::cout << "Wrote 1% summary: " << onePctSummaryPath << "\n";
    std::cout << "Wrote detailed summary: " << singleSummaryPath << "\n";

    //==========================================================
    // Convenience and graph output
    // Example toy and fit-overlay plots at the predicted 1% point.
    // This is presentation-only and does not change the toy result.
    // Parts of this graph/output setup were organised with LLM help.
    //==========================================================
    // Build Afb example plots at the predicted N for 1%
    double nPred1 = nAt1_fit;
    if (nPred1 <= 0.0) nPred1 = (double)nVals.back(); // fallback if 1% prediction not found
    const long long nPred1Rounded = llround(nPred1);
    std::cout << "Using predicted N=" << nPred1Rounded
              << " for Afb_oneToy_vs_x and Afb_fit_overlay (target sigma="
              << sigmaTarget1 << ", N="
              << nPred1Rounded << ")\n";

    {
        TRandom3 rngPlot(seed + 99991ULL); // stable toy for diagnostic plots
        const double scalePlot = nPred1 / Nref;
        TH1D* hAfbPlot = BuildAfbFromFB(hF, hB, "Afb_toy_plot_1pct", rngPlot, scalePlot);
        SubtractAfbZeroBias(hAfbPlot);

        TCanvas cAtoy("c_afb_onetoy", "Afb toy", 900, 650);
        hAfbPlot->SetMarkerStyle(20);
        hAfbPlot->SetMarkerSize(0.8);
        hAfbPlot->SetTitle(Form("One toy A_{FB}(x) at predicted 1%% N (%lld); x; A_{FB}", nPred1Rounded));
        hAfbPlot->Draw("E1");
        cAtoy.SaveAs((outDir + "/Afb_oneToy_vs_x.pdf").c_str());

        TF1 fPdraw("fPdraw_overlay", AfbModel,
                   std::max(0.4, hAfbPlot->GetXaxis()->GetXmin()),
                   hAfbPlot->GetXaxis()->GetXmax(), 2);
        fPdraw.FixParameter(1, Kglobal);
        fPdraw.SetParameter(0, Ptrue);
        fPdraw.SetLineColor(kRed + 1);
        fPdraw.SetLineWidth(2);

        TCanvas cFit("c_afb_fit_overlay", "Fit overlay", 900, 650);
        hAfbPlot->SetTitle(Form("A_{FB}(x) fit overlay at predicted 1%% N (%lld); x; A_{FB}", nPred1Rounded));
        hAfbPlot->Draw("E1");
        hAfbPlot->Fit(&fPdraw, "QR");
        cFit.SaveAs((outDir + "/Afb_fit_overlay.pdf").c_str());

        delete hAfbPlot;
    }

    //==========================================================
    // Convenience and graph output
    // Final summary graphs for sigma and bias vs N.
    // This is presentation-only and does not change the toy result.
    // Parts of this graph/output setup were organised with LLM help.
    //==========================================================
    gSigma.SetTitle("#sigma_{P} vs N;N_{accepted};#sigma_{P}"); // graph title
    gSigma.SetMarkerStyle(20); // marker style
    gSigma.SetMarkerSize(0.9); // marker size
    gSigma.SetLineWidth(2); // line width

    TGraph gBias((int)nScan.size(), nScan.data(), biasVals.data()); // graph for bias
    gBias.SetTitle("Bias vs N;N_{accepted};<#hat{P}>-P_{true}"); // graph title
    gBias.SetMarkerStyle(21); // marker style
    gBias.SetMarkerSize(0.9); // marker size
    gBias.SetLineWidth(2); // line width

    TCanvas cSigma("c_sigma_vs_n", "Sigma vs N", 900, 650); // create sigma canvas
    cSigma.SetLogx();
    gSigma.Draw("AP"); // draw measured sigma values as points only
    fScale.Draw("SAME"); // draw fitted scaling as the only line on the plot

    const double xMin = nScan.front(); // min x for lines
    const double xMax = nScan.back(); // max x for lines

    TLegend leg(0.60, 0.75, 0.88, 0.88); // legend box
    leg.SetBorderSize(0); // no border
    leg.SetFillStyle(0); // transparent fill
    leg.AddEntry(&gSigma, "Plotted #sigma_{P}", "p"); // plotted points entry
    leg.AddEntry(&fScale, "Fitted k/#sqrt{N}", "l"); // fitted curve entry
    leg.Draw(); // draw legend

    const std::string pdfSigmaPath = outDir + "/sigma_P_vs_N.pdf"; // output sigma PDF path
    const std::string pdfBiasPath = outDir + "/bias_P_vs_N.pdf"; // output bias PDF path
    const std::string rootPath = outDir + "/toy_sigma_bias_vs_N.root"; // output ROOT path
    cSigma.SaveAs(pdfSigmaPath.c_str()); // save sigma PDF

    TCanvas cBias("c_bias_vs_n", "Bias vs N", 900, 650); // create bias canvas
    cBias.SetLogx();
    gBias.Draw("APL");
    TLine lineZero(xMin, 0.0, xMax, 0.0);
    lineZero.SetLineStyle(2);
    lineZero.SetLineWidth(2);
    lineZero.Draw("SAME");
    cBias.SaveAs(pdfBiasPath.c_str()); // save bias PDF

    TFile outFile(rootPath.c_str(), "RECREATE"); // open ROOT output
    gSigma.Write("sigma_P_vs_N"); // write sigma graph
    gBias.Write("bias_P_vs_N"); // write bias graph
    outFile.Close(); // close ROOT output

    delete hF; // cleanup F
    delete hB; // cleanup B
    delete hAref; // cleanup reference Afb
    delete gAfbZeroBias; // cleanup Afb zero-bias template
    gAfbZeroBias = nullptr; // clear pointer
    delete gAvgCos; // cleanup template
    gAvgCos = nullptr; // clear pointer
    return 0; // success
} // end main
