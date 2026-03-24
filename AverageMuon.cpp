#include <cmath>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// File guide
//
// Important analysis section:
// - LoadStdRows(...) loads the fitted spread in P for each true polarization.
// - LoadGpu1PctFitByP(...) loads the GPU event count already associated with 1%.
// - the main loop combines both inputs to derive k and the required event count.
//
// What this file does:
// - reads the ROOT-fit summary table and the GPU summary table
// - matches them by Ptrue
// - uses std = k / sqrt(N) to extract k
// - computes the event count needed for the chosen 1% target
// - writes the combined summary to a new CSV file
//
// Convenience-only section:
// - CSV splitting/parsing helpers, column lookup, CLI parsing, and text output.
//
// LLM-assisted / readability-only section:
// - the section labels and convenience annotations below are for readability
//   and traceability; they do not change the calculation.
//
// Existing input/output paths are intentionally kept in place.
// ============================================================

struct StdRow {
    // Polarization value for this summary row.
    double pTrue = std::numeric_limits<double>::quiet_NaN();
    // Observed spread of fitted P values at this Ptrue.
    double sigma = std::numeric_limits<double>::quiet_NaN();
    // Remember which sigma column was used so the output stays traceable.
    std::string sigmaColumn;
};

// ------------------------------------------------------------
// Convenience section: minimal CSV parsing and column lookup
// ------------------------------------------------------------

// Minimal CSV helpers: enough for the summary files used here.
// These files are simple comma-separated tables with a single header row.
static std::string Trim(const std::string& s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static std::vector<std::string> SplitCsvSimple(const std::string& line)
{
    // Split one CSV line into fields.
    // Quoted commas are kept inside the current field.
    std::vector<std::string> out;
    std::string cur;
    bool inQuotes = false;
    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            out.push_back(Trim(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(Trim(cur));
    return out;
}

static int FindCol(const std::vector<std::string>& header, const std::string& name)
{
    // Return the position of a named column, or -1 if it is absent.
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) return static_cast<int>(i);
    }
    return -1;
}

static bool ParseDouble(const std::string& s, double& out)
{
    // Parse a finite floating-point value from a CSV cell.
    // Invalid strings, overflows, and trailing non-space characters are rejected.
    const std::string t = Trim(s);
    if (t.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(t.c_str(), &end);
    if (end == t.c_str()) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// ------------------------------------------------------------
// Important input section: load GPU and fit-summary inputs
// ------------------------------------------------------------

static std::map<double, double> LoadGpu1PctFitByP(const std::string& csvPath)
{
    // Read the GPU-produced summary table.
    // We only need two columns:
    // - Ptrue
    // - n_required_1pct_fit
    //
    // The returned map lets us look up the fitted GPU N for each polarization.
    std::map<double, double> out;
    std::ifstream ifs(csvPath);
    if (!ifs) {
        throw std::runtime_error("Cannot open GPU summary CSV: " + csvPath);
    }

    std::string line;
    if (!std::getline(ifs, line)) {
        throw std::runtime_error("GPU summary CSV is empty: " + csvPath);
    }

    const std::vector<std::string> header = SplitCsvSimple(line);
    const int iP = FindCol(header, "Ptrue");
    const int iNFit = FindCol(header, "n_required_1pct_fit");
    if (iP < 0 || iNFit < 0) {
        throw std::runtime_error(
            "GPU summary CSV missing required columns: Ptrue and n_required_1pct_fit");
    }

    while (std::getline(ifs, line)) {
        if (Trim(line).empty()) continue;
        const std::vector<std::string> cols = SplitCsvSimple(line);
        auto getCol = [&](int idx) -> std::string {
            if (idx < 0 || idx >= static_cast<int>(cols.size())) return "";
            return cols[static_cast<size_t>(idx)];
        };

        double pTrue = std::numeric_limits<double>::quiet_NaN();
        double nFit = std::numeric_limits<double>::quiet_NaN();
        if (!ParseDouble(getCol(iP), pTrue)) continue;
        // For each Ptrue keep the GPU event count that was fitted to reach 1%.
        if (!ParseDouble(getCol(iNFit), nFit) || nFit <= 0.0) continue;
        // Later rows overwrite earlier ones if the same Ptrue appears twice.
        out[pTrue] = nFit;
    }

    return out;
}

static std::vector<StdRow> LoadStdRows(const std::string& csvPath)
{
    // Read the table containing the spread of fitted polarization values.
    // This code accepts either:
    // - sigma_pfit      (current ROOT summary)
    // - std_population  (older summary format)
    // - std_sample      (older summary format)
    std::ifstream ifs(csvPath);
    if (!ifs) {
        throw std::runtime_error("Cannot open std summary CSV: " + csvPath);
    }

    std::string line;
    if (!std::getline(ifs, line)) {
        throw std::runtime_error("Std summary CSV is empty: " + csvPath);
    }

    const std::vector<std::string> header = SplitCsvSimple(line);
    const int iP = FindCol(header, "Ptrue");
    // Accept either the ROOT summary column name or the older pfit_variance names.
    int iSigma = FindCol(header, "sigma_pfit");
    std::string sigmaColumn = "sigma_pfit";
    if (iSigma < 0) {
        iSigma = FindCol(header, "std_population");
        sigmaColumn = "std_population";
    }
    if (iSigma < 0) {
        iSigma = FindCol(header, "std_sample");
        sigmaColumn = "std_sample";
    }
    if (iP < 0 || iSigma < 0) {
        throw std::runtime_error(
            "Std summary CSV missing required columns: Ptrue and sigma_pfit/std_population/std_sample");
    }

    std::vector<StdRow> rows;
    while (std::getline(ifs, line)) {
        if (Trim(line).empty()) continue;
        const std::vector<std::string> cols = SplitCsvSimple(line);
        auto getCol = [&](int idx) -> std::string {
            if (idx < 0 || idx >= static_cast<int>(cols.size())) return "";
            return cols[static_cast<size_t>(idx)];
        };

        StdRow row;
        if (!ParseDouble(getCol(iP), row.pTrue)) continue;
        // Sigma is the fitted spread in P for this true polarization.
        if (!ParseDouble(getCol(iSigma), row.sigma) || row.sigma <= 0.0) continue;
        row.sigmaColumn = sigmaColumn;
        rows.push_back(row);
    }

    if (rows.empty()) {
        throw std::runtime_error("No valid Ptrue/std rows found in: " + csvPath);
    }
    return rows;
}

// ------------------------------------------------------------
// Main workflow section: combine both summaries and derive N
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    // Edit these defaults here if you want to drive runs from the source file
    // instead of passing CLI arguments.
    // Default inputs are the current ROOT fit summary and the GPU 1% summary.
    // The output is a new CSV with the derived K and required N values.
    std::string stdSummaryCsv =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/Variancce test/Root analyse/pfit_summary_rootfit.txt";
    std::string gpuSummaryCsv =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3 layer MEG/CPU toys/toy_scan_1pct_summary.csv";
    std::string outCsv =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/Variancce test/Root analyse/average_muon_summary.csv";

    // Convenience-only CLI parsing.
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        // Allow the user to override either input file and the output path.
        if ((a == "--std-summary" || a == "--std") && i + 1 < argc) {
            stdSummaryCsv = argv[++i];
        } else if ((a == "--gpu-summary" || a == "--gpu") && i + 1 < argc) {
            gpuSummaryCsv = argv[++i];
        } else if ((a == "--out" || a == "--output") && i + 1 < argc) {
            outCsv = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " [--std-summary <csv>] [--gpu-summary <csv>] [--out <csv>]\n"
                << "  --std-summary  summary with Ptrue and sigma_pfit/std_population/std_sample\n"
                << "  --gpu-summary  summary with Ptrue and n_required_1pct_fit\n"
                << "  --out          output CSV path\n"
                << "  Default behavior can also be edited directly in this .cpp file.\n";
            return 0;
        }
    }

    try {
        // Load the per-P spread from the fit study and the GPU N that already hits 1%.
        const std::vector<StdRow> stdRows = LoadStdRows(stdSummaryCsv);
        const std::map<double, double> gpuNByP = LoadGpu1PctFitByP(gpuSummaryCsv);

        std::ofstream out(outCsv);
        if (!out) {
            throw std::runtime_error("Cannot write output CSV: " + outCsv);
        }

        out << "Ptrue,sigma_source,sigma_value,gpu_n_required_1pct_fit,"
            << "k_from_sigma_sqrtN,target_std_1pct,n_required_1pct_from_k\n";

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Using std summary: " << stdSummaryCsv << "\n";
        std::cout << "Using gpu summary: " << gpuSummaryCsv << "\n";
        std::cout << "Writing: " << outCsv << "\n";

        // Process each Ptrue row independently.
        // For every polarization we:
        // 1. get the measured sigma from the fit study
        // 2. get the GPU N that was fitted for 1%
        // 3. solve std = k / sqrt(N) for k
        // 4. plug in target std = 1% * |Ptrue| and solve for N
        for (const StdRow& row : stdRows) {
            double gpuN = std::numeric_limits<double>::quiet_NaN();
            const auto it = gpuNByP.find(row.pTrue);
            if (it != gpuNByP.end()) {
                gpuN = it->second;
            }

            double k = std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(row.sigma) && std::isfinite(gpuN) && gpuN > 0.0) {
                // From std = k / sqrt(N), solve for k using the GPU-fitted N.
                // This makes k the normalization constant implied by the current
                // sigma measurement and the known GPU event count.
                k = row.sigma * std::sqrt(gpuN);
            }

            double targetStd = std::numeric_limits<double>::quiet_NaN();
            if (std::abs(row.pTrue) > 1e-12) {
                // The requested target is a 1% relative precision on Ptrue.
                // Use |Ptrue| so the target spread stays positive for negative P.
                targetStd = 0.01;
            }

            double targetN = std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(k) && std::isfinite(targetStd) && targetStd > 0.0) {
                // Rearranged from std = k / sqrt(N): N = (k / std)^2.
                // This is the event count needed so the expected spread reaches 1%.
                const double ratio = k / targetStd;
                targetN = ratio * ratio;
            }

            // Write one output row per Ptrue so the result can be reused by other tools.
            out << std::fixed << std::setprecision(8)
                << row.pTrue << "," << row.sigmaColumn << "," << row.sigma << ","
                << gpuN << "," << k << "," << targetStd << "," << targetN << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
