//==============================================================
// Muon Michel Monte Carlo on GPU
// - Generates accepted events in (x, cos(theta))
// - Applies the acceptance map before filling the histogram
// - Scans the muon polarization P
//
// OUTPUT:
//   (1) ROOT histogram for each P value
//   (2) PDF snapshot for each P value
//==============================================================

// ROOT includes
#include "TFile.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TLatex.h"

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

// C++
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//==============================================================
// Parameters
//==============================================================
constexpr int numberOfBinsX        = 100;
constexpr int numberOfBinsCosTheta = 100;

// theta range
static const double thetaMinPlot =M_PI/2-1.3;
static const double thetaMaxPlot = M_PI/2+1.3;

// cos(theta) range
static const double cosThetaMin =
    std::min(std::cos(thetaMinPlot), std::cos(thetaMaxPlot));
static const double cosThetaMax =
    std::max(std::cos(thetaMinPlot), std::cos(thetaMaxPlot));

// energy range
constexpr double E_MAX    = 52.83;
constexpr double E_MIN    = 10.0;
constexpr double xMinPlot = E_MIN/E_MAX;
constexpr double xMaxPlot = 1.0;

// PDG constant
constexpr double x0 = 9.67e-3;

//==============================================================
// Constant memory
//==============================================================
__device__ __constant__ double d_cosThetaMin;
__device__ __constant__ double d_cosThetaMax;
__device__ __constant__ double d_xMinPlot;
__device__ __constant__ double d_xMaxPlot;
__device__ __constant__ float d_cosThetaMin_f;
__device__ __constant__ float d_cosThetaMax_f;
__device__ __constant__ float d_xMinPlot_f;
__device__ __constant__ float d_xMaxPlot_f;

//==============================================================
// Reduction kernel
// Reduce per-block histograms into one final histogram.
//==============================================================
__global__ void reduce_block_histograms(
    const unsigned int* __restrict__ d_blockHist, // size: nBlocks * histSize
    unsigned int* __restrict__ d_out,              // size: histSize
    int histSize,
    int nBlocks
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= histSize) return;

    unsigned int sum = 0;
    // sum bin k across all blocks
    for (int b = 0; b < nBlocks; ++b) {
        sum += d_blockHist[(size_t)b * (size_t)histSize + (size_t)k];
    }
    d_out[k] = sum;
}

//============================================================== 
// CUDA error check macro
//==============================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

//==============================================================
// Progress display
//==============================================================
static void PrintProgress(unsigned long long done,
                          unsigned long long total,
                          double elapsedSec)
{
    const int barWidth = 40;

    double frac = (total > 0) ? (double)done / (double)total : 0.0;
    if (frac > 1.0) frac = 1.0;

    int filled = (int)(barWidth * frac);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i)
        std::cout << (i < filled ? '=' : ' ');
    std::cout << "] "
              << std::fixed << std::setprecision(1)
              << (100.0 * frac) << "%  "
              << done << "/" << total
              << "  elapsed " << std::setprecision(2) << elapsedSec << " s"
              << std::flush;

    if (done >= total) std::cout << "\n";
}

//==============================================================
// Device helpers
//==============================================================
__device__ __forceinline__ double clampDouble(double x, double lo, double hi)
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

__device__ __forceinline__ int valueToBin(double v, double vmin, double vmax, int nbins)
{
    if (v < vmin || v >= vmax) return -1;
    double f = (v - vmin) / (vmax - vmin);
    int bin0 = (int)(f * nbins);
    return bin0 + 1; // ROOT-style 1..nbins
}

__device__ __forceinline__ int valueToBin_f(float v, float vmin, float vmax, int nbins)
{
    if (v < vmin || v >= vmax) return -1;
    float f = (v - vmin) / (vmax - vmin);
    int bin0 = (int)(f * nbins);
    return bin0 + 1; // ROOT-style 1..nbins
}

__device__ __forceinline__ double binCenterFromBin(int bin1, double vmin, double vmax, int nbins)
{
    double width = (vmax - vmin) / nbins;
    return vmin + (bin1 - 0.5) * width;
}
__device__ __forceinline__ bool inRange_f(float v, float vmin, float vmax)
{
    // [vmin, vmax)
    return (v >= vmin && v < vmax);
}

//==============================================================
// Michel model and acceptance
//==============================================================
__device__ __forceinline__ double michel_device(double xp, double cosTheta,
                                                double rho, double eta,
                                                double epsilon, double delta,
                                                double P)
{
    if (xp <= 0.0 || xp > 1.0) return 0.0;

    double value =
        xp * xp *
        (3.0 * (1.0 - xp)
         + 2.0 * rho * (4.0 * xp / 3.0 - 1.0)
         + 3.0 * eta * x0 * (1.0 - xp) / xp
         + P * epsilon * cosTheta *
               (1.0 - xp + (2.0 / 3.0) * delta * (4.0 * xp - 3.0)));

    return (value > 0.0) ? value : 0.0;
}

__device__ __forceinline__ bool AcceptancePass_device(double xCenter, double cosThetaCenter)
{
    double c = clampDouble(cosThetaCenter, -1.0, 1.0);
    double thetaCenter = acos(c);

    double t2 = thetaCenter * thetaCenter;

    if (xCenter < 0.3676 * t2 - 0.8352 * thetaCenter + 0.6687 &&
        thetaCenter < 1.2)
        return false;

    if (xCenter < 0.4518 * t2 - 1.8704 * thetaCenter + 2.1501 &&
        thetaCenter > 0.5 + M_PI / 2.0)
        return false;

    return true;
}

//==============================================================
// RNG init kernel
//==============================================================
__global__ void init_rng(curandStatePhilox4_32_10_t* states,
                         int nStates,
                         unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nStates) return;
    curand_init(seed, tid, 0ULL, &states[tid]);
}

//==============================================================
// Generator kernel
// Each block writes into its own histogram region in global memory.
//==============================================================
__global__ void generate_histogram_chunk(
    unsigned int* d_blockHist,                 // size: blocks * histSize
    unsigned long long* acceptedCount,
    unsigned long long* trialsCount,
    long long N,
    curandStatePhilox4_32_10_t* states,
    int nStates,
    const unsigned char* d_accept,
    const float* d_rate,
    float maxAllowedRate,
    int nTrialsPerThread
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nStates) return;

    const int histSize = numberOfBinsX * numberOfBinsCosTheta;
    unsigned int* myHist = d_blockHist + (size_t)blockIdx.x * (size_t)histSize;

    curandStatePhilox4_32_10_t local = states[tid];

    unsigned long long localTrials = 0;
    unsigned long long stopSeen    = 0;

    for (int t = 0; t < nTrialsPerThread; ++t) {

        localTrials++;

        // stop check every 64 iterations
        if ((t & 63) == 0) {
            stopSeen = atomicAdd(acceptedCount, 0ULL);
            if (stopSeen >= (unsigned long long)N) break;
        }

        // draw four uniform random numbers at once
        float4 u4 = curand_uniform4(&local);

        float randomX = d_xMinPlot_f + (d_xMaxPlot_f - d_xMinPlot_f) * u4.x;
        if (randomX >= d_xMaxPlot_f) randomX = nextafterf(d_xMaxPlot_f, d_xMinPlot_f);
        const float randomCosTheta =
            d_cosThetaMin_f + (d_cosThetaMax_f - d_cosThetaMin_f) * u4.y;

        // histogram bin used for the acceptance map
        const int binX = valueToBin_f(randomX, d_xMinPlot_f, d_xMaxPlot_f, numberOfBinsX);
        const int binY = valueToBin_f(randomCosTheta, d_cosThetaMin_f, d_cosThetaMax_f, numberOfBinsCosTheta);
        if (binX < 1 || binY < 1) continue;

        const int ix0 = binX - 1;
        const int iy0 = binY - 1;
        const int binIndex = ix0 + numberOfBinsX * iy0;

        if (!d_accept[binIndex]) continue;

        // rejection sampling at bin centers
        const float realRate = d_rate[binIndex];
        if (realRate <= 0.0f) continue;
        const float r = maxAllowedRate * u4.z;
        if (r >= realRate) continue;

        // Apply x smearing only after the event has already passed acceptance.
        float sigma = 0.0512f * randomX - 0.0039f;
        if (sigma < 0.0f) sigma = 0.0f;

        float xSmeared = randomX;

        // Gaussian smearing of x
        if (sigma > 0.0f) {
            float2 n2 = curand_normal2(&local);
            float trial = randomX + sigma * n2.x;

            // Keep the original value if the smeared point leaves the range.
            if (trial >= d_xMinPlot_f && trial < d_xMaxPlot_f) {
                xSmeared = trial;
            }
        }

        // Fill the final histogram with smeared x and original cos(theta).
        const int fillBinX = valueToBin_f(xSmeared, d_xMinPlot_f, d_xMaxPlot_f, numberOfBinsX);
        const int fillBinY = valueToBin_f(randomCosTheta, d_cosThetaMin_f, d_cosThetaMax_f, numberOfBinsCosTheta);
        if (fillBinX < 1 || fillBinY < 1) continue;

        // Reserve one accepted-event slot.
        const unsigned long long idx = atomicAdd(acceptedCount, 1ULL);
        if (idx >= (unsigned long long)N) break;

        const int histIndex = (fillBinX - 1) + numberOfBinsX * (fillBinY - 1);

        // Atomic add into the block-local histogram.
        atomicAdd(&myHist[histIndex], 1U);
    }

    // Accumulate the number of trials attempted by this thread.
    atomicAdd(trialsCount, localTrials);

    // Store the updated RNG state.
    states[tid] = local;
}

//==============================================================
// Formatting helper
//==============================================================
static std::string FormatPTag(double P)
{
    std::ostringstream tag;
    tag << std::fixed << std::setprecision(1) << P;
    std::string s = tag.str();
    for (char &c : s) if (c == '.') c = 'p';
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

//==============================================================
// Rejection envelope
// Find the maximum rate used for rejection sampling.
//==============================================================
double findMaximumRate_host(const double* par, double P)
{
    double rho     = par[0];
    double eta     = par[1];
    double epsilon = par[2];
    double delta   = par[3];

    double maxRate = 0.0;

    for (int ix = 1; ix <= numberOfBinsX; ++ix) {
        double xC = xMinPlot + (ix - 0.5) * (xMaxPlot - xMinPlot) / numberOfBinsX;

        for (int iy = 1; iy <= numberOfBinsCosTheta; ++iy) {
            double cC = cosThetaMin + (iy - 0.5) * (cosThetaMax - cosThetaMin) / numberOfBinsCosTheta;

            double thetaC = std::acos(std::clamp(cC, -1.0, 1.0));
            double t2 = thetaC * thetaC;

            bool pass = true;
            if (xC < 0.3676 * t2 - 0.8352 * thetaC + 0.6687 && thetaC < 1.2) pass = false;
            if (xC < 0.4518 * t2 - 1.8704 * thetaC + 2.1501 && thetaC > 0.5 + M_PI / 2.0) pass = false;
            if (!pass) continue;

            double val =
                xC * xC *
                (3.0 * (1.0 - xC)
                 + 2.0 * rho * (4.0 * xC / 3.0 - 1.0)
                 + 3.0 * eta * x0 * (1.0 - xC) / xC
                 + P * epsilon * cC *
                       (1.0 - xC + (2.0 / 3.0) * delta * (4.0 * xC - 3.0)));

            if (val > maxRate) maxRate = val;
        }
    }
    return maxRate;
}

//==============================================================
// Run one polarization value
//==============================================================
void RunOneP_GPU(long long N,
                 const double* par,
                 double P,
                 const std::string& outDir)
{
    const std::string tag      = FormatPTag(P);
    const std::string rootFile = outDir + "/MC_P_" + tag + ".root";
    const std::string pdfFile  = outDir + "/MC_P_" + tag + ".pdf";
    const std::string hname    = "HistogramRes_P_" + tag;

    // Maximum rate for the rejection envelope.
    double maxAllowedRate = findMaximumRate_host(par, P);
    float maxAllowedRate_f = (float)maxAllowedRate;
    std::cout << "[P=" << P << "] maxAllowedRate=" << maxAllowedRate << "\n";

    // Histogram size and precomputed acceptance/rate tables.
    const int histSize = numberOfBinsX * numberOfBinsCosTheta;
    std::vector<unsigned char> h_accept(histSize, 0);
    std::vector<float> h_rate(histSize, 0.0f);
    for (int iy = 1; iy <= numberOfBinsCosTheta; ++iy) {
        double cC = cosThetaMin + (iy - 0.5) * (cosThetaMax - cosThetaMin) / numberOfBinsCosTheta;
        double thetaC = std::acos(std::clamp(cC, -1.0, 1.0));
        double t2 = thetaC * thetaC;
        for (int ix = 1; ix <= numberOfBinsX; ++ix) {
            double xC = xMinPlot + (ix - 0.5) * (xMaxPlot - xMinPlot) / numberOfBinsX;

            bool pass = true;
            if (xC < 0.3676 * t2 - 0.8352 * thetaC + 0.6687 && thetaC < 1.2) pass = false;
            if (xC < 0.4518 * t2 - 1.8704 * thetaC + 2.1501 && thetaC > 0.5 + M_PI / 2.0) pass = false;
            if (!pass) continue;

            int idx = (ix - 1) + numberOfBinsX * (iy - 1);
            h_accept[idx] = 1;

            double val =
                xC * xC *
                (3.0 * (1.0 - xC)
                 + 2.0 * par[0] * (4.0 * xC / 3.0 - 1.0)
                 + 3.0 * par[1] * x0 * (1.0 - xC) / xC
                 + P * par[2] * cC *
                       (1.0 - xC + (2.0 / 3.0) * par[3] * (4.0 * xC - 3.0)));
            if (val > 0.0) h_rate[idx] = (float)val;
        }
    }

    // Copy acceptance and rate tables to the GPU.
    unsigned char* d_accept = nullptr;
    CUDA_CHECK(cudaMalloc(&d_accept, (size_t)histSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_accept, h_accept.data(),
                          (size_t)histSize * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    float* d_rate = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rate, (size_t)histSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rate, h_rate.data(),
                          (size_t)histSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Accepted-event counter.
    unsigned long long* d_accepted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_accepted, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_accepted, 0, sizeof(unsigned long long)));

    // Trial counter for diagnostics.
    unsigned long long* d_trials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_trials, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_trials, 0, sizeof(unsigned long long)));

    // launch config
    const int threadsPerBlock = 256;
    const int blocks          = 1360;
    const int nStates         = blocks * threadsPerBlock;

    // Per-block histogram storage.
    const size_t blockHistBytes =
        (size_t)blocks * (size_t)histSize * sizeof(unsigned int);

    unsigned int* d_blockHist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockHist, blockHistBytes));
    CUDA_CHECK(cudaMemset(d_blockHist, 0, blockHistBytes)); 

    // RNG states.
    curandStatePhilox4_32_10_t* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, (size_t)nStates * sizeof(curandStatePhilox4_32_10_t)));

    // Fixed random seed.
    unsigned long long seed = 123456789ULL;

    init_rng<<<blocks, threadsPerBlock>>>(d_states, nStates, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // chunk loop settings
    const int nTrialsPerThread = 4096;
    const int reportEvery      = 50;

    unsigned long long h_accepted = 0ULL;
    unsigned long long h_trials   = 0ULL;

    // Kernel-only timing.
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    double kernelSecondsTotal = 0.0;

    int launchCount = 0;
    auto t0 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventRecord(ev0));
    while (true) {

        generate_histogram_chunk<<<blocks, threadsPerBlock>>>(
            d_blockHist, d_accepted, d_trials, N, d_states, nStates,
            d_accept, d_rate, maxAllowedRate_f,
            nTrialsPerThread
        );
        CUDA_CHECK(cudaGetLastError());

        ++launchCount;

        if (launchCount % reportEvery == 0) {
            // Read back the accepted-event counter for progress reporting.
            CUDA_CHECK(cudaMemcpy(&h_accepted, d_accepted,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[P=" << std::fixed << std::setprecision(1) << P << "] ";
            PrintProgress(h_accepted, (unsigned long long)N, elapsed);
            if (h_accepted >= (unsigned long long)N) break;
        }
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    kernelSecondsTotal = 1e-3 * ms;

    // Final accepted count.
    CUDA_CHECK(cudaMemcpy(&h_accepted, d_accepted,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    // Final trial count.
    CUDA_CHECK(cudaMemcpy(&h_trials, d_trials,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    std::cout << "[P=" << std::fixed << std::setprecision(1) << P << "] "
              << "trials = " << h_trials
              << ", accepted = " << h_accepted
              << ", trials/accepted = " << std::setprecision(3)
              << (double)h_trials / (double)h_accepted
              << "\n";

    std::cout << "[P=" << std::fixed << std::setprecision(1) << P << "] "
              << "kernel-only time = " << std::setprecision(3) << kernelSecondsTotal << " s, "
              << "rate = " << std::setprecision(1)
              << (double)h_accepted / kernelSecondsTotal << " accepted/s\n";

    // Reduce the per-block histograms into one final histogram.
    unsigned int* d_final = nullptr;
    CUDA_CHECK(cudaMalloc(&d_final, (size_t)histSize * sizeof(unsigned int)));

    {
        int tpb = 256;
        int bpg = (histSize + tpb - 1) / tpb;

        reduce_block_histograms<<<bpg, tpb>>>(d_blockHist, d_final, histSize, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy back the final histogram.
    std::vector<unsigned int> h_hist(histSize, 0U);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_final,
                          (size_t)histSize * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // Build the ROOT histogram.
    TH2F hist(hname.c_str(),
              "Monte Carlo (GPU)",
              numberOfBinsX, xMinPlot, xMaxPlot,
              numberOfBinsCosTheta, cosThetaMin, cosThetaMax);

    for (int iy0 = 0; iy0 < numberOfBinsCosTheta; ++iy0) {
        for (int ix0 = 0; ix0 < numberOfBinsX; ++ix0) {
            int idx = ix0 + numberOfBinsX * iy0;
            hist.SetBinContent(ix0 + 1, iy0 + 1, (double)h_hist[idx]);
        }
    }

    // Axis labels and display range.
    hist.GetXaxis()->SetTitle("x = E_{e} / E_{max}");
    hist.GetYaxis()->SetTitle("cos#theta");
    hist.GetZaxis()->SetTitle("Events");
    hist.GetXaxis()->CenterTitle();
    hist.GetYaxis()->CenterTitle();
    hist.GetZaxis()->CenterTitle();
    hist.GetXaxis()->SetTitleOffset(1.2);
    hist.GetYaxis()->SetTitleOffset(1.3);
    hist.GetZaxis()->SetTitleOffset(1.4);
    hist.GetYaxis()->SetRangeUser(-1.0, 1.0);
    hist.GetXaxis()->SetRangeUser(0.0, 1.1);

    // Save ROOT output and PDF snapshot.
    TFile f(rootFile.c_str(), "RECREATE");
    hist.Write();

    TCanvas c("c", "GPU Monte Carlo", 900, 700);
    hist.SetStats(false);
    hist.Draw("COLZ");

    TLatex latex;
    latex.SetNDC(true);
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.15, 0.92, Form("Muon polarisation P = %.1f", P));

    c.SaveAs(pdfFile.c_str());

    // Free CUDA resources.
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_final);
    cudaFree(d_blockHist);
    cudaFree(d_states);
    cudaFree(d_rate);
    cudaFree(d_accept);
    cudaFree(d_trials);
    cudaFree(d_accepted);
}

//==============================================================
// Scan all polarization values
//==============================================================
void ScanPolarization_GPU(long long N, const double* par, const std::string& outDir)
{
    gSystem->mkdir(outDir.c_str(), true);

    for (int i = 0; i <= 20; ++i) {
        double P = -1.0 + 0.1 * i;
        RunOneP_GPU(N, par, P, outDir);
    }
}

//==============================================================
// Main
//==============================================================
int main()
{
    gROOT->SetBatch(kTRUE);

    // Copy constant values to GPU constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMin, &cosThetaMin, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMax, &cosThetaMax, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMinPlot,    &xMinPlot,    sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMaxPlot,    &xMaxPlot,    sizeof(double)));
    float cosThetaMin_f = (float)cosThetaMin;
    float cosThetaMax_f = (float)cosThetaMax;
    float xMinPlot_f    = (float)xMinPlot;
    float xMaxPlot_f    = (float)xMaxPlot;
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMin_f, &cosThetaMin_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMax_f, &cosThetaMax_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMinPlot_f,    &xMinPlot_f,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMaxPlot_f,    &xMaxPlot_f,    sizeof(float)));

    const long long Naccepted =1000000000LL; // 1e9

    double parSM[4] = {0.75, 0.0, 1.0, 0.75};

    std::string outDir = "/home/tom/Mu3e/Photos/MC Muon photos/MC GPU event";
    ScanPolarization_GPU(Naccepted, parSM, outDir);
    return 0;
}
