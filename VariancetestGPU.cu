//====================================================
// VariancetestGPU.cu
// - Fixed-N accepted events (GPU) with 3-layer acceptance
// - Fixed polarization P for all runs
// - Repeat R times with different seeds
// - Output per-run 2D histograms (ROOT+PDF), matching 3LayerMonteCarlo style
//====================================================

// ============================================================
// IMPORTANT bits:
// - generate_histogram_chunk(...)  -> event generation + acceptance
// - michel_device(...) + hit_at_radius_device(...)
// - RunOne_GPU(...) core counters and fixed-N logic
//
// === IGNORE PART START ===
// - pdf/canvas formatting
// - text summary files / pretty output blocks
// these were cleaned up with some ai help just for readability.
// === IGNORE PART END ===
// ============================================================

// ROOT
#include "TCanvas.h"
#include "TFile.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TROOT.h"

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

// C++
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

//==============================================================
// Parameters
//==============================================================
constexpr int numberOfBinsX        = 100;
constexpr int numberOfBinsCosTheta = 100;

// theta range
static const double thetaMinPlot = M_PI/2 - 1.3;
static const double thetaMaxPlot = M_PI/2 + 1.3;

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
// 3-layer acceptance (conf18-style SHAPE toy model)
//==============================================================
constexpr float B_T_f   = 1.0f;

constexpr float r1_mm_f = 23.3f;
constexpr float r2_mm_f = 29.8f;
constexpr float r3_mm_f = 73.9f;

constexpr float L1_mm_f = 124.7f;
constexpr float L2_mm_f = 124.7f;
constexpr float L3_mm_f = 351.9f;

constexpr float Z0_HALF_RANGE_MM_f = 5.0f;
constexpr float M_E_MEV_f = 0.51099895f;

//==============================================================
// Constant memory for gpu to read
//==============================================================
__device__ __constant__ double d_cosThetaMin;
__device__ __constant__ double d_cosThetaMax;
__device__ __constant__ double d_xMinPlot;
__device__ __constant__ double d_xMaxPlot;
__device__ __constant__ float d_cosThetaMin_f;
__device__ __constant__ float d_cosThetaMax_f;
__device__ __constant__ float d_xMinPlot_f;
__device__ __constant__ float d_xMaxPlot_f;

//=================================================================
// Reduce per block histogram
//=================================================================
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
// CPU helper: simple progress bar
//==============================================================
// === IGNORE PART START ===
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
// === IGNORE PART END ===

//==============================================================
// Device helpers that makes my life eaiser
//==============================================================
__device__ __forceinline__ double clampDouble(double x, double lo, double hi)
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

__device__ __forceinline__ int valueToBin_f(float v, float vmin, float vmax, int nbins)
{
    if (v < vmin || v >= vmax) return -1;
    float f = (v - vmin) / (vmax - vmin);
    int bin0 = (int)(f * nbins);
    return bin0 + 1; // ROOT-style 1..nbins
}

//==============================================================
// Physics on GPU
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

//==============================================================
//the helix part for 3 layer acceptance testing code
//==============================================================
__device__ __forceinline__ bool hit_at_radius_device(float pT_MeV, float pz_MeV,
                                                     float r_mm, float zHalf_mm,
                                                     float z0_mm, bool recurl)
{
    float pT_GeV = pT_MeV * 1e-3f;
    float rho_m  = pT_GeV / (0.3f * B_T_f);
    if (rho_m <= 0.0f) return false;

    float r_m = r_mm * 1e-3f;
    if (r_m > 2.0f * rho_m) return false;

    float arg = r_m / (2.0f * rho_m);
    arg = fminf(1.0f, fmaxf(0.0f, arg));

    float dphi1 = 2.0f * asinf(arg);
    //if recurl= true check if the rest of the angle distance traved
    float dphi  = recurl ? (2.0f * (float)M_PI - dphi1) : dphi1;

    float z_m  = rho_m * dphi * (pz_MeV / pT_MeV);
    float z_mm = z_m * 1e3f;

    return fabsf(z0_mm + z_mm) <= zHalf_mm;
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
    // curand_init sets up one RNG state per thread (same seed, different sequence id).
    curand_init(seed, tid, 0ULL, &states[tid]);
}

//==============================================================
// Generator kernel
// - Now with pre-block histogram
// - each block own one histogram region in global memory
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
    int nTrialsPerThread,
    double par_rho,
    double par_eta,
    double par_epsilon,
    double par_delta,
    double P
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

        // stop check every 64 bit
        if ((t & 63) == 0) {
            // atomicAdd(..., 0) is a safe atomic read of the shared accepted counter.
            stopSeen = atomicAdd(acceptedCount, 0ULL);
            if (stopSeen >= (unsigned long long)N) break;
        }

        // a different way for generating 4 number uniformlly
        // curand_uniform4 gives four uniform random numbers in one call.
        float4 u4 = curand_uniform4(&local);

        float randomX = d_xMinPlot_f + (d_xMaxPlot_f - d_xMinPlot_f) * u4.x;
        if (randomX >= d_xMaxPlot_f) randomX = nextafterf(d_xMaxPlot_f, d_xMinPlot_f);
        const float randomCosTheta =
            d_cosThetaMin_f + (d_cosThetaMax_f - d_cosThetaMin_f) * u4.y;
        //compute the michel rate
        double rate = michel_device((double)randomX, (double)randomCosTheta,
                            par_rho, par_eta, par_epsilon, par_delta, P);

        if (rate <= 0.0) continue;
        //rejection sampling
        double r = (double)maxAllowedRate * (double)u4.z;
        if (r >= rate) continue;

        //==========================================
        //smear x first, then evaluate event-by-event acceptance
        //===========================================
        float sigma = 0.0512f * randomX - 0.0039f;
        if (sigma < 0.0f) sigma = 0.0f;

        float xSmeared = randomX;
        if (sigma > 0.0f) {
            float2 n2 = curand_normal2(&local);
            float trial = randomX + sigma * n2.x;
            if (trial >= d_xMinPlot_f && trial < d_xMaxPlot_f) {
                xSmeared = trial;
            }
        }

        const float zH1 = 0.5f * L1_mm_f;
        const float zH2 = 0.5f * L2_mm_f;
        const float zH3 = 0.5f * L3_mm_f;
        //genrate random  z0
        const float z0_mm = (2.0f * u4.w - 1.0f) * Z0_HALF_RANGE_MM_f;
        //conver x back to p
        const float E = xSmeared * (float)E_MAX;
        //cant be lower than the electron mass energy
        if (E <= M_E_MEV_f) continue;
        //get the momentum
        const float p = sqrtf(E*E - M_E_MEV_f*M_E_MEV_f); // MeV/c
        //direction
        const float u = fminf(1.0f, fmaxf(-1.0f, randomCosTheta));
        //sin = sqrt(1-cos^2)
        const float sinT = sqrtf(fmaxf(0.0f, 1.0f - u*u));
        //get the z direction momentum for the decayed positon to check if it will be within the length
        const float pz = p * u;
        // momentum in x-y plane to check if raidius is reached
        const float pT = p * sinT;
        if (pT <= 1e-9f) continue;
        // outgoing hits at r1,r2,r3
        const bool o1 = hit_at_radius_device(pT, pz, r1_mm_f, zH1, z0_mm, false);
        const bool o2 = hit_at_radius_device(pT, pz, r2_mm_f, zH2, z0_mm, false);
        const bool o3 = hit_at_radius_device(pT, pz, r3_mm_f, zH3, z0_mm, false);
        //for recurl it has to at least hit 1 back
        const bool r2 = hit_at_radius_device(pT, pz, r3_mm_f, zH3, z0_mm, true);
        //if all three isnt achieved pass it 

        if (!(o1 && o2 && o3 && r2)) continue;


        // fill bins (smeared x, original cosTheta)
        const int fillBinX = valueToBin_f(xSmeared, d_xMinPlot_f, d_xMaxPlot_f, numberOfBinsX);
        const int fillBinY = valueToBin_f(randomCosTheta, d_cosThetaMin_f, d_cosThetaMax_f, numberOfBinsCosTheta);
        if (fillBinX < 1 || fillBinY < 1) continue;

        // reserve accepted slot
        const unsigned long long idx = atomicAdd(acceptedCount, 1ULL);
        if (idx >= (unsigned long long)N) break;

        const int histIndex = (fillBinX - 1) + numberOfBinsX * (fillBinY - 1);

        // STRIPED atomic add
        atomicAdd(&myHist[histIndex], 1U);
    }

    // accumulate trials
    atomicAdd(trialsCount, localTrials);

    // store rng state
    states[tid] = local;
}

//==============================================================
// CPU: find maximum rate for rejection sampling envelope
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
// output helpers (not the physics core)
//==============================================================
// === IGNORE PART START ===
static std::string FormatPTag(double P)
{
    std::ostringstream tag;
    tag << std::fixed << std::setprecision(1) << P;
    std::string s = tag.str();
    for (char& c : s) if (c == '.') c = 'p';
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

struct RunResult {
    double avgAbsCos = 0.0;
    double kernelSeconds = 0.0;
    unsigned long long accepted = 0ULL;
    unsigned long long trials = 0ULL;
    double maxLambda = std::numeric_limits<double>::quiet_NaN();
    bool foundLambda = false;
};

static unsigned long long splitmix64(unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static bool EnsureDir(const std::string& path)
{
    if (path.empty()) return false;
    std::string cur;
    cur.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        cur.push_back(c);
        if (c == '/' || i + 1 == path.size()) {
            if (cur.size() == 1 && cur[0] == '/') continue;
            if (mkdir(cur.c_str(), 0755) != 0) {
                if (errno == EEXIST) continue;
                return false;
            }
        }
    }
    return true;
}
// === IGNORE PART END ===

//==============================================================
// Run one fixed-P run (core workflow)
//==============================================================
static RunResult RunOne_GPU(long long N,
                            const double* par,
                            double P,
                            unsigned long long seed,
                            bool showProgress,
                            const std::string& outDir,
                            int runIndex)
{
    RunResult result;

    // === IGNORE PART START ===
    const std::string ptag = FormatPTag(P);
    std::ostringstream runTagSS;
    runTagSS << std::setfill('0') << std::setw(4) << runIndex;
    const std::string runTag = runTagSS.str();

    const std::string rootFile = outDir + "/MC_P_" + ptag + "_run_" + runTag + ".root";
    const std::string pdfFile = outDir + "/MC_P_" + ptag + "_run_" + runTag + ".pdf";
    const std::string hname = "HistogramRes_P_" + ptag + "_run_" + runTag;
    // === IGNORE PART END ===

    const double maxAllowedRate = findMaximumRate_host(par, P);
    const float maxAllowedRate_f = (float)maxAllowedRate;

    const int histSize = numberOfBinsX * numberOfBinsCosTheta;
    std::vector<unsigned char> h_accept(histSize, 1);
    std::vector<float> h_rate(histSize, 1.0f);

    unsigned char* d_accept = nullptr;
    // cudaMalloc allocates raw memory on GPU (device memory).
    CUDA_CHECK(cudaMalloc(&d_accept, (size_t)histSize * sizeof(unsigned char)));
    // cudaMemcpy HostToDevice copies CPU arrays into GPU memory.
    CUDA_CHECK(cudaMemcpy(d_accept, h_accept.data(),
                          (size_t)histSize * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    float* d_rate = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rate, (size_t)histSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rate, h_rate.data(),
                          (size_t)histSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    unsigned long long* d_accepted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_accepted, sizeof(unsigned long long)));
    // cudaMemset sets GPU to zero bytes.
    CUDA_CHECK(cudaMemset(d_accepted, 0, sizeof(unsigned long long)));

    unsigned long long* d_trials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_trials, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_trials, 0, sizeof(unsigned long long)));

    const int threadsPerBlock = 256;
    const int blocks = 1360;
    const int nStates = blocks * threadsPerBlock;
    const size_t blockHistBytes =
        (size_t)blocks * (size_t)histSize * sizeof(unsigned int);

    unsigned int* d_blockHist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockHist, blockHistBytes));
    CUDA_CHECK(cudaMemset(d_blockHist, 0, blockHistBytes));

    curandStatePhilox4_32_10_t* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, (size_t)nStates * sizeof(curandStatePhilox4_32_10_t)));

    // Kernel launch syntax <<<blocks, threads>>> runs init_rng on the GPU.
    init_rng<<<blocks, threadsPerBlock>>>(d_states, nStates, seed);
    // cudaGetLastError checks if the kernel launch itself failed.
    CUDA_CHECK(cudaGetLastError());
    // cudaDeviceSynchronize waits until the device finished queued work.
    CUDA_CHECK(cudaDeviceSynchronize());

    const int nTrialsPerThread = 4096;
    const int reportEvery = 50;

    unsigned long long h_accepted = 0ULL;
    unsigned long long h_trials = 0ULL;

    cudaEvent_t ev0, ev1;
    // cudaEventCreate creates GPU-side timing markers.
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    auto t0 = std::chrono::steady_clock::now();

    // cudaEventRecord stamps the event in the current stream (timer start).
    CUDA_CHECK(cudaEventRecord(ev0));
    int launchCount = 0;
    while (true) {
        generate_histogram_chunk<<<blocks, threadsPerBlock>>>(
            d_blockHist, d_accepted, d_trials, N, d_states, nStates,
            d_accept, d_rate, maxAllowedRate_f,
            nTrialsPerThread,
            par[0], par[1], par[2], par[3], P
        );
        CUDA_CHECK(cudaGetLastError());
        ++launchCount;

        if (launchCount % reportEvery == 0) {
            CUDA_CHECK(cudaMemcpy(&h_accepted, d_accepted,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            if (showProgress) {
                auto t1 = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(t1 - t0).count();
                PrintProgress(h_accepted, (unsigned long long)N, elapsed);
            }
            if (h_accepted >= (unsigned long long)N) break;
        }
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    // cudaEventSynchronize blocks host until the stop event is completed.
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    // cudaEventElapsedTime measures GPU time between the two events in ms.
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    result.kernelSeconds = 1e-3 * ms;

    // cudaMemcpy DeviceToHost copies counters back from GPU to CPU.
    CUDA_CHECK(cudaMemcpy(&h_accepted, d_accepted,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_trials, d_trials,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    result.accepted = h_accepted;
    result.trials = h_trials;

    unsigned int* d_final = nullptr;
    CUDA_CHECK(cudaMalloc(&d_final, (size_t)histSize * sizeof(unsigned int)));
    {
        const int tpb = 256;
        const int bpg = (histSize + tpb - 1) / tpb;
        reduce_block_histograms<<<bpg, tpb>>>(d_blockHist, d_final, histSize, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<unsigned int> h_hist(histSize, 0U);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_final,
                          (size_t)histSize * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));



    // === IGNORE PART START ===
    // plotting/output block below is not the physics core.

    TH2F hist(hname.c_str(),
              "Monte Carlo (GPU, striped) after acceptance",
              numberOfBinsX, xMinPlot, xMaxPlot,
              numberOfBinsCosTheta, cosThetaMin, cosThetaMax);

    for (int iy0 = 0; iy0 < numberOfBinsCosTheta; ++iy0) {
        for (int ix0 = 0; ix0 < numberOfBinsX; ++ix0) {
            const int idx = ix0 + numberOfBinsX * iy0;
            hist.SetBinContent(ix0 + 1, iy0 + 1, (double)h_hist[idx]);
        }
    }

    double maxLambda = -M_PI / 2.0;
    bool foundLambda = false;
    if (h_accepted > 0ULL) {
        for (int iy = numberOfBinsCosTheta; iy >= 1; --iy) {
            double rowSum = 0.0;
            for (int ix = 1; ix <= numberOfBinsX; ++ix) {
                rowSum += hist.GetBinContent(ix, iy);
            }
            if (rowSum > 0.0) {
                const double cosC = hist.GetYaxis()->GetBinCenter(iy);
                const double theta = std::acos(std::min(1.0, std::max(-1.0, cosC)));
                maxLambda = (M_PI / 2.0) - theta;
                foundLambda = true;
                break;
            }
        }
    }
    result.maxLambda = maxLambda;
    result.foundLambda = foundLambda;

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

    // output ROOT hist (this one is useful if someone wants to re-analyse later)
    TFile f(rootFile.c_str(), "RECREATE");
    hist.Write();
    f.Close();

    // output PDF is only quick visual check for each run
    TCanvas c("c", "GPU Monte Carlo", 900, 700);
    hist.SetStats(false);
    hist.Draw("COLZ");
    TLatex latex;
    latex.SetNDC(true);
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.15, 0.92, Form("Muon polarisation P = %.1f", P));
    latex.DrawLatex(0.15, 0.87, Form("Run %d", runIndex));
    c.SaveAs(pdfFile.c_str());
    // === IGNORE PART END ===

    // cudaEventDestroy releases event handles created for timing.
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    // cudaFree releases GPU allocations made with cudaMalloc.
    cudaFree(d_final);
    cudaFree(d_blockHist);
    cudaFree(d_states);
    cudaFree(d_rate);
    cudaFree(d_accept);
    cudaFree(d_trials);
    cudaFree(d_accepted);
    return result;
}

//==============================================================
// Main
//==============================================================
int main(int argc, char** argv)
{
    gROOT->SetBatch(kTRUE);

    long long Naccepted = 54192485; // 2e8
    int repeats = 500;
    double Ptrue = 1.0;
    std::string outDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/Variancce test";
    unsigned long long seedBase = 0ULL;
    bool seedProvided = false;
    bool showProgress = true;
   // === IGNORE PART START ===
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--n" || a == "--nmu" || a == "--N") && i + 1 < argc) {
            Naccepted = std::stoll(argv[++i]);
        } else if ((a == "--r" || a == "--repeats") && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if ((a == "--p" || a == "--P") && i + 1 < argc) {
            Ptrue = std::stod(argv[++i]);
        } else if ((a == "--out" || a == "--outdir") && i + 1 < argc) {
            outDir = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            seedBase = std::stoull(argv[++i]);
            seedProvided = true;
        } else if (a == "--no-progress") {
            showProgress = false;
        } else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [--n <Naccepted>] [--r <repeats>] [--p <P>]\n"
                << "       [--out <outDir>] [--seed <seed>] [--no-progress]\n"
                << "  --n           number of accepted events per run (default 2e8)\n"
                << "  --r           number of runs with different seeds (default 50)\n"
                << "  --p           fixed polarization P for all runs (default 0.5)\n"
                << "  --out         output directory for ROOT/PDF and run CSV\n"
                << "  --seed        base RNG seed (default random)\n"
                << "  --no-progress disable per-run progress bar\n";
            return 0;
        }
    }
    // === IGNORE PART END === 
    //random seed generation 
    if (!seedProvided) {
        std::random_device rd;
        seedBase =
            (static_cast<unsigned long long>(rd()) << 32) ^
            static_cast<unsigned long long>(rd()) ^
            static_cast<unsigned long long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    // cudaMemcpyToSymbol copies host constants into GPU constant memory symbols.
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMin, &cosThetaMin, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMax, &cosThetaMax, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMinPlot, &xMinPlot, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMaxPlot, &xMaxPlot, sizeof(double)));
    float cosThetaMin_f = (float)cosThetaMin;
    float cosThetaMax_f = (float)cosThetaMax;
    float xMinPlot_f = (float)xMinPlot;
    float xMaxPlot_f = (float)xMaxPlot;
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMin_f, &cosThetaMin_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMax_f, &cosThetaMax_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMinPlot_f, &xMinPlot_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMaxPlot_f, &xMaxPlot_f, sizeof(float)));

    const double parSM[4] = {0.75, 0.0, 1.0, 0.75};

    if (!EnsureDir(outDir)) {
        std::cerr << "ERROR: cannot create output directory: " << outDir << "\n";
        return 1;
    }

    // === IGNORE PART START ===
    // csv + txt below are summary/output helpers 
    const std::string runsCsv = outDir + "/mc_runs.csv";
    std::ofstream runs(runsCsv);
    if (!runs) {
        std::cerr << "ERROR: cannot write file: " << runsCsv << "\n";
        return 1;
    }
    runs << "run,seed,P,accepted,trials,trials_per_accepted,kernel_s,avg_abs_cos,max_lambda\n";

    const std::string maxLambdaSummary = outDir + "/max_lambda_summary.txt";
    {
        std::ofstream ofs(maxLambdaSummary);
        if (ofs) ofs << "run,seed,P,max_lambda\n";
    }
    // === IGNORE PART END ===

    std::cout << "Naccepted = " << Naccepted << "\n";
    std::cout << "repeats   = " << repeats << "\n";
    std::cout << "Pfixed    = " << Ptrue << "\n";
    std::cout << "outDir    = " << outDir << "\n";
    std::cout << "seedBase  = " << seedBase << (seedProvided ? " (user)" : " (random)") << "\n";

    for (int r = 0; r < repeats; ++r) {
        const unsigned long long seed = splitmix64(seedBase + (unsigned long long)r);
        std::cout << "[Run " << (r + 1) << "/" << repeats << "] seed=" << seed << "\n";

        RunResult res = RunOne_GPU(Naccepted, parSM, Ptrue, seed, showProgress, outDir, r);
        const double tpa = (res.accepted > 0ULL)
            ? ((double)res.trials / (double)res.accepted)
            : std::numeric_limits<double>::quiet_NaN();

        std::cout << "[Run " << (r + 1) << "] "
                  << "accepted=" << res.accepted
                  << ", trials=" << res.trials
                  << ", trials/accepted=" << std::setprecision(3) << tpa
                  << ", kernel_s=" << std::fixed << std::setprecision(3) << res.kernelSeconds
                  << ", max_lambda=";
        if (res.foundLambda) {
            std::cout << std::setprecision(6) << res.maxLambda;
        } else {
            std::cout << "NaN";
        }
        std::cout << "\n";

        runs << r << ","
             << seed << ","
             << std::fixed << std::setprecision(6) << Ptrue << ","
             << res.accepted << ","
             << res.trials << ","
             << tpa << ","
             << res.kernelSeconds << ","
             << res.avgAbsCos << ",";
        if (res.foundLambda) {
            runs << std::setprecision(8) << res.maxLambda << "\n";
        } else {
            runs << "NaN\n";
        }

        std::ofstream ofs(maxLambdaSummary, std::ios::app);
        if (ofs) {
            ofs << r << ","
                << seed << ","
                << std::fixed << std::setprecision(6) << Ptrue << ",";
            if (res.foundLambda) {
                ofs << std::setprecision(8) << res.maxLambda << "\n";
            } else {
                ofs << "NaN\n";
            }
        }
    }

    runs.close();
    std::cout << "Wrote:\n";
    std::cout << "  " << runsCsv << "\n";
    std::cout << "  " << maxLambdaSummary << "\n";
    return 0;
}
