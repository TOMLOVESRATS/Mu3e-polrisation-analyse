//====================================================
// Mu3e 3-layer barrel acceptance (conf18-style SHAPE, toy model)
// - Acceptance vs (pTrue, lambda)
// - GPU Michel generation + isotropic direction
// - Simple 3-layer helix reach + barrel half-length (|z|) cut
//
// OUTPUT:
//   (1) hGen(pTrue,lambda)
//   (2) hPass(pTrue,lambda)
//   (3) hAcc = hPass/hGen
//====================================================

// ROOT includes
#include "TFile.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TSystem.h"

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

// C++
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>

//==============================================================
// Parameters
//==============================================================
constexpr double B_T = 1.0;                 // Tesla (Mu3e solenoid)  :contentReference[oaicite:5]{index=5}
constexpr double M_MU = 105.6583745;        // MeV
constexpr double M_E_MEV = 0.51099895;      // MeV (avoid clash with math.h M_E)

// 3 barrel layers: radii and instrumented lengths from Mu3e TDR table 7.1 :contentReference[oaicite:6]{index=6}
constexpr double r1_mm = 23.3;
constexpr double r2_mm = 29.8;
constexpr double r3_mm = 73.9;

constexpr double L1_mm = 124.7;
constexpr double L2_mm = 124.7;
constexpr double L3_mm = 351.9;

// histogram binning
constexpr int    NBIN_P   = 160;
constexpr double P_MIN    = 0.0;     // MeV/c
constexpr double P_MAX    = 60.0;    // MeV/c (covers Michel endpoint ~52.8)

constexpr int    NBIN_LAM = 160;
constexpr double LAM_MIN  = -1.55;   // rad (roughly -pi/2..pi/2)
constexpr double LAM_MAX  = +1.55;   // rad
constexpr double Z0_HALF_RANGE_MM = 5.0;  // uniform z0 in [-5,+5] mm (safe default)

// GPU launch
constexpr int THREADS = 256;
//==============================================================
// Simple CUDA error check
//==============================================================
static inline void CudaCheck(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << where << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

//==============================================================
// Device helpers
//==============================================================
__device__ inline double clampd(double x, double a, double b) {
    return (x < a) ? a : (x > b) ? b : x;
}

__device__ inline int bin1D(double x, double xmin, double xmax, int nb) {
    if (x < xmin || x >= xmax) return -1;
    double u = (x - xmin) / (xmax - xmin);
    int ib = (int)floor(u * nb);
    if (ib < 0) ib = 0;
    if (ib >= nb) ib = nb - 1;
    return ib;
}

// Michel spectrum f(x) ∝ x^2 (3 - 2x) on x in [0,1]
__device__ inline double michel_pdf(double x) {
    return x*x*(3.0 - 2.0*x);
}

// Sample Michel energy (massless shape) by rejection; returns E in MeV
__device__ inline double sample_michel_E(curandStatePhilox4_32_10_t& st) {
    // Max of x^2(3-2x) on [0,1] is 1 at x=1
    while (true) {
        double x = curand_uniform_double(&st);      // (0,1]
        double y = curand_uniform_double(&st);      // (0,1]
        if (y <= michel_pdf(x)) {
            return 0.5 * M_MU * x;                 // E = x * m_mu/2
        }
    }
}

// Compute whether a helix can hit a given layer (radius r_mm, half-length zHalf_mm)
__device__ inline bool hit_at_radius(double pT_MeV, double pz_MeV,
                                     double r_mm, double zHalf_mm,
                                     double z0_mm, bool recurl) {
    double pT_GeV = pT_MeV * 1e-3;
    double rho_m  = pT_GeV / (0.3 * B_T);
    if (rho_m <= 0.0) return false;

    double r_m = r_mm * 1e-3;
    if (r_m > 2.0 * rho_m) return false;

    double arg  = r_m / (2.0 * rho_m);
    arg = clampd(arg, 0.0, 1.0);

    double dphi1 = 2.0 * asin(arg);
    //if we need recurl it would get 2pi-phi1 
    double dphi  = recurl ? (2.0 * M_PI - dphi1) : dphi1;

    double z_m  = rho_m * dphi * (pz_MeV / pT_MeV);
    double z_mm = z_m * 1e3;

    return fabs(z0_mm + z_mm) <= zHalf_mm;
}


//==============================================================
// Kernel: fill 2D counters in (pTrue, lambda)
//==============================================================
__global__ void acceptanceKernel(uint64_t N,
                                 uint32_t* gen2D, uint32_t* pass2D,
                                 unsigned long long seed) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)tid, 0ULL, &st);



    for (uint64_t i = tid; i < N; i += stride) {

        // --- Michel energy
        double E = sample_michel_E(st); // MeV

        // include electron mass for momentum
        if (E <= M_E_MEV) continue;
        double p = sqrt(E*E - M_E_MEV*M_E_MEV); // MeV/c

        // --- isotropic direction
        double u = 2.0 * curand_uniform_double(&st) - 1.0; // cos(theta) in [-1,1]
        double sinT = sqrt(fmax(0.0, 1.0 - u*u));

        double pz = p * u;
        double pT = p * sinT;

        // lambda = atan2(pz, pT)  (inclination from x-y plane) :contentReference[oaicite:7]{index=7}
        double lam = atan2(pz, pT);

        // histogram bin
        int ibP   = bin1D(p,   P_MIN,   P_MAX,   NBIN_P);
        int ibLam = bin1D(lam, LAM_MIN, LAM_MAX, NBIN_LAM);
        if (ibP < 0 || ibLam < 0) continue;

        int idx = ibP + NBIN_P * ibLam;

        // gen
        atomicAdd(&gen2D[idx], 1U);

        // use overlap length (short barrel)
        const float zH1 = 0.5f * L1_mm_f;
        const float zH2 = 0.5f * L2_mm_f;
        const float zH3 = 0.5f * L3_mm_f;

        // z0 smearing
        double z0_mm = (2.0 * curand_uniform_double(&st) - 1.0) * Z0_HALF_RANGE_MM;

        // outgoing 3 hits
        bool o1 = hit_at_radius(pT, pz, r1_mm, zHeff, z0_mm, false);
        bool o2 = hit_at_radius(pT, pz, r2_mm, zHeff, z0_mm, false);
        bool o3 = hit_at_radius(pT, pz, r3_mm, zHeff, z0_mm, false);

        // recurl hit proxy (at least one inner layer again)
        bool r2 = hit_at_radius(pT, pz, r2_mm, zHeff, z0_mm, true);


        if (o1 && o2 && o3 && r2) {
            atomicAdd(&pass2D[idx], 1U);
        }

    }
}

//==============================================================
// Main
//==============================================================
int main(int argc, char** argv) {

    // N events (default)
    uint64_t N = 20000000ULL; // 2e7 default
    if (argc >= 2) {
        N = std::stoull(argv[1]);
    }

    std::cout << "Running 3-layer acceptance GPU MC with N = " << N << "\n";

    // allocate device hist arrays
    size_t nCells = (size_t)NBIN_P * (size_t)NBIN_LAM;
    size_t bytes  = nCells * sizeof(uint32_t);

    uint32_t* d_gen = nullptr;
    uint32_t* d_pass = nullptr;
    CudaCheck(cudaMalloc(&d_gen, bytes),  "cudaMalloc d_gen");
    CudaCheck(cudaMalloc(&d_pass, bytes), "cudaMalloc d_pass");
    CudaCheck(cudaMemset(d_gen, 0, bytes),  "cudaMemset d_gen");
    CudaCheck(cudaMemset(d_pass, 0, bytes), "cudaMemset d_pass");

    // launch config
    int device = 0;
    cudaDeviceProp prop{};
    CudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    CudaCheck(cudaSetDevice(device), "cudaSetDevice");

    int blocks = (prop.multiProcessorCount * 8);
    std::cout << "GPU: " << prop.name << "  SMs=" << prop.multiProcessorCount
              << "  blocks=" << blocks << "  threads=" << THREADS << "\n";

    unsigned long long seed = 123456789ULL;

    acceptanceKernel<<<blocks, THREADS>>>(N, d_gen, d_pass, seed);
    CudaCheck(cudaGetLastError(), "kernel launch");
    CudaCheck(cudaDeviceSynchronize(), "kernel sync");

    // copy back
    std::vector<uint32_t> h_gen(nCells), h_pass(nCells);
    CudaCheck(cudaMemcpy(h_gen.data(), d_gen, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy gen");
    CudaCheck(cudaMemcpy(h_pass.data(), d_pass, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy pass");

    CudaCheck(cudaFree(d_gen), "cudaFree gen");
    CudaCheck(cudaFree(d_pass), "cudaFree pass");

    // ROOT histograms
    TH2F* hGen  = new TH2F("hGen",  "Generated; p_{true} [MeV/c]; #lambda [rad]",
                          NBIN_P, P_MIN, P_MAX, NBIN_LAM, LAM_MIN, LAM_MAX);
    TH2F* hPass = new TH2F("hPass", "Passed (3 layers); p_{true} [MeV/c]; #lambda [rad]",
                          NBIN_P, P_MIN, P_MAX, NBIN_LAM, LAM_MIN, LAM_MAX);

    // fill from flat arrays
    for (int j = 0; j < NBIN_LAM; ++j) {
        for (int i = 0; i < NBIN_P; ++i) {
            int idx = i + NBIN_P * j;
            hGen->SetBinContent(i+1, j+1, (double)h_gen[idx]);
            hPass->SetBinContent(i+1, j+1, (double)h_pass[idx]);
        }
    }

    // Acceptance

    // style + plots
    gStyle->SetOptStat(0);

    TCanvas* cGen = new TCanvas("cGen","Generated",800,700);
    hGen->Draw("COLZ");

    TCanvas* cPass = new TCanvas("cPass","Passed (3 layers)",800,700);
    hPass->Draw("COLZ");

    // save
    const char* outDir =
        "/home/tom/Mu3e/Photos/MC Muon photos/Acceptance_only";

    gSystem->mkdir(outDir, kTRUE);

    // ROOT file
    TFile out(Form("%s/Mu3e_3Layer_Acceptance_GPU.root", outDir),
            "RECREATE");

    hGen->Write();
    hPass->Write();
    cGen->Write();
    cPass->Write();

    out.Close();

    // PDF snapshots
    cGen->SaveAs(Form("%s/Mu3e_3Layer_Gen_GPU.pdf", outDir));
    cPass->SaveAs(Form("%s/Mu3e_3Layer_Pass_GPU.pdf", outDir));
    std::cout << "Wrote Mu3e_3Layer_Acceptance_GPU.root\n";
    return 0;
}
