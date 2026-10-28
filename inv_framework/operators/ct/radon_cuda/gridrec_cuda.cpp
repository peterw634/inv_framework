// PyTorch bindings and CPU lookup-table builders for the TomoPy Gridrec port
// in the nasn_ct.operators.radon_cuda package.
//
// Derived from TomoPy, Copyright (c) 2015 UChicago Argonne, LLC.
// The BSD-3-Clause terms are retained in LICENSES/TomoPy-BSD-3-Clause.txt.
// This file is a
// modified CUDA/PyTorch implementation, not an upstream TomoPy source file.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void launch_gridrec_gridding(
    const float* cproj,
    float* grid,
    const float* filphase,
    const float* sine,
    const float* cose,
    const float* wtbl,
    int batch_pairs,
    int n_ang,
    int pdim,
    int filter_rows,
    cudaStream_t stream);

#define CHECK_CPU(x) TORCH_CHECK(!(x).is_cuda(), #x " must be CPU")
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_COMPLEX64(x) TORCH_CHECK((x).scalar_type() == at::kComplexFloat, #x " must be complex64")

static float legendre_tomopy(int n, const float* coefs, float x) {
    float y = coefs[0];
    float penult = 1.0f;
    float last = x;
    for (int j = 2; j <= n; ++j) {
        float mxlast = -(x * last);
        float cur = -(2 * mxlast + penult) + (penult + mxlast) / j;
        if (!(j & 1)) {
            y += cur * coefs[j >> 1];
        }
        penult = last;
        last = cur;
    }
    return y;
}

std::vector<at::Tensor> prepare_pswf_tables(int64_t pdim) {
    TORCH_CHECK(pdim >= 16 && (pdim & (pdim - 1)) == 0,
                "pdim must be a power of two >= 16");

    constexpr float C = 7.0f;
    constexpr int nt = 20;
    constexpr float lambda = 0.99998546f;
    constexpr int ltbl = 512;
    const float coefs[11] = {
        0.5767616E+02f, -0.8931343E+02f, 0.4167596E+02f,
        -0.1053599E+02f, 0.1662374E+01f, -0.1780527E-00f,
        0.1372983E-01f, -0.7963169E-03f, 0.3593372E-04f,
        -0.1295941E-05f, 0.3817796E-07f
    };

    const int linv = static_cast<int>(pdim / 2) - 1;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    auto wtbl = at::empty({ltbl + 1}, options);
    auto winv = at::empty({pdim - 1}, options);
    float* wtbl_ptr = wtbl.data_ptr<float>();
    float* winv_ptr = winv.data_ptr<float>();

    const float polyz = legendre_tomopy(nt, coefs, 0.0f);
    wtbl_ptr[0] = 1.0f;
    for (int i = 1; i <= ltbl; ++i) {
        wtbl_ptr[i] = legendre_tomopy(
            nt, coefs, static_cast<float>(i) / ltbl) / polyz;
    }

    const float fac = static_cast<float>(ltbl) / (linv + 0.5f);
    float norm = static_cast<float>(
        std::sqrt(M_PI / 2.0 / C / lambda) / 1.2);
    winv_ptr[linv] = norm / wtbl_ptr[0];
    for (int i = 1; i <= linv; ++i) {
        norm = -norm;
        int table_index = static_cast<int>(std::roundf(i * fac));
        float value = norm / wtbl_ptr[table_index];
        winv_ptr[linv + i] = value;
        winv_ptr[linv - i] = value;
    }
    return {wtbl, winv};
}

std::vector<at::Tensor> prepare_trig_tables(at::Tensor theta) {
    CHECK_CPU(theta);
    CHECK_CONTIGUOUS(theta);
    CHECK_FLOAT(theta);
    TORCH_CHECK(theta.dim() == 1, "theta must have shape (n_ang,)");

    auto sine = at::empty_like(theta);
    auto cose = at::empty_like(theta);
    const float* theta_ptr = theta.data_ptr<float>();
    float* sine_ptr = sine.data_ptr<float>();
    float* cose_ptr = cose.data_ptr<float>();
    for (int64_t p = 0; p < theta.numel(); ++p) {
        sine_ptr[p] = ::sinf(theta_ptr[p]);
        cose_ptr[p] = ::cosf(theta_ptr[p]);
    }
    return {sine, cose};
}

static bool filter_is_2d(const std::string& name) {
    return name == "custom2d";
}

static float filter_value(
    const std::string& name,
    float x,
    int frequency,
    int angle,
    int filter_width,
    const float* pars) {
    if (name == "none") {
        return 1.0f;
    }
    if (name == "shepp" || name.empty()) {
        if (frequency == 0) return 0.0f;
        return std::fabs(2.0f * x)
            * (::sinf(static_cast<float>(M_PI) * x)
               / (static_cast<float>(M_PI) * x));
    }
    if (name == "cosine") {
        return std::fabs(2.0f * x)
            * ::cosf(static_cast<float>(M_PI) * x);
    }
    if (name == "hann") {
        return std::fabs(2.0f * x) * 0.5f
            * (1.0f + ::cosf(
                2.0f * static_cast<float>(M_PI) * x / pars[0]));
    }
    if (name == "hamming") {
        return std::fabs(2.0f * x)
            * (0.54f + 0.46f * ::cosf(
                2.0f * static_cast<float>(M_PI) * x / pars[0]));
    }
    if (name == "ramlak") {
        return std::fabs(2.0f * x);
    }
    if (name == "parzen") {
        return std::fabs(2.0f * x)
            * ::powf(1.0f - std::fabs(x) / pars[0], 3.0f);
    }
    if (name == "butterworth") {
        return std::fabs(2.0f * x)
            / (1.0f + ::powf(x / pars[0], 2.0f * pars[1]));
    }
    if (name == "custom") {
        return pars[frequency];
    }
    if (name == "custom2d") {
        return pars[angle * filter_width + frequency];
    }
    // TomoPy get_filter() falls back to Shepp for unknown names.
    if (frequency == 0) return 0.0f;
    return std::fabs(2.0f * x)
        * (::sinf(static_cast<float>(M_PI) * x)
           / (static_cast<float>(M_PI) * x));
}

at::Tensor prepare_filter_phase(
    at::Tensor pair_centers,
    int64_t n_ang,
    int64_t pdim,
    const std::string& filter_name,
    at::Tensor filter_par) {
    CHECK_CPU(pair_centers);
    CHECK_CPU(filter_par);
    CHECK_CONTIGUOUS(pair_centers);
    CHECK_CONTIGUOUS(filter_par);
    CHECK_FLOAT(pair_centers);
    CHECK_FLOAT(filter_par);
    TORCH_CHECK(pair_centers.dim() == 1,
                "pair_centers must have shape (batch_pairs,)");
    TORCH_CHECK(filter_par.dim() == 1, "filter_par must be flat");
    TORCH_CHECK(n_ang > 0 && pdim >= 16, "invalid n_ang or pdim");

    const int pdim2 = static_cast<int>(pdim / 2);
    const int rows = filter_is_2d(filter_name)
        ? static_cast<int>(n_ang) : 1;
    if (filter_name == "custom") {
        TORCH_CHECK(filter_par.numel() >= pdim2,
                    "custom filter requires pdim/2 values");
    }
    if (filter_name == "custom2d") {
        TORCH_CHECK(filter_par.numel() >= n_ang * pdim2,
                    "custom2d filter requires n_ang*pdim/2 values");
    }
    if (filter_name == "hann" || filter_name == "hamming"
        || filter_name == "parzen") {
        TORCH_CHECK(filter_par.numel() >= 1,
                    "selected filter requires filter_par[0]");
    }
    if (filter_name == "butterworth") {
        TORCH_CHECK(filter_par.numel() >= 2,
                    "butterworth requires two filter parameters");
    }

    auto output = at::empty(
        {pair_centers.numel(), rows, pdim2, 2},
        pair_centers.options());
    const float* centers = pair_centers.data_ptr<float>();
    const float* pars = filter_par.data_ptr<float>();
    float* out = output.data_ptr<float>();
    const float norm = static_cast<float>(M_PI / pdim / n_ang);

    for (int64_t b = 0; b < pair_centers.numel(); ++b) {
        const float phase_step = static_cast<float>(
            2.0 * M_PI * centers[b] / pdim);
        for (int angle_row = 0; angle_row < rows; ++angle_row) {
            int filter_angle = rows == 1 ? 0 : angle_row;
            for (int j = 0; j < pdim2; ++j) {
                const float x_frequency = static_cast<float>(j) / pdim;
                std::complex<float> value(filter_value(
                    filter_name,
                    x_frequency,
                    j,
                    filter_angle,
                    pdim2,
                    pars), 0.0f);
                const float phase = j * phase_step;
                value *= std::complex<float>(
                    ::cosf(phase), -::sinf(phase)) * norm;
                const int64_t index =
                    (((b * rows + angle_row) * pdim2 + j) * 2);
                out[index] = value.real();
                out[index + 1] = value.imag();
            }
        }
    }
    return output;
}

void gridding(
    at::Tensor cproj,
    at::Tensor grid,
    at::Tensor filphase,
    at::Tensor sine,
    at::Tensor cose,
    at::Tensor wtbl) {
    CHECK_CUDA(cproj); CHECK_CUDA(grid); CHECK_CUDA(filphase);
    CHECK_CUDA(sine); CHECK_CUDA(cose); CHECK_CUDA(wtbl);
    CHECK_CONTIGUOUS(cproj); CHECK_CONTIGUOUS(grid);
    CHECK_CONTIGUOUS(filphase); CHECK_CONTIGUOUS(sine);
    CHECK_CONTIGUOUS(cose); CHECK_CONTIGUOUS(wtbl);
    CHECK_COMPLEX64(cproj); CHECK_COMPLEX64(grid);
    CHECK_COMPLEX64(filphase);
    CHECK_FLOAT(sine); CHECK_FLOAT(cose); CHECK_FLOAT(wtbl);
    TORCH_CHECK(cproj.dim() == 3,
                "cproj must be (batch_pairs, n_ang, pdim)");
    TORCH_CHECK(grid.dim() == 3,
                "grid must be (batch_pairs, pdim, pdim)");
    TORCH_CHECK(filphase.dim() == 3,
                "filphase must be (batch_pairs, rows, pdim/2)");

    const int64_t batch_pairs = cproj.size(0);
    const int64_t n_ang = cproj.size(1);
    const int64_t pdim = cproj.size(2);
    TORCH_CHECK(grid.size(0) == batch_pairs
                && grid.size(1) == pdim && grid.size(2) == pdim,
                "grid shape mismatch");
    TORCH_CHECK(filphase.size(0) == batch_pairs
                && (filphase.size(1) == 1 || filphase.size(1) == n_ang)
                && filphase.size(2) == pdim / 2,
                "filphase shape mismatch");
    TORCH_CHECK(sine.numel() == n_ang && cose.numel() == n_ang,
                "angle table shape mismatch");
    TORCH_CHECK(wtbl.numel() == 513, "wtbl must contain 513 values");
    TORCH_CHECK(cproj.device() == grid.device()
                && cproj.device() == filphase.device()
                && cproj.device() == sine.device()
                && cproj.device() == cose.device()
                && cproj.device() == wtbl.device(),
                "all gridding tensors must share one CUDA device");

    c10::cuda::CUDAGuard guard(cproj.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(cproj.get_device());
    launch_gridrec_gridding(
        reinterpret_cast<const float*>(cproj.data_ptr<c10::complex<float>>()),
        reinterpret_cast<float*>(grid.data_ptr<c10::complex<float>>()),
        reinterpret_cast<const float*>(
            filphase.data_ptr<c10::complex<float>>()),
        sine.data_ptr<float>(),
        cose.data_ptr<float>(),
        wtbl.data_ptr<float>(),
        static_cast<int>(batch_pairs),
        static_cast<int>(n_ang),
        static_cast<int>(pdim),
        static_cast<int>(filphase.size(1)),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prepare_pswf_tables", &prepare_pswf_tables,
               py::arg("pdim"));
    module.def("prepare_trig_tables", &prepare_trig_tables,
               py::arg("theta"));
    module.def("prepare_filter_phase", &prepare_filter_phase,
               py::arg("pair_centers"), py::arg("n_ang"),
               py::arg("pdim"), py::arg("filter_name"),
               py::arg("filter_par"));
    module.def("gridding", &gridding,
               py::arg("cproj"), py::arg("grid"),
               py::arg("filphase"), py::arg("sine"),
               py::arg("cose"), py::arg("wtbl"));
}
