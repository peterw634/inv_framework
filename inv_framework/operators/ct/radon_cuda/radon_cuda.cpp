// PyTorch bindings for the CUDA Siddon projector / back-projector.
//
// prepare_geometry(theta_cpu):
//     evaluates fmodf/sinf/cosf/calc_quadrant on the CPU, matching TomoPy.
//
// project(img, sin_theta, cos_theta, quadrant, center):
//     img    : (B, ox, oz)  float32 cuda
//     angle tables: (dt,)    float32/int32 cuda
//     center : (B,)          float32 cuda
//     returns sino (B, dt, ox) float32 cuda
//
// bp(sino, sin_theta, cos_theta, quadrant, center, recon):
//     sino   : (B, dt, dx)  float32 cuda
//     recon  : (B, ngridx, ngridy) float32 cuda, updated in place
//
// Both match tomopy.project(..., emission=True, pad=False) and
// tomopy.recon(..., algorithm='fbp', filter_name='none', init_recon=zeros).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <vector>

void launch_project(
    const float* img, float* sino,
    const float* sin_theta, const float* cos_theta,
    const int32_t* quadrant, const float* center,
    int dy, int dt, int dx, int ox, int oz,
    cudaStream_t stream);

void launch_bp(
    const float* sino, float* recon,
    const float* sin_theta, const float* cos_theta,
    const int32_t* quadrant, const float* center,
    int dy, int dt, int dx, int ox, int oz,
    cudaStream_t stream);

void launch_bp_linear(
    const float* sino, float* recon,
    const float* sin_theta, const float* cos_theta,
    const float* center,
    float output_scale,
    int batch, int dt, int dx, int ngridx, int ngridy,
    cudaStream_t stream);

#define CHECK_CUDA(x)      TORCH_CHECK(x.is_cuda(),       #x " must be CUDA")
#define CHECK_CPU(x)       TORCH_CHECK(!x.is_cuda(),      #x " must be CPU")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x)     TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT32(x)     TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")
#define CHECK_SAME_DEVICE(x, y) \
    TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")

static int32_t calc_quadrant_host(float theta_p) {
    // Deliberately mirrors libtomo/recon/utils.c:calc_quadrant, including
    // float/double promotion around M_PI.
    const int32_t ipi_c = 340870420;
    int32_t theta_i = static_cast<int32_t>(theta_p * ipi_c);
    theta_i += (theta_i < 0)
        ? static_cast<int32_t>(2.0f * M_PI * ipi_c)
        : 0;
    return ((theta_i >= 0 && theta_i < 0.5f * M_PI * ipi_c) ||
            (theta_i >= 1.0f * M_PI * ipi_c &&
             theta_i < 1.5f * M_PI * ipi_c)) ? 1 : 0;
}

std::vector<at::Tensor> prepare_geometry(at::Tensor theta) {
    CHECK_CPU(theta);
    CHECK_CONTIGUOUS(theta);
    CHECK_FLOAT(theta);
    TORCH_CHECK(theta.dim() == 1, "theta must be (dt,)");

    auto sin_theta = at::empty_like(theta);
    auto cos_theta = at::empty_like(theta);
    auto quadrant = at::empty(theta.sizes(), theta.options().dtype(at::kInt));

    const float* theta_ptr = theta.data_ptr<float>();
    float* sin_ptr = sin_theta.data_ptr<float>();
    float* cos_ptr = cos_theta.data_ptr<float>();
    int32_t* quadrant_ptr = quadrant.data_ptr<int32_t>();
    const int64_t dt = theta.numel();

    for (int64_t p = 0; p < dt; ++p) {
        const float theta_p = fmodf(theta_ptr[p], 2.0f * static_cast<float>(M_PI));
        sin_ptr[p] = sinf(theta_p);
        cos_ptr[p] = cosf(theta_p);
        quadrant_ptr[p] = calc_quadrant_host(theta_p);
    }
    return {sin_theta, cos_theta, quadrant};
}

std::vector<at::Tensor> prepare_astra_geometry(at::Tensor theta) {
    CHECK_CPU(theta);
    CHECK_CONTIGUOUS(theta);
    CHECK_FLOAT(theta);
    TORCH_CHECK(theta.dim() == 1, "theta must be (dt,)");

    auto sin_theta = at::empty_like(theta);
    auto cos_theta = at::empty_like(theta);
    const float* theta_ptr = theta.data_ptr<float>();
    float* sin_ptr = sin_theta.data_ptr<float>();
    float* cos_ptr = cos_theta.data_ptr<float>();
    const int64_t dt = theta.numel();

    // ASTRA's GeometryUtil2D.cpp calls the double-precision std::sin/std::cos
    // overloads with a float angle and stores the results back into float.
    for (int64_t p = 0; p < dt; ++p) {
        const double angle = static_cast<double>(theta_ptr[p]);
        sin_ptr[p] = static_cast<float>(std::sin(angle));
        cos_ptr[p] = static_cast<float>(std::cos(angle));
    }
    return {sin_theta, cos_theta};
}

static void check_angle_tables(
    const at::Tensor& reference,
    const at::Tensor& sin_theta,
    const at::Tensor& cos_theta,
    const at::Tensor& quadrant,
    int64_t dt) {
    CHECK_CUDA(sin_theta); CHECK_CUDA(cos_theta); CHECK_CUDA(quadrant);
    CHECK_CONTIGUOUS(sin_theta); CHECK_CONTIGUOUS(cos_theta);
    CHECK_CONTIGUOUS(quadrant);
    CHECK_FLOAT(sin_theta); CHECK_FLOAT(cos_theta); CHECK_INT32(quadrant);
    CHECK_SAME_DEVICE(reference, sin_theta);
    CHECK_SAME_DEVICE(reference, cos_theta);
    CHECK_SAME_DEVICE(reference, quadrant);
    TORCH_CHECK(sin_theta.dim() == 1 && sin_theta.numel() == dt,
                "sin_theta must be (dt,)");
    TORCH_CHECK(cos_theta.dim() == 1 && cos_theta.numel() == dt,
                "cos_theta must be (dt,)");
    TORCH_CHECK(quadrant.dim() == 1 && quadrant.numel() == dt,
                "quadrant must be (dt,)");
}

static void check_linear_angle_tables(
    const at::Tensor& reference,
    const at::Tensor& sin_theta,
    const at::Tensor& cos_theta,
    int64_t dt) {
    CHECK_CUDA(sin_theta); CHECK_CUDA(cos_theta);
    CHECK_CONTIGUOUS(sin_theta); CHECK_CONTIGUOUS(cos_theta);
    CHECK_FLOAT(sin_theta); CHECK_FLOAT(cos_theta);
    CHECK_SAME_DEVICE(reference, sin_theta);
    CHECK_SAME_DEVICE(reference, cos_theta);
    TORCH_CHECK(sin_theta.dim() == 1 && sin_theta.numel() == dt,
                "sin_theta must be (dt,)");
    TORCH_CHECK(cos_theta.dim() == 1 && cos_theta.numel() == dt,
                "cos_theta must be (dt,)");
}

at::Tensor project(
    at::Tensor img,
    at::Tensor sin_theta,
    at::Tensor cos_theta,
    at::Tensor quadrant,
    at::Tensor center) {
    CHECK_CUDA(img); CHECK_CUDA(center);
    CHECK_CONTIGUOUS(img); CHECK_CONTIGUOUS(center);
    CHECK_FLOAT(img); CHECK_FLOAT(center);
    TORCH_CHECK(img.dim() == 3, "img must be (B, H, W)");

    int64_t B  = img.size(0);
    int64_t ox = img.size(1);
    int64_t oz = img.size(2);
    int64_t dt = sin_theta.numel();
    int64_t dx = oz;  // tomopy convention: detector width = oz when pad=False
    check_angle_tables(img, sin_theta, cos_theta, quadrant, dt);
    CHECK_SAME_DEVICE(img, center);
    TORCH_CHECK(center.dim() == 1 && center.numel() == B,
                "center must be (B,)");

    auto sino = at::zeros({B, dt, dx}, img.options());

    c10::cuda::CUDAGuard device_guard(img.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(img.get_device());
    launch_project(
        img.data_ptr<float>(),
        sino.data_ptr<float>(),
        sin_theta.data_ptr<float>(),
        cos_theta.data_ptr<float>(),
        quadrant.data_ptr<int32_t>(),
        center.data_ptr<float>(),
        (int)B, (int)dt, (int)dx, (int)ox, (int)oz,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sino;
}

at::Tensor bp(
    at::Tensor sino,
    at::Tensor sin_theta,
    at::Tensor cos_theta,
    at::Tensor quadrant,
    at::Tensor center,
    at::Tensor recon) {
    CHECK_CUDA(sino); CHECK_CUDA(center); CHECK_CUDA(recon);
    CHECK_CONTIGUOUS(sino); CHECK_CONTIGUOUS(center);
    CHECK_CONTIGUOUS(recon);
    CHECK_FLOAT(sino); CHECK_FLOAT(center); CHECK_FLOAT(recon);
    TORCH_CHECK(sino.dim() == 3, "sino must be (B, dt, dx)");
    TORCH_CHECK(recon.dim() == 3, "recon must be (B, ngridx, ngridy)");

    int64_t B  = sino.size(0);
    int64_t dt = sino.size(1);
    int64_t dx = sino.size(2);
    int64_t ngridx = recon.size(1);
    int64_t ngridy = recon.size(2);
    check_angle_tables(sino, sin_theta, cos_theta, quadrant, dt);
    CHECK_SAME_DEVICE(sino, center);
    CHECK_SAME_DEVICE(sino, recon);
    TORCH_CHECK(center.dim() == 1 && center.numel() == B,
                "center must be (B,)");
    TORCH_CHECK(recon.size(0) == B, "recon batch must match sino batch");

    c10::cuda::CUDAGuard device_guard(sino.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sino.get_device());
    launch_bp(
        sino.data_ptr<float>(),
        recon.data_ptr<float>(),
        sin_theta.data_ptr<float>(),
        cos_theta.data_ptr<float>(),
        quadrant.data_ptr<int32_t>(),
        center.data_ptr<float>(),
        (int)B, (int)dt, (int)dx, (int)ngridx, (int)ngridy,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return recon;
}

at::Tensor bp_linear(
    at::Tensor sino,
    at::Tensor sin_theta,
    at::Tensor cos_theta,
    at::Tensor center,
    at::Tensor recon,
    double output_scale) {
    CHECK_CUDA(sino); CHECK_CUDA(center); CHECK_CUDA(recon);
    CHECK_CONTIGUOUS(sino); CHECK_CONTIGUOUS(center);
    CHECK_CONTIGUOUS(recon);
    CHECK_FLOAT(sino); CHECK_FLOAT(center); CHECK_FLOAT(recon);
    TORCH_CHECK(sino.dim() == 3, "sino must be (B, dt, dx)");
    TORCH_CHECK(recon.dim() == 3, "recon must be (B, ngridx, ngridy)");

    int64_t B = sino.size(0);
    int64_t dt = sino.size(1);
    int64_t dx = sino.size(2);
    int64_t ngridx = recon.size(1);
    int64_t ngridy = recon.size(2);
    check_linear_angle_tables(sino, sin_theta, cos_theta, dt);
    CHECK_SAME_DEVICE(sino, center);
    CHECK_SAME_DEVICE(sino, recon);
    TORCH_CHECK(center.dim() == 1 && center.numel() == B,
                "center must be (B,)");
    TORCH_CHECK(recon.size(0) == B, "recon batch must match sino batch");

    c10::cuda::CUDAGuard device_guard(sino.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sino.get_device());
    launch_bp_linear(
        sino.data_ptr<float>(),
        recon.data_ptr<float>(),
        sin_theta.data_ptr<float>(),
        cos_theta.data_ptr<float>(),
        center.data_ptr<float>(),
        static_cast<float>(output_scale),
        (int)B,
        (int)dt,
        (int)dx,
        (int)ngridx,
        (int)ngridy,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return recon;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_geometry", &prepare_geometry,
          "Evaluate TomoPy-compatible angle tables on CPU",
          py::arg("theta"));
    m.def("prepare_astra_geometry", &prepare_astra_geometry,
          "Evaluate ASTRA-compatible angle tables on CPU",
          py::arg("theta"));
    m.def("project", &project, "TomoPy Siddon project on CUDA",
          py::arg("img"), py::arg("sin_theta"), py::arg("cos_theta"),
          py::arg("quadrant"), py::arg("center"));
    m.def("bp", &bp, "TomoPy Siddon unfiltered back-projection on CUDA",
          py::arg("sino"), py::arg("sin_theta"), py::arg("cos_theta"),
          py::arg("quadrant"), py::arg("center"), py::arg("recon"));
    m.def("bp_linear", &bp_linear,
          "ASTRA-like linear unfiltered back-projection on CUDA",
          py::arg("sino"), py::arg("sin_theta"), py::arg("cos_theta"),
          py::arg("center"), py::arg("recon"),
          py::arg("output_scale") = 1.0);
}
