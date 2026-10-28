// CUDA port of tomopy.project / tomopy.recon(fbp,filter=none).
//
// References (file:line):
//   project()      tomopy/source/libtomo/recon/project.c:50
//   fbp()          tomopy/source/libtomo/recon/fbp.c:48
//   preprocessing  tomopy/source/libtomo/recon/utils.c:67
//   calc_coords    tomopy/source/libtomo/recon/utils.c:118
//   trim_coords    tomopy/source/libtomo/recon/utils.c:143
//   sort_intersect tomopy/source/libtomo/recon/utils.c:179
//   calc_dist      tomopy/source/libtomo/recon/utils.c:260
//
// Each ray traces the Siddon intersections with grid lines in the same order
// as the C reference, but in a streaming fashion (no per-thread O(N) arrays).
// fmodf/sinf/cosf/quadrant are precomputed on the CPU by radon_cuda.cpp so the
// clipping decisions use the same host-libm values as TomoPy.
//
// Layouts (row-major, matches the C kernels):
//   img/recon : (B, ox, oz)         float32
//   sino/data : (B, dt, dx)         float32
//
// Forward kernel : sino[s, p, d]  = sum_n  img[s, indi_n]  * dist[n]
// Adjoint kernel : recon[s, indi_n] += data[s, p, d] * dist[n]    (atomicAdd)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <stdint.h>
#include <torch/extension.h>

// ---- preprocessing (utils.c:67) -------------------------------------------
// gridx[i] = -ox/2 + i,   gridy[j] = -oz/2 + j
// mov = (dx-1)/2 - center;  if mov-floor(mov) < 0.01: mov += 0.01;  mov += 0.5
__device__ __forceinline__ float compute_mov(float center, int dx) {
    float mov = ((float)dx - 1.0f) * 0.5f - center;
    if ((mov - floorf(mov)) < 0.01f) {
        mov += 0.01f;
    }
    mov += 0.5f;
    return mov;
}

// Per-ray Siddon walk. Templated on whether we forward project (gather, no
// atomics) or back project (scatter via atomicAdd).
template <bool IS_FORWARD>
__device__ __forceinline__ void process_ray(
    int s, int p, int d,
    const float* __restrict__ img_or_data,
    float*       __restrict__ data_or_img,
    float sin_p, float cos_p, int quadrant,
    int ox, int oz, int dt, int dx, float mov)
{
    // Ray endpoints in object frame.
    // xi very negative pushes the source far behind the image; the actual
    // intersections are determined by the trim filter on the grid edges.
    float xi = (float)(-ox - oz);
    float yi = 0.5f * (float)(1 - dx) + (float)d + mov;

    float srcx =  xi * cos_p - yi * sin_p;
    float srcy =  xi * sin_p + yi * cos_p;
    float detx = -xi * cos_p - yi * sin_p;
    float dety = -xi * sin_p + yi * cos_p;

    float slope  = (srcy - dety) / (srcx - detx);
    float islope = (srcx - detx) / (srcy - dety);

    // Trim band from utils.c:149 / utils.c:162.
    const float gridx0 = -(float)ox * 0.5f;
    const float gridy0 = -(float)oz * 0.5f;
    float gridx_gt = gridx0 + 0.01f;
    float gridx_le = gridx0 + (float)ox - 0.01f;
    float gridy_gt = gridy0 + 0.01f;
    float gridy_le = gridy0 + (float)oz - 0.01f;

    // Two streams (Siddon merge). Stream A: gridy crossings (vary i_a in
    // [0, oz]); Stream B: gridx crossings (vary i_b in [0, ox]). For each
    // we emit (x, y) in ascending x order.
    //   Stream A direction: forward if quadrant==1 (islope>=0), reverse if
    //                       quadrant==0 (islope<0). Matches sort_intersections.
    //   Stream B is always forward (gridx itself is monotone increasing).
    int i_a, i_a_end, da;
    if (quadrant == 1) { i_a = 0;       i_a_end = oz + 1; da = +1; }
    else               { i_a = oz;      i_a_end = -1;     da = -1; }
    int i_b = 0, i_b_end = ox + 1, db = +1;

    // skip-while helpers, recompute candidate each step
    float ax_x = 0.0f, ax_y = 0.0f;
    float bx_x = 0.0f, bx_y = 0.0f;
    bool have_a = false, have_b = false;

    // initial fill
    while (i_a != i_a_end) {
        float gy = gridy0 + (float)i_a;
        float cx = islope * (gy - srcy) + srcx;
        if (cx >= gridx_gt && cx <= gridx_le) {
            ax_x = cx; ax_y = gy; have_a = true; break;
        }
        i_a += da;
    }
    while (i_b != i_b_end) {
        float gx = gridx0 + (float)i_b;
        float cy = slope * (gx - srcx) + srcy;
        if (cy >= gridy_gt && cy <= gridy_le) {
            bx_x = gx; bx_y = cy; have_b = true; break;
        }
        i_b += db;
    }

    // Pick the first emit (smallest x). Set as prev.
    float prev_x, prev_y;
    if (have_a && have_b) {
        if (ax_x < bx_x) {
            prev_x = ax_x; prev_y = ax_y; i_a += da; have_a = false;
        } else {
            prev_x = bx_x; prev_y = bx_y; i_b += db; have_b = false;
        }
    } else if (have_a) {
        prev_x = ax_x; prev_y = ax_y; i_a += da; have_a = false;
    } else if (have_b) {
        prev_x = bx_x; prev_y = bx_y; i_b += db; have_b = false;
    } else {
        // No intersections at all.
        if (IS_FORWARD) {
            data_or_img[s * dt * dx + p * dx + d] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    int sino_off = s * dt * dx + p * dx + d;
    float data_v = IS_FORWARD ? 0.0f : img_or_data[sino_off];

    while (true) {
        // refill missing side
        if (!have_a) {
            while (i_a != i_a_end) {
                float gy = gridy0 + (float)i_a;
                float cx = islope * (gy - srcy) + srcx;
                if (cx >= gridx_gt && cx <= gridx_le) {
                    ax_x = cx; ax_y = gy; have_a = true; break;
                }
                i_a += da;
            }
        }
        if (!have_b) {
            while (i_b != i_b_end) {
                float gx = gridx0 + (float)i_b;
                float cy = slope * (gx - srcx) + srcy;
                if (cy >= gridy_gt && cy <= gridy_le) {
                    bx_x = gx; bx_y = cy; have_b = true; break;
                }
                i_b += db;
            }
        }
        if (!have_a && !have_b) break;

        float cur_x, cur_y;
        if (have_a && (!have_b || ax_x < bx_x)) {
            cur_x = ax_x; cur_y = ax_y; i_a += da; have_a = false;
        } else {
            cur_x = bx_x; cur_y = bx_y; i_b += db; have_b = false;
        }

        // segment dist + midpoint pixel (calc_dist utils.c:260)
        float diffx = cur_x - prev_x;
        float diffy = cur_y - prev_y;
        float diffx2 = diffx * diffx;
        float diffy2 = diffy * diffy;
        float dist = sqrtf(diffx2 + diffy2);

        // Match utils.c:calc_dist expression order and cast semantics.
        float midx = 0.5f * (prev_x + cur_x);
        float x1 = midx + 0.5f * (float)ox;
        float i1 = (float)((int)x1);
        int indx = (int)(i1 - (i1 > x1));

        float midy = 0.5f * (prev_y + cur_y);
        float x2 = midy + 0.5f * (float)oz;
        float i2 = (float)((int)x2);
        int indy = (int)(i2 - (i2 > x2));

        if (indx >= 0 && indx < ox && indy >= 0 && indy < oz) {
            int vox = indx * oz + indy;
            int img_off = s * ox * oz + vox;
            if (IS_FORWARD) {
                sum += img_or_data[img_off] * dist;
            } else {
                atomicAdd(&data_or_img[img_off], data_v * dist);
            }
        }

        prev_x = cur_x;
        prev_y = cur_y;
    }

    if (IS_FORWARD) {
        data_or_img[sino_off] = sum;
    }
}

// ---- kernels --------------------------------------------------------------
// gridDim = (ceil(dx/blockX), dt, dy);  blockDim = (blockX,)
__global__ void project_kernel(
    const float* __restrict__ img,
    float*       __restrict__ sino,
    const float* __restrict__ sin_theta,
    const float* __restrict__ cos_theta,
    const int32_t* __restrict__ quadrant,
    const float* __restrict__ center,
    int dy, int dt, int dx, int ox, int oz)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dx) return;
    int p = blockIdx.y;
    int s = blockIdx.z;

    float mov = compute_mov(center[s], dx);

    process_ray<true>(s, p, d, img, sino,
                      sin_theta[p], cos_theta[p], quadrant[p],
                      ox, oz, dt, dx, mov);
}

__global__ void bp_kernel(
    const float* __restrict__ sino,
    float*       __restrict__ recon,
    const float* __restrict__ sin_theta,
    const float* __restrict__ cos_theta,
    const int32_t* __restrict__ quadrant,
    const float* __restrict__ center,
    int dy, int dt, int dx, int ox, int oz)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dx) return;
    int p = blockIdx.y;
    int s = blockIdx.z;

    float mov = compute_mov(center[s], dx);

    process_ray<false>(s, p, d, sino, recon,
                       sin_theta[p], cos_theta[p], quadrant[p],
                       ox, oz, dt, dx, mov);
}

// ASTRA-like pixel-driven parallel-beam back-projection.  This follows
// astra-toolbox v2.4.0 cuda/2d/par_bp.cu::devBP for unit detector spacing and
// the postalignment convention used by TomoPy's ASTRA wrapper.  In this array
// layout ASTRA's volume X coordinate is the second image index and volume Y is
// the first image index, hence the apparently swapped ngridx/ngridy terms.
__global__ void bp_linear_kernel(
    float* __restrict__ recon,
    cudaTextureObject_t sino_texture,
    const float* __restrict__ sin_theta,
    const float* __restrict__ cos_theta,
    const float* __restrict__ center,
    float output_scale,
    int start_angle,
    int dt,
    int ngridx,
    int ngridy)
{
    int astra_x = blockIdx.x * blockDim.x + threadIdx.x;
    int astra_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (astra_x >= ngridy || astra_y >= ngridx) return;

    int end_angle = min(start_angle + 16, dt);
    float f_x = (float)astra_x - 0.5f * (float)ngridy + 0.5f;
    float f_y = (float)astra_y - 0.5f * (float)ngridx + 0.5f;
    float f_a = (float)start_angle + 0.5f;
    float value = 0.0f;

    for (int angle = start_angle; angle < end_angle; ++angle) {
        float detector_coordinate =
            f_x * cos_theta[angle] - f_y * sin_theta[angle] + center[0];
        value += tex2D<float>(sino_texture, detector_coordinate, f_a);
        f_a += 1.0f;
    }

    recon[astra_y * ngridy + astra_x] += value * output_scale;
}

// ---- C++ entrypoints (called from radon_cuda.cpp) -------------------------
void launch_project(
    const float* img, float* sino,
    const float* sin_theta, const float* cos_theta,
    const int32_t* quadrant, const float* center,
    int dy, int dt, int dx, int ox, int oz,
    cudaStream_t stream)
{
    int blockX = 128;
    dim3 block(blockX);
    dim3 grid((dx + blockX - 1) / blockX, dt, dy);
    project_kernel<<<grid, block, 0, stream>>>(
        img, sino, sin_theta, cos_theta, quadrant, center,
        dy, dt, dx, ox, oz);
}

void launch_bp(
    const float* sino, float* recon,
    const float* sin_theta, const float* cos_theta,
    const int32_t* quadrant, const float* center,
    int dy, int dt, int dx, int ox, int oz,
    cudaStream_t stream)
{
    int blockX = 128;
    dim3 block(blockX);
    dim3 grid((dx + blockX - 1) / blockX, dt, dy);
    bp_kernel<<<grid, block, 0, stream>>>(
        sino, recon, sin_theta, cos_theta, quadrant, center,
        dy, dt, dx, ox, oz);
}

void launch_bp_linear(
    const float* sino, float* recon,
    const float* sin_theta, const float* cos_theta,
    const float* center,
    float output_scale,
    int batch, int dt, int dx, int ngridx, int ngridy,
    cudaStream_t stream)
{
    cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    dim3 block(16, 32);
    dim3 grid(
        (ngridy + block.x - 1) / block.x,
        (ngridx + block.y - 1) / block.y);

    for (int sample = 0; sample < batch; ++sample) {
        cudaArray_t sino_array = nullptr;
        C10_CUDA_CHECK(cudaMallocArray(
            &sino_array, &channel_desc, (size_t)dx, (size_t)dt));
        C10_CUDA_CHECK(cudaMemcpy2DToArrayAsync(
            sino_array,
            0,
            0,
            sino + (size_t)sample * (size_t)dt * (size_t)dx,
            (size_t)dx * sizeof(float),
            (size_t)dx * sizeof(float),
            (size_t)dt,
            cudaMemcpyDeviceToDevice,
            stream));

        cudaResourceDesc resource_desc;
        memset(&resource_desc, 0, sizeof(resource_desc));
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = sino_array;

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0] = cudaAddressModeBorder;
        texture_desc.addressMode[1] = cudaAddressModeBorder;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = 0;

        cudaTextureObject_t sino_texture = 0;
        C10_CUDA_CHECK(cudaCreateTextureObject(
            &sino_texture, &resource_desc, &texture_desc, nullptr));

        float* sample_recon =
            recon + (size_t)sample * (size_t)ngridx * (size_t)ngridy;
        for (int start_angle = 0; start_angle < dt; start_angle += 16) {
            bp_linear_kernel<<<grid, block, 0, stream>>>(
                sample_recon,
                sino_texture,
                sin_theta,
                cos_theta,
                center + sample,
                output_scale,
                start_angle,
                dt,
                ngridx,
                ngridy);
        }

        // A texture object and its backing array must remain alive until all
        // kernels using them have finished.  This mirrors ASTRA's own BP path.
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        C10_CUDA_CHECK(cudaDestroyTextureObject(sino_texture));
        C10_CUDA_CHECK(cudaFreeArray(sino_array));
    }
}
