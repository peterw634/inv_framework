// CUDA Phase-1 PSWF gridding for TomoPy 1.15.x Gridrec in the unified
// nasn_ct.operators.radon_cuda package.
//
// Derived from TomoPy, Copyright (c) 2015 UChicago Argonne, LLC.
// The BSD-3-Clause terms are retained in LICENSES/TomoPy-BSD-3-Clause.txt.
// This file is a
// modified CUDA/PyTorch implementation, not an upstream TomoPy source file.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

__global__ void tomopy_gridrec_gridding_kernel(
    const float2* __restrict__ cproj,
    float2* __restrict__ grid,
    const float2* __restrict__ filphase,
    const float* __restrict__ sine,
    const float* __restrict__ cose,
    const float* __restrict__ wtbl,
    int batch_pairs,
    int n_ang,
    int pdim,
    int filter_rows)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int angle = blockIdx.y;
    const int batch = blockIdx.z;
    const int pdim2 = pdim >> 1;
    if (j >= pdim2 || angle >= n_ang || batch >= batch_pairs) return;

    constexpr int ltbl = 512;
    constexpr int L2 = 2;          // int(C/pi), C=7
    constexpr float tblspcg = 256.0f; // 2*ltbl/int(2*C/pi)

    const int filter_row = filter_rows == 1 ? 0 : angle;
    const float2 phase = filphase[
        (batch * filter_rows + filter_row) * pdim2 + j];
    const int projection_base = (batch * n_ang + angle) * pdim;
    const float2 positive = cproj[projection_base + j];
    const float2 negative = cproj[projection_base + pdim - j];

    // Cdata1 = phase * positive.
    float2 cdata1;
    cdata1.x = phase.x * positive.x - phase.y * positive.y;
    cdata1.y = phase.x * positive.y + phase.y * positive.x;
    // Cdata2 = conj(phase) * negative.
    float2 cdata2;
    cdata2.x = phase.x * negative.x + phase.y * negative.y;
    cdata2.y = phase.x * negative.y - phase.y * negative.x;

    const float U = j * cose[angle] + pdim2;
    const float V = j * sine[angle] + pdim2;
    int iul = max(static_cast<int>(ceilf(U - L2)), 1);
    int iuh = min(static_cast<int>(floorf(U + L2)), pdim - 1);
    int ivl = max(static_cast<int>(ceilf(V - L2)), 1);
    int ivh = min(static_cast<int>(floorf(V + L2)), pdim - 1);

    float v_weights[5];
    int v_count = ivh - ivl + 1;
    for (int offset = 0; offset < v_count; ++offset) {
        float table_coordinate = fabsf(V - (ivl + offset)) * tblspcg;
        int table_index = static_cast<int>(roundf(table_coordinate));
        table_index = min(max(table_index, 0), ltbl);
        v_weights[offset] = wtbl[table_index];
    }

    float2* batch_grid = grid + batch * pdim * pdim;
    for (int iu = iul; iu <= iuh; ++iu) {
        float table_coordinate = fabsf(U - iu) * tblspcg;
        int table_index = static_cast<int>(roundf(table_coordinate));
        table_index = min(max(table_index, 0), ltbl);
        float u_weight = wtbl[table_index];
        for (int offset = 0; offset < v_count; ++offset) {
            int iv = ivl + offset;
            float weight = u_weight * v_weights[offset];
            float2* target1 = batch_grid + iu * pdim + iv;
            float2* target2 = batch_grid
                + (pdim - iu) * pdim + (pdim - iv);
            atomicAdd(&target1->x, weight * cdata1.x);
            atomicAdd(&target1->y, weight * cdata1.y);
            atomicAdd(&target2->x, weight * cdata2.x);
            atomicAdd(&target2->y, weight * cdata2.y);
        }
    }
}

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
    cudaStream_t stream)
{
    const int frequency_count = pdim / 2 - 1;
    dim3 block(256);
    dim3 launch_grid(
        (frequency_count + block.x - 1) / block.x,
        n_ang,
        batch_pairs);
    tomopy_gridrec_gridding_kernel<<<launch_grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(cproj),
        reinterpret_cast<float2*>(grid),
        reinterpret_cast<const float2*>(filphase),
        sine,
        cose,
        wtbl,
        batch_pairs,
        n_ang,
        pdim,
        filter_rows);
}
