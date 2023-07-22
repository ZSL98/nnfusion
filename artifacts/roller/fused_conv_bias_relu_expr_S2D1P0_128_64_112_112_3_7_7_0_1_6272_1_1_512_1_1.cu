#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cu_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <string>

//full_dimensions: [64, 1605632, 147]

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(512) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv_unpad, float* __restrict__ bias) {
  float conv_local[32];
  __shared__ float data_pad_shared[8192];
  __shared__ float kernel_pad_shared[2048];
  float data_pad_shared_local[4];
  float kernel_pad_shared_local[8];
  conv_local[0] = 0.000000e+00f;
  conv_local[4] = 0.000000e+00f;
  conv_local[8] = 0.000000e+00f;
  conv_local[12] = 0.000000e+00f;
  conv_local[16] = 0.000000e+00f;
  conv_local[20] = 0.000000e+00f;
  conv_local[24] = 0.000000e+00f;
  conv_local[28] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[5] = 0.000000e+00f;
  conv_local[9] = 0.000000e+00f;
  conv_local[13] = 0.000000e+00f;
  conv_local[17] = 0.000000e+00f;
  conv_local[21] = 0.000000e+00f;
  conv_local[25] = 0.000000e+00f;
  conv_local[29] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[6] = 0.000000e+00f;
  conv_local[10] = 0.000000e+00f;
  conv_local[14] = 0.000000e+00f;
  conv_local[18] = 0.000000e+00f;
  conv_local[22] = 0.000000e+00f;
  conv_local[26] = 0.000000e+00f;
  conv_local[30] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  conv_local[7] = 0.000000e+00f;
  conv_local[11] = 0.000000e+00f;
  conv_local[15] = 0.000000e+00f;
  conv_local[19] = 0.000000e+00f;
  conv_local[23] = 0.000000e+00f;
  conv_local[27] = 0.000000e+00f;
  conv_local[31] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 5; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + (((ra_fused0_outer * 4) + (((int)threadIdx.x) >> 8)) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 512)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 2) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 2) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 2) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 1024)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 4) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 4) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 4) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 1536)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 6) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 6) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 6) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 2048)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 8) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 8) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 1) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 2560)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 10) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 10) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 3) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 3072)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 12) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 12) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 5) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 3584)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 14) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) / 7) + 2) % 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + (((ra_fused0_outer * 4) + (((int)threadIdx.x) >> 8)) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 4096)] = data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 16) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 16) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 2) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 4608)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 129) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 18) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 18) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 4) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 5120)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 127) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 20) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 20) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 6) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 5632)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 125) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 22) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 22) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 1) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 6144)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 123) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 24) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 24) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 3) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 6656)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 121) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 26) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 26) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 5) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 7168)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 119) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 28) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) / 7) + 4) % 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + (((ra_fused0_outer * 4) + (((int)threadIdx.x) >> 8)) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 7680)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) < 117) ? data[(((((((((int)blockIdx.x) / 49) * 158700) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 30) / 49) * 52900)) + (((((((int)blockIdx.x) % 49) * 16) + ((((int)threadIdx.x) & 255) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 30) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 255)) % 112) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 8)) + 2) % 7))] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) < 147) ? kernel[((((((int)threadIdx.x) >> 5) * 147) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) < 147) ? kernel[(((((((int)threadIdx.x) >> 5) * 147) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2352)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 1024)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) < 147) ? kernel[(((((((int)threadIdx.x) >> 5) * 147) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 4704)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 1536)] = ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) < 147) ? kernel[(((((((int)threadIdx.x) >> 5) * 147) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 7056)] : 0.000000e+00f);
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 256) + (((int)threadIdx.x) & 63))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 256) + (((int)threadIdx.x) & 63)) + 64)];
      data_pad_shared_local[2] = data_pad_shared[(((ra_fused0_inner_outer * 256) + (((int)threadIdx.x) & 63)) + 128)];
      data_pad_shared_local[3] = data_pad_shared[(((ra_fused0_inner_outer * 256) + (((int)threadIdx.x) & 63)) + 192)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 256)];
      kernel_pad_shared_local[2] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 512)];
      kernel_pad_shared_local[3] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 768)];
      kernel_pad_shared_local[4] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 1024)];
      kernel_pad_shared_local[5] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 1280)];
      kernel_pad_shared_local[6] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 1536)];
      kernel_pad_shared_local[7] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 32) + ra_fused0_inner_outer) + 1792)];
      if (((ra_fused0_outer * 32) + ra_fused0_inner_outer) < 147) {
        conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
        conv_local[4] = (conv_local[4] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
        conv_local[8] = (conv_local[8] + (data_pad_shared_local[0] * kernel_pad_shared_local[2]));
        conv_local[12] = (conv_local[12] + (data_pad_shared_local[0] * kernel_pad_shared_local[3]));
        conv_local[16] = (conv_local[16] + (data_pad_shared_local[0] * kernel_pad_shared_local[4]));
        conv_local[20] = (conv_local[20] + (data_pad_shared_local[0] * kernel_pad_shared_local[5]));
        conv_local[24] = (conv_local[24] + (data_pad_shared_local[0] * kernel_pad_shared_local[6]));
        conv_local[28] = (conv_local[28] + (data_pad_shared_local[0] * kernel_pad_shared_local[7]));
        conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
        conv_local[5] = (conv_local[5] + (data_pad_shared_local[1] * kernel_pad_shared_local[1]));
        conv_local[9] = (conv_local[9] + (data_pad_shared_local[1] * kernel_pad_shared_local[2]));
        conv_local[13] = (conv_local[13] + (data_pad_shared_local[1] * kernel_pad_shared_local[3]));
        conv_local[17] = (conv_local[17] + (data_pad_shared_local[1] * kernel_pad_shared_local[4]));
        conv_local[21] = (conv_local[21] + (data_pad_shared_local[1] * kernel_pad_shared_local[5]));
        conv_local[25] = (conv_local[25] + (data_pad_shared_local[1] * kernel_pad_shared_local[6]));
        conv_local[29] = (conv_local[29] + (data_pad_shared_local[1] * kernel_pad_shared_local[7]));
        conv_local[2] = (conv_local[2] + (data_pad_shared_local[2] * kernel_pad_shared_local[0]));
        conv_local[6] = (conv_local[6] + (data_pad_shared_local[2] * kernel_pad_shared_local[1]));
        conv_local[10] = (conv_local[10] + (data_pad_shared_local[2] * kernel_pad_shared_local[2]));
        conv_local[14] = (conv_local[14] + (data_pad_shared_local[2] * kernel_pad_shared_local[3]));
        conv_local[18] = (conv_local[18] + (data_pad_shared_local[2] * kernel_pad_shared_local[4]));
        conv_local[22] = (conv_local[22] + (data_pad_shared_local[2] * kernel_pad_shared_local[5]));
        conv_local[26] = (conv_local[26] + (data_pad_shared_local[2] * kernel_pad_shared_local[6]));
        conv_local[30] = (conv_local[30] + (data_pad_shared_local[2] * kernel_pad_shared_local[7]));
        conv_local[3] = (conv_local[3] + (data_pad_shared_local[3] * kernel_pad_shared_local[0]));
        conv_local[7] = (conv_local[7] + (data_pad_shared_local[3] * kernel_pad_shared_local[1]));
        conv_local[11] = (conv_local[11] + (data_pad_shared_local[3] * kernel_pad_shared_local[2]));
        conv_local[15] = (conv_local[15] + (data_pad_shared_local[3] * kernel_pad_shared_local[3]));
        conv_local[19] = (conv_local[19] + (data_pad_shared_local[3] * kernel_pad_shared_local[4]));
        conv_local[23] = (conv_local[23] + (data_pad_shared_local[3] * kernel_pad_shared_local[5]));
        conv_local[27] = (conv_local[27] + (data_pad_shared_local[3] * kernel_pad_shared_local[6]));
        conv_local[31] = (conv_local[31] + (data_pad_shared_local[3] * kernel_pad_shared_local[7]));
      }
    }
  }
  conv_unpad[((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63))] = max((conv_local[0] + bias[(((int)threadIdx.x) >> 6)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 12845056)] = max((conv_local[4] + bias[((((int)threadIdx.x) >> 6) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 25690112)] = max((conv_local[8] + bias[((((int)threadIdx.x) >> 6) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 38535168)] = max((conv_local[12] + bias[((((int)threadIdx.x) >> 6) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 51380224)] = max((conv_local[16] + bias[((((int)threadIdx.x) >> 6) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 64225280)] = max((conv_local[20] + bias[((((int)threadIdx.x) >> 6) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 77070336)] = max((conv_local[24] + bias[((((int)threadIdx.x) >> 6) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 89915392)] = max((conv_local[28] + bias[((((int)threadIdx.x) >> 6) + 56)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 64)] = max((conv_local[1] + bias[(((int)threadIdx.x) >> 6)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 12845120)] = max((conv_local[5] + bias[((((int)threadIdx.x) >> 6) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 25690176)] = max((conv_local[9] + bias[((((int)threadIdx.x) >> 6) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 38535232)] = max((conv_local[13] + bias[((((int)threadIdx.x) >> 6) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 51380288)] = max((conv_local[17] + bias[((((int)threadIdx.x) >> 6) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 64225344)] = max((conv_local[21] + bias[((((int)threadIdx.x) >> 6) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 77070400)] = max((conv_local[25] + bias[((((int)threadIdx.x) >> 6) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 89915456)] = max((conv_local[29] + bias[((((int)threadIdx.x) >> 6) + 56)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 128)] = max((conv_local[2] + bias[(((int)threadIdx.x) >> 6)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 12845184)] = max((conv_local[6] + bias[((((int)threadIdx.x) >> 6) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 25690240)] = max((conv_local[10] + bias[((((int)threadIdx.x) >> 6) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 38535296)] = max((conv_local[14] + bias[((((int)threadIdx.x) >> 6) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 51380352)] = max((conv_local[18] + bias[((((int)threadIdx.x) >> 6) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 64225408)] = max((conv_local[22] + bias[((((int)threadIdx.x) >> 6) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 77070464)] = max((conv_local[26] + bias[((((int)threadIdx.x) >> 6) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 89915520)] = max((conv_local[30] + bias[((((int)threadIdx.x) >> 6) + 56)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 192)] = max((conv_local[3] + bias[(((int)threadIdx.x) >> 6)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 12845248)] = max((conv_local[7] + bias[((((int)threadIdx.x) >> 6) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 25690304)] = max((conv_local[11] + bias[((((int)threadIdx.x) >> 6) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 38535360)] = max((conv_local[15] + bias[((((int)threadIdx.x) >> 6) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 51380416)] = max((conv_local[19] + bias[((((int)threadIdx.x) >> 6) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 64225472)] = max((conv_local[23] + bias[((((int)threadIdx.x) >> 6) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 77070528)] = max((conv_local[27] + bias[((((int)threadIdx.x) >> 6) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 6) * 1605632) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) & 63)) + 89915584)] = max((conv_local[31] + bias[((((int)threadIdx.x) >> 6) + 56)]), 0.000000e+00f);
}


int main(int argc, char *argv[])
{
    std::string path;
    int input_size0 = 20313600;
    int input_size1 = 9408;
    int input_size2 = 64;
    int output_size0 = 102760448;

    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

    float *input0h, *input1h, *input2h, *output0h;
    float *input0d, *input1d, *input2d, *output0d;
    input0h = (float*)malloc(81254400);
    input1h = (float*)malloc(37632);
    input2h = (float*)malloc(256);

    cudaMalloc((void **)&input0d, 81254400);
    cudaMalloc((void **)&input1d, 37632);
    cudaMalloc((void **)&input2d, 256);
    cudaMalloc((void **)&output0d, 411041792);

    srand(1);
    for (int i = 0; i < input_size0; ++ i)
        input0h[i] = 1;
    for (int i = 0; i < input_size1; ++ i)
        input1h[i] = 1;
    for (int i = 0; i < input_size2; ++ i)
        input2h[i] = 1;

    cudaMemcpy(input0d, input0h, 81254400, cudaMemcpyHostToDevice);
    cudaMemcpy(input1d, input1h, 37632, cudaMemcpyHostToDevice);
    cudaMemcpy(input2d, input2h, 256, cudaMemcpyHostToDevice);

    dim3 grid(6272, 1, 1);
    dim3 block(512, 1, 1);

    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)input0d, (float*)input1d, (float*)output0d, (float*)input2d);
        cudaDeviceSynchronize();
    }
}
