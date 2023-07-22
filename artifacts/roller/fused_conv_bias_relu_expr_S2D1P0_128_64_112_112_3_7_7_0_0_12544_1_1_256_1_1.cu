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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv_unpad, float* __restrict__ bias) {
  float conv_local[32];
  __shared__ float data_pad_shared[1024];
  __shared__ float kernel_pad_shared[512];
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
  for (int ra_fused0_outer = 0; ra_fused0_outer < 19; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = data[(((((((((int)blockIdx.x) / 98) * 158700) + ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) / 49) * 52900)) + (((((((int)blockIdx.x) % 98) * 8) + ((((int)threadIdx.x) & 127) >> 4)) / 7) * 460)) + (((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) & 127)) % 112) * 2)) + (((((int)threadIdx.x) >> 7) + ra_fused0_outer) % 7))];
    data_pad_shared[(((int)threadIdx.x) + 256)] = ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) < 145) ? data[(((((((((int)blockIdx.x) / 98) * 158700) + (((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 2) / 49) * 52900)) + (((((((int)blockIdx.x) % 98) * 8) + ((((int)threadIdx.x) & 127) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 2) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) & 127)) % 112) * 2)) + ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 2) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 512)] = ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) < 143) ? data[(((((((((int)blockIdx.x) / 98) * 158700) + (((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 4) / 49) * 52900)) + (((((((int)blockIdx.x) % 98) * 8) + ((((int)threadIdx.x) & 127) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 4) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) & 127)) % 112) * 2)) + ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 4) % 7))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 768)] = ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) < 141) ? data[(((((((((int)blockIdx.x) / 98) * 158700) + (((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 6) / 49) * 52900)) + (((((((int)blockIdx.x) % 98) * 8) + ((((int)threadIdx.x) & 127) >> 4)) / 7) * 460)) + ((((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 6) % 49) / 7) * 230)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) & 127)) % 112) * 2)) + ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 7)) + 6) % 7))] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = ((((ra_fused0_outer * 8) + (((int)threadIdx.x) & 7)) < 147) ? kernel[((((((int)threadIdx.x) >> 3) * 147) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = ((((ra_fused0_outer * 8) + (((int)threadIdx.x) & 7)) < 147) ? kernel[(((((((int)threadIdx.x) >> 3) * 147) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7)) + 4704)] : 0.000000e+00f);
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 8; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 128) + (((int)threadIdx.x) & 31))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 32)];
      data_pad_shared_local[2] = data_pad_shared[(((ra_fused0_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 64)];
      data_pad_shared_local[3] = data_pad_shared[(((ra_fused0_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 96)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 64)];
      kernel_pad_shared_local[2] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 128)];
      kernel_pad_shared_local[3] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 192)];
      kernel_pad_shared_local[4] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 256)];
      kernel_pad_shared_local[5] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 320)];
      kernel_pad_shared_local[6] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 384)];
      kernel_pad_shared_local[7] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 8) + ra_fused0_inner_outer) + 448)];
      if (((ra_fused0_outer * 8) + ra_fused0_inner_outer) < 147) {
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
  conv_unpad[((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31))] = max((conv_local[0] + bias[(((int)threadIdx.x) >> 5)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 12845056)] = max((conv_local[4] + bias[((((int)threadIdx.x) >> 5) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 25690112)] = max((conv_local[8] + bias[((((int)threadIdx.x) >> 5) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 38535168)] = max((conv_local[12] + bias[((((int)threadIdx.x) >> 5) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 51380224)] = max((conv_local[16] + bias[((((int)threadIdx.x) >> 5) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64225280)] = max((conv_local[20] + bias[((((int)threadIdx.x) >> 5) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 77070336)] = max((conv_local[24] + bias[((((int)threadIdx.x) >> 5) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 89915392)] = max((conv_local[28] + bias[((((int)threadIdx.x) >> 5) + 56)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32)] = max((conv_local[1] + bias[(((int)threadIdx.x) >> 5)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 12845088)] = max((conv_local[5] + bias[((((int)threadIdx.x) >> 5) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 25690144)] = max((conv_local[9] + bias[((((int)threadIdx.x) >> 5) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 38535200)] = max((conv_local[13] + bias[((((int)threadIdx.x) >> 5) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 51380256)] = max((conv_local[17] + bias[((((int)threadIdx.x) >> 5) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64225312)] = max((conv_local[21] + bias[((((int)threadIdx.x) >> 5) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 77070368)] = max((conv_local[25] + bias[((((int)threadIdx.x) >> 5) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 89915424)] = max((conv_local[29] + bias[((((int)threadIdx.x) >> 5) + 56)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64)] = max((conv_local[2] + bias[(((int)threadIdx.x) >> 5)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 12845120)] = max((conv_local[6] + bias[((((int)threadIdx.x) >> 5) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 25690176)] = max((conv_local[10] + bias[((((int)threadIdx.x) >> 5) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 38535232)] = max((conv_local[14] + bias[((((int)threadIdx.x) >> 5) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 51380288)] = max((conv_local[18] + bias[((((int)threadIdx.x) >> 5) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64225344)] = max((conv_local[22] + bias[((((int)threadIdx.x) >> 5) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 77070400)] = max((conv_local[26] + bias[((((int)threadIdx.x) >> 5) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 89915456)] = max((conv_local[30] + bias[((((int)threadIdx.x) >> 5) + 56)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 96)] = max((conv_local[3] + bias[(((int)threadIdx.x) >> 5)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 12845152)] = max((conv_local[7] + bias[((((int)threadIdx.x) >> 5) + 8)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 25690208)] = max((conv_local[11] + bias[((((int)threadIdx.x) >> 5) + 16)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 38535264)] = max((conv_local[15] + bias[((((int)threadIdx.x) >> 5) + 24)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 51380320)] = max((conv_local[19] + bias[((((int)threadIdx.x) >> 5) + 32)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64225376)] = max((conv_local[23] + bias[((((int)threadIdx.x) >> 5) + 40)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 77070432)] = max((conv_local[27] + bias[((((int)threadIdx.x) >> 5) + 48)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 5) * 1605632) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 89915488)] = max((conv_local[31] + bias[((((int)threadIdx.x) >> 5) + 56)]), 0.000000e+00f);
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

    dim3 grid(12544, 1, 1);
    dim3 block(256, 1, 1);

    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)input0d, (float*)input1d, (float*)output0d, (float*)input2d);
        cudaDeviceSynchronize();
    }
}
