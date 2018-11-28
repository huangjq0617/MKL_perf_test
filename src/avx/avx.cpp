// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "stdafx.h"
#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <malloc.h>
#include <time.h>

#include "sse_gemm.h"
#include "avx2_gemm.h"
#include <mkl.h>

// Comptue A*B^T very naively.
void SlowRef_MatrixMult(const float * A, const float * B, float * C, int num_A_rows, int num_B_rows, int width)
{
  for (int i = 0; i < num_A_rows; i++) {
    const float * A_row = A + i*width;
    float * C_row = C + i*num_B_rows;
    for (int j = 0; j < num_B_rows; j++) {
      const float * B_row = B + j*width;
      float sum = 0.0f;
      for (int k = 0; k < width; k++) {
        sum += A_row[k] * B_row[k];
      }
      C_row[j] = sum;
    }
  }
}

int avx2_test() {
  srand(45678);

  // A is usually an activation matrix, B is usually a weight matrix.
  // We actually comptue A * B^T. num_B_rows is the rows in B^T. 
  int num_A_rows =4;
  int num_B_rows = 1024;
  // This is the shared dimension.
  int width = 1024;
  int DIV = 1;

  printf("Computing matrix multiplication: %d x %d x %d\n", num_A_rows, width, num_B_rows);

  //assert(num_A_rows % 4 == 0);
  //assert(width % 8 == 0);



  float * A = new float[num_A_rows*width];
  float * B = new float[num_B_rows*width];

  for (int i = 0; i < num_A_rows*width; i++) {
    A[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
  }

  for (int i = 0; i < num_B_rows*width; i++) {
    B[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
  }

  // C will thus be num_A_rows x num_B_rows
  float * ref_C = new float[num_A_rows*num_B_rows];
  double per_mul_time_slow = 0;
  memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);
  {
    int TIMES = 100 / DIV;
    clock_t start = clock();
    for (int i = 0; i < TIMES; i++)
      SlowRef_MatrixMult(A, B, /* out */ ref_C, num_A_rows, num_B_rows, width);

    clock_t end = clock();
    per_mul_time_slow = 1.0 * (end - start) / CLOCKS_PER_SEC / TIMES;
  }

  float * ref_C_MKL = new float[num_A_rows*num_B_rows];
  double per_mul_time_mkl = 0;
  memset(ref_C_MKL, 0, sizeof(float)*num_A_rows*num_B_rows);
  {
    int TIMES = 100 / DIV;
    clock_t start = clock();
    for (int i = 0; i < TIMES; i++)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_A_rows, width, num_B_rows, 1, A, width, B, width, 0, ref_C_MKL, num_B_rows);

    clock_t end = clock();
    per_mul_time_mkl = 1.0 * (end - start) / CLOCKS_PER_SEC / TIMES;
  }

  // The quantized version of C is never explicity created. We de-quantize on the fly
  // to avoid extraneous memory accesses.
  float * fast_C = new float[num_A_rows*num_B_rows];
  memset(fast_C, 0, sizeof(float)*num_A_rows*num_B_rows);

  // Each __m128i fits 8 16-bit integers, so we assume the width is a multiple of 8.
  // We could pad with 0 in the general case.
  __m256i * quant_A = (__m256i*)_aligned_malloc(sizeof(__m256i) * num_A_rows*width / 16, 32);
  __m256i * quant_B = (__m256i*)_aligned_malloc(sizeof(__m256i) *num_B_rows*width / 16, 32);

  // We quantize with 10 bits of precision. This works well "universally". 
  // See the top of this file for more info on why.
  //double quant_mult = pow(2.0, 10.0);
  double quant_mult = 1000.0;

  // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
  // So we must divide by 1.0/(n^2) to get back the original value.
  double unquant_mult = 1.0 / (quant_mult*quant_mult);

  // The weight matrix should be quantized before starting decoding, since it is known beforehand.
  avx256::QuantizedMatrix::Quantize(B, quant_B, (float)quant_mult, num_B_rows, width);

  // The activation matrix must be quantized on-the-fly.
  
  double per_mul_time_fast = 0;
  {
    int TIMES = 10000 / DIV;
    clock_t start = clock();
    for (int i = 0; i < TIMES; i++)
    {
      avx256::QuantizedMatrix::Quantize(A, quant_A, (float)quant_mult, num_A_rows, width);
      avx256::QuantizedMatrix::MatrixMult(quant_A, quant_B, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
    }

    clock_t end = clock();
    per_mul_time_fast = 1.0 * (end - start) / CLOCKS_PER_SEC / TIMES;
  }

  printf("Common %lg MKL: %lg, AVX2(16bit)%lg Speed X: %g\n", per_mul_time_slow, per_mul_time_mkl, per_mul_time_fast, per_mul_time_mkl / per_mul_time_fast);

  double max_diff = 0.0;
  double mean_diff = 0.0;
  for (int i = 0; i < num_A_rows; i++) {
    for (int j = 0; j < num_B_rows; j++) {
      float r = ref_C[i*num_B_rows + j];
      float f = fast_C[i*num_B_rows + j];
      double diff = fabs(r - f);
      if (diff > max_diff) {
        max_diff = diff;
      }
      mean_diff += diff;
    }
  }

  mean_diff /= (double)num_A_rows*(double)num_B_rows;

  printf("Diff between 32-bit float and 16-bit integer:\n");
  printf("  Mean = %g\n", mean_diff);
  printf("  Max = %g\n", max_diff);

  return 0;
}

int sse_test() {
  srand(45678);

  // A is usually an activation matrix, B is usually a weight matrix.
  // We actually comptue A * B^T. num_B_rows is the rows in B^T. 
  int num_A_rows = 4;
  int num_B_rows = 1024;
  // This is the shared dimension.
  int width = 1024;

  printf("Computing matrix multiplication: %d x %d x %d\n", num_A_rows, width, num_B_rows);

  //assert(num_A_rows % 4 == 0);
  //assert(width % 8 == 0);



  float * A = new float[num_A_rows*width];
  float * B = new float[num_B_rows*width];

  for (int i = 0; i < num_A_rows*width; i++) {
    A[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
  }

  for (int i = 0; i < num_B_rows*width; i++) {
    B[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
  }

  // C will thus be num_A_rows x num_B_rows
  float * ref_C = new float[num_A_rows*num_B_rows];
  double per_mul_time_slow = 0;
  memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);
  {
    int TIMES = 100;
    clock_t start = clock();
    for (int i = 0; i < TIMES; i++)
      SlowRef_MatrixMult(A, B, /* out */ ref_C, num_A_rows, num_B_rows, width);

    clock_t end = clock();
    per_mul_time_slow = 1.0 * (end - start) / CLOCKS_PER_SEC / TIMES;
  }

  double per_mul_time_mkl = 0;
  memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);
  {
    int TIMES = 100;
    clock_t start = clock();
    for (int i = 0; i < TIMES; i++)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_A_rows, width, num_B_rows, 1, A, width, B, width, 0, ref_C, num_B_rows);

    clock_t end = clock();
    per_mul_time_mkl = 1.0 * (end - start) / CLOCKS_PER_SEC / TIMES;
  }


  // The quantized version of C is never explicity created. We de-quantize on the fly
  // to avoid extraneous memory accesses.
  float * fast_C = new float[num_A_rows*num_B_rows];
  memset(fast_C, 0, sizeof(float)*num_A_rows*num_B_rows);

  // Each __m128i fits 8 16-bit integers, so we assume the width is a multiple of 8.
  // We could pad with 0 in the general case.
  __m128i * quant_A = (__m128i*)_aligned_malloc(sizeof(__m128i) * num_A_rows*width / 8, 16);
  __m128i * quant_B = (__m128i*)_aligned_malloc(sizeof(__m128i) *num_B_rows*width / 8, 16);

  // We quantize with 10 bits of precision. This works well "universally". 
  // See the top of this file for more info on why.
  //double quant_mult = pow(2.0, 10.0);
  double quant_mult = 1000.0;

  // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
  // So we must divide by 1.0/(n^2) to get back the original value.
  double unquant_mult = 1.0 / (quant_mult*quant_mult);

  // The weight matrix should be quantized before starting decoding, since it is known beforehand.
  Quantize(B, quant_B, (float)quant_mult, num_B_rows, width);

  // The activation matrix must be quantized on-the-fly.
  Quantize(A, quant_A, (float)quant_mult, num_A_rows, width);
  double per_mul_time_fast = 0;
  {
    int TIMES = 10000;
    clock_t start = clock();
    for (int i = 0; i < TIMES; i++)
      SSE_MatrixMult(quant_A, quant_B, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);

    clock_t end = clock();
    per_mul_time_fast = 1.0 * (end - start) / CLOCKS_PER_SEC / TIMES;
  }

  printf("Common: %lg MKL: %lg SSE(16 bit): %lg X: %g\n", per_mul_time_slow, per_mul_time_mkl, per_mul_time_fast, per_mul_time_mkl / per_mul_time_fast);

  double max_diff = 0.0;
  double mean_diff = 0.0;
  for (int i = 0; i < num_A_rows; i++) {
    for (int j = 0; j < num_B_rows; j++) {
      float r = ref_C[i*num_B_rows + j];
      float f = fast_C[i*num_B_rows + j];
      double diff = fabs(r - f);
      if (diff > max_diff) {
        max_diff = diff;
      }
      mean_diff += diff;
    }
  }

  mean_diff /= (double)num_A_rows*(double)num_B_rows;

  printf("Diff between 32-bit float and 16-bit integer:\n");
  printf("  Mean = %g\n", mean_diff);
  printf("  Max = %g\n", max_diff);

  return 0;
}

void TestHADD4() {
  __m256i test[4];
  for (int i = 1; i <= 4; i++) {
    test[i - 1] = _mm256_set_epi32(1 * i, 2 * i, 3 * i, 4 * i, 5 * i, i * 6, i * 7, i * 8);
  }

  __m128i sum;
  avx256::HADD4(test[0], test[1], test[2], test[3], sum);
  printf("%d %d %d %d\n", sum.m128i_i32[0], sum.m128i_i32[1], sum.m128i_i32[2], sum.m128i_i32[3]);
}

// Program takes no input
int main(int argc, char ** argv) {
  avx2_test();
  sse_test();
  return 0;
  int num_rows = 4;
  int width = 32;
  float * test_1 = new float[num_rows * width];
  float * test_2 = new float[num_rows * width];
  for (int i = 0; i < num_rows; i++)
  {
    for (int j = 0; j < width; j++) {
      auto v = 0.001f * (i + 1) * (j + 1);
      test_1[i * width + j] = v;
    }
  }

  for (int i = 0; i < 100; i++) {
    printf("%f ", test_1[i]);
  }
  
  printf("\n");

  __m256i * quant_A = (__m256i*)_aligned_malloc(sizeof(__m256i) * num_rows*width / 16, 32);

  avx256::QuantizedMatrix::Quantize(test_1, quant_A, 1000.0f, num_rows, width);

  avx256::QuantizedMatrix::Unquantize(quant_A, test_2, 1000.0f, num_rows, width);

  for (int i = 0; i < 100; i++) {
    if (test_1[i] - test_2[i] > 0.000001)
      printf("%f ", test_2[i]);
  }

  printf("\n");

  TestHADD4();

  return 0;
}
