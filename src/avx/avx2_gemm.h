#pragma once
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


namespace avx256 {
  const int PerfInt16OneVector = (int)(sizeof(__m256i) / sizeof(__int16));
  const int PerfFloatOneVector = (int)(sizeof(__m256i) / sizeof(float));
  
  // We now have each sum spread across 4 32-bit ints in SSE register, e.g., 
  // R = [r0, r1, r2, r3, r4, r5, r6, r7]. We need to compute r0 + r1 + r2 + r3 + r4 + ... + r7.
  // S = [s0, s1, s2, s3, s4, s5, s6, s7].
  // Q = [q0, q1, q2, q3, q4, q5, q6, q7].
  // P = [p0, p1, p2, p3, p4, p5, p6, p7].

  // _mm256_hadd_epi32(R, S)
  // R_hadd_S = [r0 + r1, r2 + r3, s0 + s1, s2 + s3, r4 + r5, r6 + r7, s4 + s5, s6 + s7]
  // _mm256_hadd_epi32(Q, P)
  // Q_hadd_P = [q0 + q1, q2 + q3, p0 + p1, p2 + p3, q4 + q5, q6 + q7, p4 + p5, p6 + p7]
  // _mm256_hadd_epi32(R_hadd_S, Q_hadd_P)
  // R_hadd_S_Q_hadd_P = [r0 + r1 + r2 + r3, s0 + s1 + s2 + s3, q0 + q1 + q2 + q3, p0 + p1+  p2 + p3, 
  //                      r4 + r5 + r6 + r7, s4 + s5 + s6 + s7, q4 + q5 + q6 + q7, p4 + p5 + p6 + p7]
  // finally, _mm256_add_epi32(R_hadd_S, Q_hadd_P)

  inline void HADD4(const __m256i& R, __m256i S, const __m256i& Q, const __m256i& P, __m128i& sum) {
    __m256i R_hadd_S = _mm256_hadd_epi32(R, S);
    __m256i Q_hadd_P = _mm256_hadd_epi32(Q, P);
    __m256i R_hadd_S_Q_hadd_P = _mm256_hadd_epi32(R_hadd_S, Q_hadd_P);
    __m128i first = _mm256_castsi256_si128(R_hadd_S_Q_hadd_P);
    sum = _mm_add_epi32(first, *(((__m128i*)&R_hadd_S_Q_hadd_P) + 1) );
  }

  void Unquantize(const float * input, __m256i * output, float quant_mult, int num_rows, int width) {
  }

  class QuantizedMatrix {
    QuantizedMatrix(const QuantizedMatrix&);
  public:
    QuantizedMatrix() {}
    ~QuantizedMatrix() {}

    void init(float quant_mult, int num_rows, int width) {
      init(nullptr, quant_mult, num_rows, width);
    }

    void init(const float * input, float quant_mult, int num_rows, int width) {
      M_ = num_rows;
      N_ = width;
      quant_mult_ = quant_mult;
      if (quantized_) {
        _aligned_free(quantized_);
        quantized_ = nullptr;
      }

      quantized_ = (__m256i*)_aligned_malloc(sizeof(__m256i) * M_ * N_ / 16, 32);
      if (input) {
        Quantize(input, quantized_, quant_mult, num_rows, width);
      }
      else {

      }
    }

    float quant_mult() const {
      return quant_mult_;
    }

    int M() const {
      return M_;
    }

    int N() const {
      return N_;
    }

    int MN() const {
      return M_ * N_;
    } 

    void Mult(const QuantizedMatrix* B, QuantizedMatrix* C) {
    }

    static void MatrixMult(const __m256i * A, const __m256i * B, __m256i * C, int num_A_rows, int num_B_rows, int width) {

    }

    static void MatrixMult(const __m256i * A, const __m256i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {

      assert(num_A_rows % 4 == 0);
      assert(width % 8 == 0);

      int sse_width = width / 16;

      // We do loop unrolling over A. This is *significantly* faster
      // since B can live in the registers. We are assuming that
      // A is a multiple of 4, but we can add extra code to handle values of 1, 2, 3.
      //
      // We could also do loop unrolling over B, which adds some additional speedup.
      // We don't do that for the sake of clarity.
      // 
      // There are other memory access patterns we could do, e.g., put B on the outer loop.
      // The justification is that A is typically small enough that it can live in L1 cache.
      // B is usually a larger weight matrix, so it might not be able to. However, we are using
      // each element of B four times while it's still in a register, so caching is not as important.
      for (int i = 0; i < num_A_rows; i += 4) {
        const __m256i * A1_row = A + (i + 0)*sse_width;
        const __m256i * A2_row = A + (i + 1)*sse_width;
        const __m256i * A3_row = A + (i + 2)*sse_width;
        const __m256i * A4_row = A + (i + 3)*sse_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m256i * B_row = B + j*sse_width;

          __m256i sum1 = _mm256_setzero_si256();
          __m256i sum2 = _mm256_setzero_si256();
          __m256i sum3 = _mm256_setzero_si256();
          __m256i sum4 = _mm256_setzero_si256();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < sse_width; k++) {
            __m256i b = *(B_row + k);

            __m256i a1 = *(A1_row + k);
            __m256i a2 = *(A2_row + k);
            __m256i a3 = *(A3_row + k);
            __m256i a4 = *(A4_row + k);

            // _mm_madd_epi16 does multiply add on 8 16-bit integers and accumulates into a four 32-bit register.
            // E.g.,
            // a1 = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fa, fb, fc, fd, fe, ff] (16-bit ints)
            // b1 = [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, ha, hb, hc, hd, he, hf] (16-bit ints)
            // result = [f0*h0 + f1*h1, f2*h2+f3*h3,..., fe*he, + ff* hf] (32-bit ints)
            // Then _mm256_add_epi32 just effectively does a += on these 32-bit integers.
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(b, a3));
            sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(b, a4));
          }
          
          

          __m128i sum;
          HADD4(sum1, sum2, sum3, sum4, sum);

          float * C1 = C + (i + 0)*num_B_rows + j;
          float * C2 = C + (i + 1)*num_B_rows + j;
          float * C3 = C + (i + 2)*num_B_rows + j;
          float * C4 = C + (i + 3)*num_B_rows + j;

          __m128 f_sum = _mm_cvtepi32_ps(sum);
          
          *(C1) = f_sum.m128_f32[0] * unquant_mult;
          *(C2) = f_sum.m128_f32[1] * unquant_mult;
          *(C3) = f_sum.m128_f32[2] * unquant_mult;
          *(C4) = f_sum.m128_f32[3] * unquant_mult;
        }
      }
    }

    static int martix_index_to_m256(int n) {
      /*
      dst[15:0] := Saturate_Int32_To_Int16 (a[31:0]) #0 0
      dst[31:16] := Saturate_Int32_To_Int16 (a[63:32]) # 1 1
      dst[47:32] := Saturate_Int32_To_Int16 (a[95:64]) # 2 2
      dst[63:48] := Saturate_Int32_To_Int16 (a[127:96]) # 3 3
      dst[79:64] := Saturate_Int32_To_Int16 (b[31:0]) # 8 4
      dst[95:80] := Saturate_Int32_To_Int16 (b[63:32]) # 9 5
      dst[111:96] := Saturate_Int32_To_Int16 (b[95:64]) # 10 6
      dst[127:112] := Saturate_Int32_To_Int16 (b[127:96]) # 11 7
      dst[143:128] := Saturate_Int32_To_Int16 (a[159:128]) # 4 8
      dst[159:144] := Saturate_Int32_To_Int16 (a[191:160]) # 5 9
      dst[175:160] := Saturate_Int32_To_Int16 (a[223:192]) # 6 10
      dst[191:176] := Saturate_Int32_To_Int16 (a[255:224]) # 7 11
      dst[207:192] := Saturate_Int32_To_Int16 (b[159:128]) # 12 12
      dst[223:208] := Saturate_Int32_To_Int16 (b[191:160]) # 13 13
      dst[239:224] := Saturate_Int32_To_Int16 (b[223:192]) # 14 14
      dst[255:240] := Saturate_Int32_To_Int16 (b[255:224]) # 15 15
      */
      static const int g_indexs[] = { 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};
      return g_indexs[n];
    }

    static void Unquantize(const __m256i * input, float * output, float quant_mult, int num_rows, int width)
    {
      quant_mult = 1.0f / quant_mult;
      __m256 m_quant_mult = _mm256_set_ps(quant_mult, quant_mult, quant_mult, quant_mult, quant_mult, quant_mult, quant_mult, quant_mult);

      const int num_output_chunks = width / PerfInt16OneVector;
      for (int i = 0; i < num_rows; i++) {
        float * output_row = output + i * width;
        const __m256i * input_row = input + i * num_output_chunks;
        for (int j = 0; j < num_output_chunks; j++) {
          float * x = output_row + j * PerfInt16OneVector;
          const __m256i * y = input_row + j;
          const __m128i* a1 = (const __m128i*)y;
          const __m128i* a2 = a1 + 1;
          __m256i a32_1 = _mm256_cvtepu16_epi32(*a1);
          __m256i a32_2 = _mm256_cvtepu16_epi32(*a2);
          __m256 f_1 =  _mm256_cvtepi32_ps(a32_1);
          __m256 f_2 = _mm256_cvtepi32_ps(a32_2);
          f_1 = _mm256_mul_ps(f_1, m_quant_mult);
          f_2 = _mm256_mul_ps(f_2, m_quant_mult);

          const __m128* f_128_1 = (const __m128*)&f_1;
          const __m128* f_128_2 = (const __m128*)&f_2;
          _mm_store_ps(&x[0], *f_128_1);
          _mm_store_ps(&x[8], *(f_128_1 + 1));
          _mm_store_ps(&x[4], *f_128_2);
          _mm_store_ps(&x[12], *(f_128_2 + 1));

          /*
          x[0] = f_1.m256_f32[0];
          x[1] = f_1.m256_f32[1];
          x[2] = f_1.m256_f32[2];
          x[3] = f_1.m256_f32[3];
          x[8] = f_1.m256_f32[4];
          x[9] = f_1.m256_f32[5];
          x[10] = f_1.m256_f32[6];
          x[11] = f_1.m256_f32[7];

          x[4] = f_2.m256_f32[0];
          x[5] = f_2.m256_f32[1];
          x[6] = f_2.m256_f32[2];
          x[7] = f_2.m256_f32[3];
          x[12] = f_2.m256_f32[4];
          x[13] = f_2.m256_f32[5];
          x[14] = f_2.m256_f32[6];
          x[15] = f_2.m256_f32[7];
          */
        }
      }
    }

    static void Quantize(const float * input, __m256i * output, float quant_mult, int num_rows, int width) {
      //static_assert(PerfSize == 16);
      assert(width % PerfInt16OneVector == 0);
      assert(PerfInt16OneVector == 16);

      const int num_input_chunks = width / PerfInt16OneVector;

      __m256 sse_quant_mult = _mm256_set_ps(quant_mult, quant_mult, quant_mult, quant_mult, quant_mult, quant_mult, quant_mult, quant_mult);

      for (int i = 0; i < num_rows; i++) {
        const float * input_row = input + i * width;
        __m256i * output_row = output + i * num_input_chunks;
        for (int j = 0; j < num_input_chunks; j++) {
          const float * x = input_row + j * PerfInt16OneVector;
          // Process 16 floats at once, since each __m256i can contain 16 16-bit integers.

          __m256 f_0 = _mm256_loadu_ps(x);
          __m256 f_1 = _mm256_loadu_ps(x + PerfFloatOneVector);

          // Multiply by quantization factor (e.g., if quant_mult = 1000.0, 0.34291 --> 342.21)
          __m256 m_0 = _mm256_mul_ps(f_0, sse_quant_mult);
          __m256 m_1 = _mm256_mul_ps(f_1, sse_quant_mult);

          // Cast float to 32-bit int (e.g., 342.21 --> 342)
          __m256i i_0 = _mm256_cvtps_epi32(m_0);
          __m256i i_1 = _mm256_cvtps_epi32(m_1);

          // Cast 32-bit int to 16-bit int. You must ensure that these fit into the 16-bit range
          // by clipping values during training.
          /*
          dst[15:0] := Saturate_Int32_To_Int16 (a[31:0]) #0
          dst[31:16] := Saturate_Int32_To_Int16 (a[63:32]) # 1
          dst[47:32] := Saturate_Int32_To_Int16 (a[95:64]) # 2
          dst[63:48] := Saturate_Int32_To_Int16 (a[127:96]) # 3
          dst[79:64] := Saturate_Int32_To_Int16 (b[31:0]) # 8
          dst[95:80] := Saturate_Int32_To_Int16 (b[63:32]) # 9
          dst[111:96] := Saturate_Int32_To_Int16 (b[95:64]) # 10
          dst[127:112] := Saturate_Int32_To_Int16 (b[127:96]) # 11
          dst[143:128] := Saturate_Int32_To_Int16 (a[159:128]) # 4 
          dst[159:144] := Saturate_Int32_To_Int16 (a[191:160]) # 5
          dst[175:160] := Saturate_Int32_To_Int16 (a[223:192]) # 6
          dst[191:176] := Saturate_Int32_To_Int16 (a[255:224]) # 7
          dst[207:192] := Saturate_Int32_To_Int16 (b[159:128]) # 12
          dst[223:208] := Saturate_Int32_To_Int16 (b[191:160]) # 13
          dst[239:224] := Saturate_Int32_To_Int16 (b[223:192]) # 14
          dst[255:240] := Saturate_Int32_To_Int16 (b[255:224]) # 15
          */
          *(output_row + j) = _mm256_packs_epi32(i_0, i_1);
        }
      }
    }
  private:
    int M_ = 0;
    int N_ = 0;
    float quant_mult_ = 1.0f;
    __m256i* quantized_ = nullptr;
  };

  
}