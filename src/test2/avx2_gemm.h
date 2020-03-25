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

#include "avx_common.h"

namespace avx256 {

    void Unquantize(const float * input, __m256i * output, float quant_mult, int num_rows, int width) {
    }

    class QuantizedMatrix16 {
        QuantizedMatrix16(const QuantizedMatrix16&);
        public:
        QuantizedMatrix16() {}
        ~QuantizedMatrix16() {}

        void init(float quant_mult, int num_rows, int width) {
            init(nullptr, quant_mult, num_rows, width);
        }

        void init(const float * input, float quant_mult, int num_rows, int width) {
            M_ = num_rows;
            N_ = width;
            quant_mult_ = quant_mult;
            if (quantized_) {
                free(quantized_);
                quantized_ = nullptr;
            }

            quantized_ = (__m256i*)aligned_alloc(32, sizeof(__m256i) * M_ * N_ / 16);
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

        static void MatrixMult2(const __m256i * A, const __m256i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {

            const int sse_width = width / 16;
            __m128 unquant_mults = _mm_set1_ps(unquant_mult);

            memset(C, 0, sizeof(float)*num_A_rows*num_B_rows);

            for (int j = 0; j < num_B_rows; j += 4) {
                const __m256i * B1_row = B + j * sse_width;
                const __m256i * B2_row = B1_row + sse_width;
                const __m256i * B3_row = B2_row + sse_width;
                const __m256i * B4_row = B3_row + sse_width;

                for (int i = 0; i < num_A_rows; i++) {
                    const __m256i * A_row = A + i * sse_width;

                    __m256i sum1 = _mm256_setzero_si256();
                    __m256i sum2 = _mm256_setzero_si256();
                    __m256i sum3 = _mm256_setzero_si256();
                    __m256i sum4 = _mm256_setzero_si256();

                    /*
                       why unrolling 4*4? That is because _mm256_madd_epi16's latency is 5, unroll 4*4 only waste 1 cycles.
                       */
#define UNROLL_K__INT16__  4

                    for (int k = 0; k < sse_width; k+= UNROLL_K__INT16__) {
                        const __m256i& a = *(A_row + k);
                        const __m256i& b1 = *(B1_row + k);
                        const __m256i& b2 = *(B2_row + k);
                        const __m256i& b3 = *(B3_row + k);
                        const __m256i& b4 = *(B4_row + k);

#if UNROLL_K__INT16__ == 4
                        const __m256i& a_1 = *(A_row + k + 1);
                        const __m256i& a_2 = *(A_row + k + 2);
                        const __m256i& a_3 = *(A_row + k + 3);

                        const __m256i& b1_1 = *(B1_row + k + 1);
                        const __m256i& b1_2 = *(B1_row + k + 2);
                        const __m256i& b1_3 = *(B1_row + k + 3);

                        const __m256i& b2_1 = *(B2_row + k + 1);
                        const __m256i& b2_2 = *(B2_row + k + 2);
                        const __m256i& b2_3 = *(B2_row + k + 3);

                        const __m256i& b3_1 = *(B3_row + k + 1);
                        const __m256i& b3_2 = *(B3_row + k + 2);
                        const __m256i& b3_3 = *(B3_row + k + 3);

                        const __m256i& b4_1 = *(B4_row + k + 1);
                        const __m256i& b4_2 = *(B4_row + k + 2);
                        const __m256i& b4_3 = *(B4_row + k + 3);
#endif
                        __m256i b1xa = _mm256_madd_epi16(b1, a);
                        __m256i b2xa = _mm256_madd_epi16(b2, a);
                        __m256i b3xa = _mm256_madd_epi16(b3, a);
                        __m256i b4xa = _mm256_madd_epi16(b4, a);

#if UNROLL_K__INT16__ == 4
                        __m256i b1_1xa_1 = _mm256_madd_epi16(b1_1, a_1);
                        __m256i b2_1xa_1 = _mm256_madd_epi16(b2_1, a_1);
                        __m256i b3_1xa_1 = _mm256_madd_epi16(b3_1, a_1);
                        __m256i b4_1xa_1 = _mm256_madd_epi16(b4_1, a_1);

                        __m256i b1_2xa_2 = _mm256_madd_epi16(b1_2, a_2);
                        __m256i b2_2xa_2 = _mm256_madd_epi16(b2_2, a_2);
                        __m256i b3_2xa_2 = _mm256_madd_epi16(b3_2, a_2);
                        __m256i b4_2xa_2 = _mm256_madd_epi16(b4_2, a_2);

                        __m256i b1_3xa_3 = _mm256_madd_epi16(b1_3, a_3);
                        __m256i b2_3xa_3 = _mm256_madd_epi16(b2_3, a_3);
                        __m256i b3_3xa_3 = _mm256_madd_epi16(b3_3, a_3);
                        __m256i b4_3xa_3 = _mm256_madd_epi16(b4_3, a_3);

                        sum1 = _mm256_add_epi32(sum1, b1_1xa_1);
                        sum2 = _mm256_add_epi32(sum2, b2_1xa_1);
                        sum3 = _mm256_add_epi32(sum3, b3_1xa_1);
                        sum4 = _mm256_add_epi32(sum4, b4_1xa_1);

                        sum1 = _mm256_add_epi32(sum1, b1_2xa_2);
                        sum2 = _mm256_add_epi32(sum2, b2_2xa_2);
                        sum3 = _mm256_add_epi32(sum3, b3_2xa_2);
                        sum4 = _mm256_add_epi32(sum4, b4_2xa_2);

                        sum1 = _mm256_add_epi32(sum1, b1_3xa_3);
                        sum2 = _mm256_add_epi32(sum2, b2_3xa_3);
                        sum3 = _mm256_add_epi32(sum3, b3_3xa_3);
                        sum4 = _mm256_add_epi32(sum4, b4_3xa_3);
#endif
                        sum1 = _mm256_add_epi32(sum1, b1xa);
                        sum2 = _mm256_add_epi32(sum2, b2xa);
                        sum3 = _mm256_add_epi32(sum3, b3xa);
                        sum4 = _mm256_add_epi32(sum4, b4xa);
                    }

                    __m128i sum;
                    HADD4(sum1, sum2, sum3, sum4, sum);
                    __m128 f_sum = _mm_cvtepi32_ps(sum);

#if 1
                    float * C1 = C + i*num_B_rows + j;
                    f_sum = _mm_mul_ps(f_sum, unquant_mults);
                    f_sum = _mm_add_ps(f_sum, _mm_load_ps(C1));
                    _mm_store_ps(C1, f_sum);
#else

                    float * C1 = C + i*num_B_rows + j;
                    float * C2 = C + i*num_B_rows + j+1;
                    float * C3 = C + i*num_B_rows + j+2;
                    float * C4 = C + i*num_B_rows + j+3;

                    _mm_storeu_ps(extractSum, f_sum);

                    *(C1) += extractSum[0] * unquant_mult;
                    *(C2) += extractSum[1] * unquant_mult;
                    *(C3) += extractSum[2] * unquant_mult;
                    *(C4) += extractSum[3] * unquant_mult;
#endif
                }
            }
        }

        static void MatrixMult(const __m256i * A, const __m256i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {

            assert(num_A_rows % 4 == 0);
            assert(width % 16 == 0);

            int sse_width = width / 16;
            __m128 unquant_mults = _mm_set1_ps(unquant_mult);

            float extractSum[4];

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
                        const __m256i& b = *(B_row + k);

                        const __m256i& a1 = *(A1_row + k);
                        const __m256i& a2 = *(A2_row + k);
                        const __m256i& a3 = *(A3_row + k);
                        const __m256i& a4 = *(A4_row + k);

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
                    f_sum = _mm_mul_ps(f_sum, unquant_mults);

                    _mm_storeu_ps(extractSum, f_sum);
                    *(C1) = extractSum[0];
                    *(C2) = extractSum[1];
                    *(C3) = extractSum[2];
                    *(C4) = extractSum[3];
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
