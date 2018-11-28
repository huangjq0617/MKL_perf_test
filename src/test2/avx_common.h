#pragma once
#include <immintrin.h>

namespace avx256 {
  const int PerfInt16OneVector = (int)(sizeof(__m256i) / sizeof(__int16_t));
  const int PerfInt8OneVector = (int)(sizeof(__m256i) / sizeof(__int8_t));
  const int PerfFloatOneVector = (int)(sizeof(__m256i) / sizeof(float));

#define AVX256_256i_LOW(T, m256) *((T*)&(m256))
#define AVX256_256i_HIGH(T, m256) *( ((T*)&(m256)) + 1 )

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
    sum = _mm_add_epi32(first, *(((__m128i*)&R_hadd_S_Q_hadd_P) + 1));
  }
}
