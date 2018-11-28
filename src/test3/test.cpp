#include <iostream>
#include <ctime>
#include <sys/time.h>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <malloc.h>

#include "avx2_gemm.h"

#include <mkl.h>

using namespace std;
using namespace avx256;


inline long getTimeTT()
{
    struct timeval iTime;
    gettimeofday(&iTime, NULL);
    long lTime = ((long) iTime.tv_sec) * 1000000 + (long) iTime.tv_usec;
    return lTime;
}

union combine {
  short x[sizeof(__m256i)/sizeof(short)];
  __m256i y;
};

void SlowRef_MatrixMult(const float * A, const float * B, float * C, int num_A_rows, int num_B_rows, int width, bool transB = true)
{
    if (transB) {
        for (int i = 0; i < num_A_rows; i++) {
            const float * A_row = A + i*width;
            float * C_row = C + i*num_B_rows;
            for (int j = 0; j < num_B_rows; j++) {
                const float * B_row = B + j*width;
                float sum = 0.0f;
                for (int k = 0; k < width; k++) {
                    sum += A_row[k]*B_row[k];
                }
                C_row[j] = sum;
            }
        }
    }
    else {
        for (int i = 0; i < num_A_rows; i++) {
            const float * A_row = A + i*width;
            float * C_row = C + i*num_B_rows;
            for (int j = 0; j < num_B_rows; j++)
                C_row[j] = 0;
            for (int k = 0; k < width; k++) {
                const float * B_row = B + k*num_B_rows;
                for (int j = 0; j < num_B_rows; j++) {
                    C_row[j] += A_row[k]* B_row[j];
                }
            }
        }
    }
}
int main(int argc, char *argv[])
{

    int system = 0;
    if (argc > 1) {
        system = atoi(argv[1]);
    }
    int loopNum = 100000;

    int num_A_rows = 4;
    int num_B_rows = 1024;
    int width = 1024;

    void* raw( nullptr );
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B = reinterpret_cast<float*>( raw );
    float * A = new float[num_A_rows*width];


    srand(456789);
    for (int i = 0; i < num_A_rows*width; i++) {
        // A[i] = i+1;
        A[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        // std::cerr << "A[" << i << "] = " << A[i] << std::endl;
    }

    for (int i = 0; i < num_B_rows*width; i++) {
        // B[i] = i+1;
        // B[i] = (i % 4) * 3 + i / 4 + 1;
        B[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        // std::cerr << "B[" << i << "] = " << B[i] << std::endl;
    }




    float * fast_C;
    if (0 == system || 1 == system) {

        long start = getTimeTT();
        // The quantized version of C is never explicity created. We de-quantize on the fly
        // to avoid extraneous memory accesses.
        fast_C = new float[num_A_rows*num_B_rows];
        memset(fast_C, 0, sizeof(float)*num_A_rows*num_B_rows);

        // Each __m128i fits 8 16-bit integers, so we assume the width is a multiple of 8.
        // We could pad with 0 in the general case.
        __m256i * quant_A = (__m256i*)aligned_alloc(32, sizeof(__m256i) * num_A_rows*width / 16);
        __m256i * quant_B = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);

        // We quantize with 10 bits of precision. This works well "universally".
        // See the top of this file for more info on why.
        //double quant_mult = pow(2.0, 10.0);
        double quant_mult = 1024.0;

        // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
        // So we must divide by 1.0/(n^2) to get back the original value.
        double unquant_mult = 1.0 / (quant_mult*quant_mult);

        // The weight matrix should be quantized before starting decoding, since it is known beforehand.
        avx256::QuantizedMatrix16::Quantize(B, quant_B, (float)quant_mult, num_B_rows, width);
        /*
            combine tt;

            printf("B ------------------\n");
            for (int i = 0; i < num_B_rows; i++) {
                for (int j = 0; j < width; j++) {
                    tt.y = *(quant_B + i * width /16 + j/16);
                    printf("(%d, %d): %g %d\t", i, j, B[i*width+j], tt.x[j%16]);
                }
                printf("\n");
            }
        */
        // The activation matrix must be quantized on-the-fly.
        for (int i = 0; i < loopNum; i++)
        {
            avx256::QuantizedMatrix16::Quantize(A, quant_A, (float)quant_mult, num_A_rows, width);
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
        }

        long cost = (getTimeTT() - start) / 1000; //ms

        cout << "avx2 cost time:  " << cost << "ms, flops: " << 1.0 * loopNum * num_A_rows * num_B_rows * width / cost / 1000000 << " GFLOPS" << endl;

       // for (int i = 0; i < num_A_rows; i++) {
       //     for (int j = 0; j < num_B_rows; j++) {
       //         std::cerr << "fast_C[" << i << "][" << j << "]= " << fast_C[i * num_B_rows + j] << std::endl;
       //     }
       // }
    }


    // // MKL
    float * ref_C = new float[num_A_rows*num_B_rows];
    if (0 == system || 2 == system) {

        // C will thus be num_A_rows x num_B_rows

        memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);

        long start = getTimeTT();

       for (int i = 0; i < loopNum; i++) {
            // std::cerr << CblasRowMajor << ", " << CblasNoTrans << ", " << CblasNoTrans << ", " << num_A_rows << ", " << num_B_rows << ", " << width << ", " << 1 << ", " << width << ", " << num_B_rows << ", " << num_B_rows << std::endl;

            long begin1 = getTimeTT();
            // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_A_rows, num_B_rows, width, 1, A, width, B, num_B_rows, 0, ref_C, num_B_rows);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_A_rows, num_B_rows, width, 1, A, width, B, width, 0, ref_C, num_B_rows);
            long end1 = getTimeTT();

        }

        long cost = (getTimeTT() - start) / 1000; //ms

        cout << "mkl  cost time: " << cost << "ms, flops: " << 1.0 * loopNum * num_A_rows * num_B_rows * width / cost / 1000000 << " GFLOPS" << endl;

        // for (int i = 0; i < num_A_rows; i++) {
        //     for (int j = 0; j < num_B_rows; j++) {
        //         std::cerr << "ref_C[" << i << "][" << j << "]= " << ref_C[i * num_B_rows + j] << std::endl;
        //     }
        // }
    }

    return 0;
}

