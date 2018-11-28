#include <iostream>
#include <ctime>
#include <sys/time.h>

#include <blaze/Math.h>

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <malloc.h>

#include "avx2_gemm.h"

// #include <mkl.h>

using namespace std;
using namespace avx256;

typedef blaze::CustomMatrix<float, blaze::unaligned,
                blaze::unpadded, blaze::rowMajor> BlazeWrapper;

typedef blaze::DynamicMatrix<float, blaze::rowMajor> Matrix;


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

void TransposeMatrix(const float *in, float *out, int orig_rows, int orig_cols)
{
    for (int i=0; i < orig_rows; i++) {
        for (int j=0; j < orig_cols; j++) {
            out[j * orig_rows + i] = in[i * orig_cols + j];
        }
    }
}


#define MATRIX_B_NUM    1

int main(int argc, char *argv[])
{

    int system = 0;
    int loopNum = 40000;

    int num_A_rows = 7;
    int num_B_rows = 1280;
    int width = 1280;

    if (argc > 1) {
        system = atoi(argv[1]);
    }

    if (argc == 6) {
        num_A_rows = atoi(argv[2]);
        num_B_rows = atoi(argv[3]);
        width = atoi(argv[4]);
        loopNum = atoi(argv[5]);
    }

    cout << "M=" << num_A_rows << ", N=" << num_B_rows << ", K=" << width << ", MATRIX_B_NUM=" << MATRIX_B_NUM << ", loopNum=" << loopNum << endl;

    void* raw( nullptr );
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B = reinterpret_cast<float*>( raw );

#if MATRIX_B_NUM > 1
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B2 = reinterpret_cast<float*>( raw );

#if MATRIX_B_NUM > 2
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B3 = reinterpret_cast<float*>( raw );

    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B4 = reinterpret_cast<float*>( raw );

#if MATRIX_B_NUM > 4
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B5 = reinterpret_cast<float*>( raw );

    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B6 = reinterpret_cast<float*>( raw );

    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B7 = reinterpret_cast<float*>( raw );

    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B8 = reinterpret_cast<float*>( raw );

#endif // > 4

#endif // > 3

#endif // > 1

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

#if MATRIX_B_NUM > 1
        B2[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;

#if MATRIX_B_NUM > 2
        B3[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        B4[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;

#if MATRIX_B_NUM > 4
        B5[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        B6[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        B7[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        B8[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
#endif // > 4

#endif // > 2

#endif // > 1
        // std::cerr << "B[" << i << "] = " << B[i] << std::endl;
    }


    float * fast_C;
    if (0 == system || 1 == system) {

        // The quantized version of C is never explicity created. We de-quantize on the fly
        // to avoid extraneous memory accesses.
        fast_C = new float[num_A_rows*num_B_rows];
        memset(fast_C, 0, sizeof(float)*num_A_rows*num_B_rows);

        long start = getTimeTT();

        // Each __m128i fits 8 16-bit integers, so we assume the width is a multiple of 8.
        // We could pad with 0 in the general case.
        __m256i * quant_A = (__m256i*)aligned_alloc(32, sizeof(__m256i) * num_A_rows*width / 16);
        __m256i * quant_B = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);

#if MATRIX_B_NUM > 1
        __m256i * quant_B2 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);

#if MATRIX_B_NUM > 2
        __m256i * quant_B3 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);
        __m256i * quant_B4 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);

#if MATRIX_B_NUM > 4
        __m256i * quant_B5 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);
        __m256i * quant_B6 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);
        __m256i * quant_B7 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);
        __m256i * quant_B8 = (__m256i*)aligned_alloc(32, sizeof(__m256i) *num_B_rows*width / 16);
#endif // > 4

#endif // > 2

#endif // > 1

        // We quantize with 10 bits of precision. This works well "universally".
        // See the top of this file for more info on why.
        //double quant_mult = pow(2.0, 10.0);
        double quant_mult = 1024.0;

        // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
        // So we must divide by 1.0/(n^2) to get back the original value.
        double unquant_mult = 1.0 / (quant_mult*quant_mult);

        // The weight matrix should be quantized before starting decoding, since it is known beforehand.
        avx256::QuantizedMatrix16::Quantize(B, quant_B, (float)quant_mult, num_B_rows, width);

#if MATRIX_B_NUM > 1
        avx256::QuantizedMatrix16::Quantize(B2, quant_B2, (float)quant_mult, num_B_rows, width);

#if MATRIX_B_NUM > 2
        avx256::QuantizedMatrix16::Quantize(B3, quant_B3, (float)quant_mult, num_B_rows, width);
        avx256::QuantizedMatrix16::Quantize(B4, quant_B4, (float)quant_mult, num_B_rows, width);

#if MATRIX_B_NUM > 4
        avx256::QuantizedMatrix16::Quantize(B5, quant_B5, (float)quant_mult, num_B_rows, width);
        avx256::QuantizedMatrix16::Quantize(B6, quant_B6, (float)quant_mult, num_B_rows, width);
        avx256::QuantizedMatrix16::Quantize(B7, quant_B7, (float)quant_mult, num_B_rows, width);
        avx256::QuantizedMatrix16::Quantize(B8, quant_B8, (float)quant_mult, num_B_rows, width);

#endif // > 4

#endif // > 2

#endif // > 1

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

#if MATRIX_B_NUM > 1
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B2, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);

#if MATRIX_B_NUM > 2
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B3, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B4, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);

#if MATRIX_B_NUM > 4
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B5, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B6, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B7, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
            avx256::QuantizedMatrix16::MatrixMult2(quant_A, quant_B8, fast_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
#endif // > 4

#endif // > 2

#endif // > 1
        }

        long cost = (getTimeTT() - start) / 1000; //ms

        cout << "avx2 cost time:  " << cost << "ms, flops: " << 1.0 * loopNum * num_A_rows * num_B_rows * width * MATRIX_B_NUM / cost / 1000000 << " GFLOPS" << endl;

       // for (int i = 0; i < num_A_rows; i++) {
       //     for (int j = 0; j < num_B_rows; j++) {
       //         std::cerr << "fast_C[" << i << "][" << j << "]= " << fast_C[i * num_B_rows + j] << std::endl;
       //     }
       // }
    }


    // // MKL
    float * ref_C = new float[num_A_rows*num_B_rows];

    float * B2 = new float[num_B_rows * width];
    TransposeMatrix(B, B2, width, num_B_rows);
    float * ref_C_2 = new float[num_A_rows*num_B_rows];

    if (0 == system || 2 == system) {

        // C will thus be num_A_rows x num_B_rows

        memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);

        long start1 = getTimeTT();
        for (int i = 0; i < loopNum; i++) {
            // std::cerr << CblasRowMajor << ", " << CblasNoTrans << ", " << CblasNoTrans << ", " << num_A_rows << ", " << num_B_rows << ", " << width << ", " << 1 << ", " << width << ", " << num_B_rows << ", " << num_B_rows << std::endl;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_A_rows, num_B_rows, width, 1, A, width, B, num_B_rows, 0, ref_C, num_B_rows);
        }
        long cost1 = (getTimeTT() - start1) / 1000; //ms

        long start2 = getTimeTT();
        for (int i = 0; i < loopNum; i++) {

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_A_rows, num_B_rows, width, 1, A, width, B2, width, 0, ref_C_2, num_B_rows);
        }
        long cost2 = (getTimeTT() - start2) / 1000; //ms

        cout << "mkl  cost time1: " << cost1 << "ms, flops: " << 1.0 * loopNum * num_A_rows * num_B_rows * width * MATRIX_B_NUM / cost1 / 1000000 << " GFLOPS" << endl;
        cout << "mkl  cost time2: " << cost2 << "ms, flops: " << 1.0 * loopNum * num_A_rows * num_B_rows * width * MATRIX_B_NUM / cost2 / 1000000 << " GFLOPS" << endl;

        // for (int i = 0; i < num_A_rows; i++) {
        //     for (int j = 0; j < num_B_rows; j++) {
        //         std::cerr << "ref_C[" << i << "][" << j << "]= " << ref_C[i * num_B_rows + j] << std::endl;
        //     }
        // }
    }


/*
    cout << "A data:" << endl;
    for (int i=0; i < num_A_rows; i++) {
        for (int j=0; j < width; j++) {
            cout << A[i*width + j] << ", ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "B data:" << endl;
    for (int i=0; i < width; i++) {
        for (int j=0; j < num_B_rows; j++) {
            cout << B[i*num_B_rows + j] << ", ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "C data:" << endl;
    for (int i=0; i < num_A_rows; i++) {
        for (int j=0; j < num_B_rows; j++) {
            cout << ref_C[i*num_B_rows + j] << ", ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "C2 data:" << endl;
    for (int i=0; i < num_A_rows; i++) {
        for (int j=0; j < num_B_rows; j++) {
            cout << ref_C_2[i*num_B_rows + j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
*/



    Matrix matrixC;
    if (0 == system || 3 == system) {

        long start = getTimeTT();
        Matrix matrixA = BlazeWrapper(A, num_A_rows, width);
        Matrix matrixB = blaze::trans(BlazeWrapper(B, num_B_rows, width));

#if MATRIX_B_NUM > 1
        Matrix matrixB2 = blaze::trans(BlazeWrapper(B2, num_B_rows, width));

#if MATRIX_B_NUM > 2
        Matrix matrixB3 = blaze::trans(BlazeWrapper(B3, num_B_rows, width));
        Matrix matrixB4 = blaze::trans(BlazeWrapper(B4, num_B_rows, width));

#if MATRIX_B_NUM > 4
        Matrix matrixB5 = blaze::trans(BlazeWrapper(B5, num_B_rows, width));
        Matrix matrixB6 = blaze::trans(BlazeWrapper(B6, num_B_rows, width));
        Matrix matrixB7 = blaze::trans(BlazeWrapper(B7, num_B_rows, width));
        Matrix matrixB8 = blaze::trans(BlazeWrapper(B8, num_B_rows, width));
#endif // > 4

#endif // > 2

#endif // > 1

        for (int i=0; i < loopNum; i++) {

            matrixC = matrixA * matrixB;

#if MATRIX_B_NUM > 1
            matrixC = matrixA * matrixB2;

#if MATRIX_B_NUM > 2
            matrixC = matrixA * matrixB3;
            matrixC = matrixA * matrixB4;

#if MATRIX_B_NUM > 4
            matrixC = matrixA * matrixB5;
            matrixC = matrixA * matrixB6;
            matrixC = matrixA * matrixB7;
            matrixC = matrixA * matrixB8;
#endif // > 4

#endif // > 2

#endif // > 1
        }

        long cost = (getTimeTT() - start) / 1000; //ms

        cout << "blaze cost time: " << cost << "ms, flops: " << 1.0 * loopNum * num_A_rows * num_B_rows * width * MATRIX_B_NUM / cost / 1000000 << " GFLOPS" << endl;

        // for (int i = 0; i < matrixC.rows(); i++) {
        //     for (int j = 0; j < matrixC.columns(); j++) {
        //         std::cerr << "matrixC[" << i << "][" << j << "]= " << matrixC(i, j) << std::endl;
        //     }
        // }
    }

    // All
    if (0 == system) {

        double max_diff = 0.0;
        double mean_diff = 0.0;
        for (int i = 0; i < num_A_rows; i++) {
            for (int j = 0; j < num_B_rows; j++) {
                float f = fast_C[i*num_B_rows + j];
                // float f = ref_C[i*num_B_rows + j];
                float r = matrixC(i, j);
                double diff = fabs(r-f);
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
    }

    return 0;
}

