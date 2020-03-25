#include <iostream>
#include <chrono>
#include <thread>

#include <string.h>
#include <unistd.h>
#include <mkl.h>

using namespace std;

void TransposeMatrix(const float *in, float *out, int orig_rows, int orig_cols)
{
    for (int i=0; i < orig_rows; i++) {
        for (int j=0; j < orig_cols; j++) {
            out[j * orig_rows + i] = in[i * orig_cols + j];
        }
    }
}

int main(int argc, char *argv[])
{

    int loopNum = 1000;

    int num_A_rows = 32;
    int num_B_rows = 256;
    int width = 256;

    int B_matrix_num = 100;

    if (argc == 7) {
        num_A_rows = atoi(argv[2]);
        num_B_rows = atoi(argv[3]);
        width = atoi(argv[4]);
        loopNum = atoi(argv[5]);
        B_matrix_num = atoi(argv[6]);
    }   

    void* raw( nullptr );
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * A = new float[num_A_rows*width];

    float **B_matrix = new float *[B_matrix_num];
    float **B_matrix_T = new float *[B_matrix_num];
    for (int i=0; i < B_matrix_num; i++) {
        posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
        B_matrix[i] = reinterpret_cast<float*>( raw );

        posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
        B_matrix_T[i] = reinterpret_cast<float*>( raw );
    }

    srand(456789);
    for (int i = 0; i < num_A_rows*width; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
    }

    for (int i = 0; i < num_B_rows*width; i++) {
        for (int j = 0; j < B_matrix_num; j++) {
            B_matrix[j][i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        }
    }

    for (int i = 0; i < B_matrix_num; i++) {
        TransposeMatrix(B_matrix[i], B_matrix_T[i], width, num_B_rows);
    }

    // MKL
    float * ref_C = new float[num_A_rows*num_B_rows];
    float * ref_C_2 = new float[num_A_rows*num_B_rows];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_A_rows, num_B_rows, width, 1, A, width, B_matrix_T[0], width, 1, ref_C_2, num_B_rows);

    for (int i = 0; i < num_A_rows*num_B_rows; i++) {
        ref_C[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
        ref_C_2[i] = ref_C[i];
    }

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loopNum; i++) {

        for (int j = 0; j < B_matrix_num; j++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_A_rows, num_B_rows, width, 1, A, width, B_matrix_T[j], width, 1, ref_C_2, num_B_rows);
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    long cost2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();


    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loopNum; i++) {
        // std::cerr << CblasRowMajor << ", " << CblasNoTrans << ", " << CblasNoTrans << ", " << num_A_rows << ", " << num_B_rows << ", " << width << ", " << 1 << ", " << width << ", " << num_B_rows << ", " << num_B_rows << std::endl;

        for (int j = 0; j < B_matrix_num; j++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_A_rows, num_B_rows, width, 1, A, width, B_matrix[j], num_B_rows, 1, ref_C, num_B_rows);
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    long cost1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();



    // cout << "mkl cost notrans: " << cost1 << " ms, flops: " << 1.0 * B_matrix_num * loopNum * num_A_rows * num_B_rows * width *2 / cost1 / 1000000 << " GFLOPS" << endl;
    // cout << "mkl cost trans: " << cost2 << " ms, flops: " << 1.0 * B_matrix_num * loopNum * num_A_rows * num_B_rows * width * 2 / cost2 / 1000000 << " GFLOPS" << endl;
    cout << "M: " << num_A_rows << " N: " << num_B_rows << " K: " << width << " loopNum: " << loopNum << " B_matrix_num: " << B_matrix_num 
         << " notrans: " << cost1 << " trans: " << cost2 << endl;


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

    float sum = 0;
    for (int i=0; i < num_A_rows; i++) {
        for (int j=0; j < num_B_rows; j++) {
            sum += ref_C[i*num_B_rows + j] - ref_C_2[i*num_B_rows + j];
        }
    }
    cout << "sum= " << sum << endl;
*/

    return 0;
}

