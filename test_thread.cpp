#include <iostream>
#include <thread>
#include <malloc.h>
#include <string.h>

#include <mkl.h>

using namespace std;

void calMatrixMul(int loopNum)
{
    int num_A_rows = 4;
    int num_B_rows = 1024;
    int width = 1024;

    void* raw( nullptr );
    posix_memalign( &raw, 32, num_B_rows*width*sizeof(float) );
    float * B = reinterpret_cast<float*>( raw );
    float * A = new float[num_A_rows*width];


    srand(456789);
    for (int i = 0; i < num_A_rows*width; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
    }

    for (int i = 0; i < num_B_rows*width; i++) {
        B[i] = ((float)rand() / (float)RAND_MAX)*2.0f - 1.0f;
    }

    float * ref_C = new float[num_A_rows * num_B_rows];
    memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);
    for (int i = 0; i < loopNum; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_A_rows, num_B_rows, width, 1, A, width, B, width, 0, ref_C, num_B_rows);
    }
}

int main(int argc, char *argv[])
{
    std::thread t1(calMatrixMul, 100000);
    std::thread t2(calMatrixMul, 100000);
    t1.join();
    t2.join();
    return 0;
}
