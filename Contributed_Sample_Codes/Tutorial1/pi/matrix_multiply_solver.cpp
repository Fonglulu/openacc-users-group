#include <stdlib.h>
#include <cassert>
#include <chrono>
#include<cstdio>


double *A, *B, *C;


int main(int argc, char* argv[]) {

    long long N = 100;

    A = new double[N * N];
    B = new double[N * N];
    C = new double[N * N];

    srand(42);

    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand();
            B[i * N + j] = rand();
        }
    }

    
    for (int x = 0; x < 10; x++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        #pragma acc kernels 
        {
            #pragma acc loop independent
            for (int i = 0; i < N; i++) {
                #pragma acc loop independent
                for (int j = 0; j < N; j++) {
                    double total = 0;
                    #pragma acc loop independent reduction (+: total)
                    for (int k = 0; k < N; k++) {
                        total += A[i * N + j] * B[k * N + j];
                    }
                    C[i * N + j] = total;
                }
            }
        }
        


        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        printf("%f seconds\n", duration.count());
    
    }

    

    return 0;
}