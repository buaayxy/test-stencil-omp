#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
using namespace std;


#define N 512
#define HALO 1
#define SIZE (N + 2 * HALO)

#define TILEK 16
#define TILEJ 16
#define TILEI 16
float input[SIZE][SIZE][SIZE];
float output[SIZE][SIZE][SIZE];
void stencil3d27p_omp(float in[SIZE][SIZE][SIZE], float output[SIZE][SIZE][SIZE]) {
    #pragma omp parallel for schedule(dynamic, 1) collapse(3) proc_bind(spread)
    for (long tk = HALO; tk < N + HALO; tk+=TILEK) {
        for (long tj = HALO; tj < N + HALO; tj+=TILEJ) {
            for (long ti = HALO; ti < N + HALO; ti+=TILEI) {
                omp_set_num_threads(8);
                // inner tile
                #pragma omp parallel for schedule(static, 1) proc_bind(close)
                for (long k = tk; k < tk + TILEK; k++) {
                    for (long j = tj; j < tj + TILEJ; j++) {
                        #pragma vector nontemporal
                        #pragma omp simd
                        for (long i = ti; i < ti + TILEI; i++) {
                            output[k][j][i] = in[k][j][i] * 0.5f + (in[k][j][i - 1] + in[k][j][i + 1] + in[k][j - 1][i] + in[k][j + 1][i] + in[k - 1][j][i] + in[k + 1][j][i]) * 0.125f;
                        }
                    }
                }
            }
        }
    }
} 

void init_array(float in[SIZE][SIZE][SIZE]) {
    for (long k = 0; k < SIZE; k++) {
        for (long j = 0; j < SIZE; j++) {
            for (long i = 0; i < SIZE; i++) {
                in[k][j][i] = (float) (k + j + i) / SIZE;
            }
        }
    }
}

void zero_array(float in[SIZE][SIZE][SIZE]) {
    for (long k = 0; k < SIZE; k++) {
        for (long j = 0; j < SIZE; j++) {
            for (long i = 0; i < SIZE; i++) {
                in[k][j][i] = 0.0f;
            }
        }
    }
}

int main() {

    init_array(input);
    zero_array(output);
    // time the code here
    chrono::time_point<chrono::system_clock> start, end;
    start = chrono::system_clock::now();
    stencil3d27p_omp(input, output);
    end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    cout << "elapsed time: " << elapsed_seconds.count() << "s" << endl;
    // stencil3d27p_omp(input, output);
    return 0;
}