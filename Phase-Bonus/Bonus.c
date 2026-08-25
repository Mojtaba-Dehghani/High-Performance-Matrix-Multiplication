#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void fill(double* x, int n) {
    int i;
    for (i=0, n=n*n; i<n; i++, x++)
        *x = ((double) (1 + rand() % 12345)) / ((double) (1 + rand() % 6789));
}


void matrix_mult_vector_4x32(int n, double* A, double* b, double* C) {

    typedef double v4df __attribute__ ((vector_size (32)));

    register int i, j, k;

    register v4df *c  asm("r11");
    register v4df *bt asm("r12");
    register double *at  asm("r10");


    register v4df av0 asm("ymm4");
    register v4df av1 asm("ymm5");
    register v4df av2 asm("ymm6");
    register v4df av3 asm("ymm7");

    register v4df cij0 asm ("ymm8");
    register v4df cij1 asm ("ymm9");
    register v4df cij2 asm ("ymm10");
    register v4df cij3 asm ("ymm11");
    register v4df cij4 asm ("ymm12");
    register v4df cij5 asm ("ymm13");
    register v4df cij6 asm ("ymm14");
    register v4df cij7 asm ("ymm15");

    #define prefetch  __builtin_prefetch

    if(n % 32){
        printf("\n\n\tMatrix size (%0d) should be a multiple of %0d\n\n\tExiting...\n", n, 32);
        exit(-1);
    }

    double *B  = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if(B == NULL){
        printf("Memory Allocation Error\n\n");
        exit(-1);
    }

    double * bb = B;
        for(j=0; j<n; j+=32)
            for(int i=0 ; i<n; i++)
                for(k=0; k<32; k++)
                    *bb++ = b[i*n+j+k];

    for (j=0; j<n; j+=32){
        for (i=0; i<n; i++){

            cij0 = (v4df) {0, 0, 0, 0}; cij1 = (v4df) {0, 0, 0, 0};
            cij2 = (v4df) {0, 0, 0, 0}; cij3 = (v4df) {0, 0, 0, 0};
            cij4 = (v4df) {0, 0, 0, 0}; cij5 = (v4df) {0, 0, 0, 0};
            cij6 = (v4df) {0, 0, 0, 0}; cij7 = (v4df) {0, 0, 0, 0};

            for(k=0, at=A+i*n, bt=((v4df*)(B+j*n)); k<n; k+=4, bt+=32, at+=4){

                prefetch(at    , 0, 3);
                prefetch(bt+4*0, 0, 3);
                prefetch(bt+4*1, 0, 3);
                prefetch(bt+4*2, 0, 3);
                prefetch(bt+4*3, 0, 3);
                prefetch(bt+4*4, 0, 3);
                prefetch(bt+4*5, 0, 3);
                prefetch(bt+4*6, 0, 3);
                prefetch(bt+4*7, 0, 3);

                asm ("vbroadcastsd 0x00(%r10), %ymm4");
                asm ("vbroadcastsd 0x08(%r10), %ymm5");
                asm ("vbroadcastsd 0x10(%r10), %ymm6");
                asm ("vbroadcastsd 0x18(%r10), %ymm7");

                asm ("vfmadd231pd 0x000(%r12), %ymm4, %ymm8 ");
                asm ("vfmadd231pd 0x020(%r12), %ymm4, %ymm9 ");
                asm ("vfmadd231pd 0x040(%r12), %ymm4, %ymm10");
                asm ("vfmadd231pd 0x060(%r12), %ymm4, %ymm11");
                asm ("vfmadd231pd 0x080(%r12), %ymm4, %ymm12");
                asm ("vfmadd231pd 0x0a0(%r12), %ymm4, %ymm13");
                asm ("vfmadd231pd 0x0c0(%r12), %ymm4, %ymm14");
                asm ("vfmadd231pd 0x0e0(%r12), %ymm4, %ymm15");
                asm ("vfmadd231pd 0x100(%r12), %ymm5, %ymm8 ");
                asm ("vfmadd231pd 0x120(%r12), %ymm5, %ymm9 ");
                asm ("vfmadd231pd 0x140(%r12), %ymm5, %ymm10");
                asm ("vfmadd231pd 0x160(%r12), %ymm5, %ymm11");
                asm ("vfmadd231pd 0x180(%r12), %ymm5, %ymm12");
                asm ("vfmadd231pd 0x1a0(%r12), %ymm5, %ymm13");
                asm ("vfmadd231pd 0x1c0(%r12), %ymm5, %ymm14");
                asm ("vfmadd231pd 0x1e0(%r12), %ymm5, %ymm15");
                asm ("vfmadd231pd 0x200(%r12), %ymm6, %ymm8 ");
                asm ("vfmadd231pd 0x220(%r12), %ymm6, %ymm9 ");
                asm ("vfmadd231pd 0x240(%r12), %ymm6, %ymm10");
                asm ("vfmadd231pd 0x260(%r12), %ymm6, %ymm11");
                asm ("vfmadd231pd 0x280(%r12), %ymm6, %ymm12");
                asm ("vfmadd231pd 0x2a0(%r12), %ymm6, %ymm13");
                asm ("vfmadd231pd 0x2c0(%r12), %ymm6, %ymm14");
                asm ("vfmadd231pd 0x2e0(%r12), %ymm6, %ymm15");
                asm ("vfmadd231pd 0x300(%r12), %ymm7, %ymm8 ");
                asm ("vfmadd231pd 0x320(%r12), %ymm7, %ymm9 ");
                asm ("vfmadd231pd 0x340(%r12), %ymm7, %ymm10");
                asm ("vfmadd231pd 0x360(%r12), %ymm7, %ymm11");
                asm ("vfmadd231pd 0x380(%r12), %ymm7, %ymm12");
                asm ("vfmadd231pd 0x3a0(%r12), %ymm7, %ymm13");
                asm ("vfmadd231pd 0x3c0(%r12), %ymm7, %ymm14");
                asm ("vfmadd231pd 0x3e0(%r12), %ymm7, %ymm15");
            }

            c = ((v4df *) (C+i*n+j));
            asm("vmovapd %ymm8 , 0x00(%r11)");
            asm("vmovapd %ymm9 , 0x20(%r11)");
            asm("vmovapd %ymm10, 0x40(%r11)");
            asm("vmovapd %ymm11, 0x60(%r11)");
            asm("vmovapd %ymm12, 0x80(%r11)");
            asm("vmovapd %ymm13, 0xA0(%r11)");
            asm("vmovapd %ymm14, 0xC0(%r11)");
            asm("vmovapd %ymm15, 0xE0(%r11)");
        }
    }
    _aligned_free(B);
    return(0);
}

void matrix_mult_vector_4x32_optimized(int n, double* A, double* b, double* C) {
    if(n % 32){
        printf("\n\n\tMatrix size (%0d) should be a multiple of %0d\n\n\tExiting...\n", n, 32);
        exit(-1);
    }

    double *B_pack = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if(B_pack == NULL){
        printf("Memory Allocation Error\n");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 8) {

            int dest = j * n + i * 8;

            int src = i * n + j;

            B_pack[dest + 0] = b[src + 0];
            B_pack[dest + 1] = b[src + 1];
            B_pack[dest + 2] = b[src + 2];
            B_pack[dest + 3] = b[src + 3];
            B_pack[dest + 4] = b[src + 4];
            B_pack[dest + 5] = b[src + 5];
            B_pack[dest + 6] = b[src + 6];
            B_pack[dest + 7] = b[src + 7];
        }
    }

    int BS = 128;
    if (n < BS) BS = n;

    for (int j_blk = 0; j_blk < n; j_blk += BS) {
        int j_max = (j_blk + BS > n) ? n : j_blk + BS;

        for (int i = 0; i < n; i += 4) {
            for (int j = j_blk; j < j_max; j += 8) {

                __m256d c00 = _mm256_setzero_pd();
                __m256d c01 = _mm256_setzero_pd();
                __m256d c10 = _mm256_setzero_pd();
                __m256d c11 = _mm256_setzero_pd();
                __m256d c20 = _mm256_setzero_pd();
                __m256d c21 = _mm256_setzero_pd();
                __m256d c30 = _mm256_setzero_pd();
                __m256d c31 = _mm256_setzero_pd();

                double *at0 = A + (i+0)*n;
                double *at1 = A + (i+1)*n;
                double *at2 = A + (i+2)*n;
                double *at3 = A + (i+3)*n;
                double *bt  = B_pack + j*n;

                for (int k = 0; k < n; k += 4) {

                    __m256d b0 = _mm256_load_pd(bt + 0);
                    __m256d b1 = _mm256_load_pd(bt + 4);

                    __m256d a0 = _mm256_set1_pd(*(at0 + 0));
                    c00 = _mm256_fmadd_pd(a0, b0, c00);
                    c01 = _mm256_fmadd_pd(a0, b1, c01);

                    __m256d a1 = _mm256_set1_pd(*(at1 + 0));
                    c10 = _mm256_fmadd_pd(a1, b0, c10);
                    c11 = _mm256_fmadd_pd(a1, b1, c11);

                    __m256d a2 = _mm256_set1_pd(*(at2 + 0));
                    c20 = _mm256_fmadd_pd(a2, b0, c20);
                    c21 = _mm256_fmadd_pd(a2, b1, c21);

                    __m256d a3 = _mm256_set1_pd(*(at3 + 0));
                    c30 = _mm256_fmadd_pd(a3, b0, c30);
                    c31 = _mm256_fmadd_pd(a3, b1, c31);

                    b0 = _mm256_load_pd(bt + 8);
                    b1 = _mm256_load_pd(bt + 12);

                    a0 = _mm256_set1_pd(*(at0 + 1));
                    c00 = _mm256_fmadd_pd(a0, b0, c00);
                    c01 = _mm256_fmadd_pd(a0, b1, c01);

                    a1 = _mm256_set1_pd(*(at1 + 1));
                    c10 = _mm256_fmadd_pd(a1, b0, c10);
                    c11 = _mm256_fmadd_pd(a1, b1, c11);

                    a2 = _mm256_set1_pd(*(at2 + 1));
                    c20 = _mm256_fmadd_pd(a2, b0, c20);
                    c21 = _mm256_fmadd_pd(a2, b1, c21);

                    a3 = _mm256_set1_pd(*(at3 + 1));
                    c30 = _mm256_fmadd_pd(a3, b0, c30);
                    c31 = _mm256_fmadd_pd(a3, b1, c31);

                    b0 = _mm256_load_pd(bt + 16);
                    b1 = _mm256_load_pd(bt + 20);

                    a0 = _mm256_set1_pd(*(at0 + 2));
                    c00 = _mm256_fmadd_pd(a0, b0, c00);
                    c01 = _mm256_fmadd_pd(a0, b1, c01);

                    a1 = _mm256_set1_pd(*(at1 + 2));
                    c10 = _mm256_fmadd_pd(a1, b0, c10);
                    c11 = _mm256_fmadd_pd(a1, b1, c11);

                    a2 = _mm256_set1_pd(*(at2 + 2));
                    c20 = _mm256_fmadd_pd(a2, b0, c20);
                    c21 = _mm256_fmadd_pd(a2, b1, c21);

                    a3 = _mm256_set1_pd(*(at3 + 2));
                    c30 = _mm256_fmadd_pd(a3, b0, c30);
                    c31 = _mm256_fmadd_pd(a3, b1, c31);

                    b0 = _mm256_load_pd(bt + 24);
                    b1 = _mm256_load_pd(bt + 28);

                    a0 = _mm256_set1_pd(*(at0 + 3));
                    c00 = _mm256_fmadd_pd(a0, b0, c00);
                    c01 = _mm256_fmadd_pd(a0, b1, c01);

                    a1 = _mm256_set1_pd(*(at1 + 3));
                    c10 = _mm256_fmadd_pd(a1, b0, c10);
                    c11 = _mm256_fmadd_pd(a1, b1, c11);

                    a2 = _mm256_set1_pd(*(at2 + 3));
                    c20 = _mm256_fmadd_pd(a2, b0, c20);
                    c21 = _mm256_fmadd_pd(a2, b1, c21);

                    a3 = _mm256_set1_pd(*(at3 + 3));
                    c30 = _mm256_fmadd_pd(a3, b0, c30);
                    c31 = _mm256_fmadd_pd(a3, b1, c31);

                    bt += 32;
                    at0 += 4;
                    at1 += 4;
                    at2 += 4;
                    at3 += 4;
                }

                _mm256_store_pd(C + (i+0)*n + j + 0, c00);
                _mm256_store_pd(C + (i+0)*n + j + 4, c01);
                _mm256_store_pd(C + (i+1)*n + j + 0, c10);
                _mm256_store_pd(C + (i+1)*n + j + 4, c11);
                _mm256_store_pd(C + (i+2)*n + j + 0, c20);
                _mm256_store_pd(C + (i+2)*n + j + 4, c21);
                _mm256_store_pd(C + (i+3)*n + j + 0, c30);
                _mm256_store_pd(C + (i+3)*n + j + 4, c31);
            }
        }
    }

    _aligned_free(B_pack);
}

int main()
{
    clock_t t0, t1;
    int n, ref;

    do{
        printf("Input size of matrix, n = ");
        scanf("%d", &n);

        ref = 0;

        double *A  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
        double *B  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
        double *C1 = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
        double *C2 = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);

        if(A == NULL || B == NULL || C1 == NULL || C2 == NULL){
            printf("Memory Allocation Error\n\n");
            return(-1);
        }

        unsigned int seed = time(NULL);
        printf("\nseed = %u\n", seed);

        srand(seed);
        fill(A, n);
        fill(B, n);

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_vector_4x32 (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_vector_4x32(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_vector_4x32 = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_vector_4x32_optimized (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_vector_4x32_optimized(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_vector_4x32_optimized = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        printf("\n\n\nEnd Of Execution\n\n");

        if (ref == 2) {
            int i;
            double *c1, *c2;
            printf("\n\nStart of Compare: ");
            for (i = 0, c1 = C1, c2 = C2, n = n * n; i < n; i++, c1++, c2++) {
                if (fabs((*c1 - *c2) / *c1) > 1E-10)
                    break;
                if (i % (n / 20) == 0)
                    printf(".");
            }

            if (i != n)
                printf(" Ooops, Error Found @ %d: %0.3f vs %0.3f\n\n", i, *c1, *c2);
            else
                printf(" OK, OK, Matrixes are equivalent.\n\n");
        }
        else
            printf("\n\nNo Compare due to No Reference or No Data.\n\n");

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C1);
        _aligned_free(C2);

        fflush(stdin);
        printf("\n\nDo you want to continue (y/n)? ");

    } while (getchar() == 'y');

    return 0;
}
