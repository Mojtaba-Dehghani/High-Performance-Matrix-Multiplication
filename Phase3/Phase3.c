#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void fill(double* x, int n) {
    int i;
    for (i = 0, n = n * n; i < n; i++, x++)
        *x = ((double)(1 + rand() % 12345)) / ((double)(1 + rand() % 6789));
}

void matrix_mult_index(int n, double* a, double* b, double* c) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            c[i * n + j] = 0;
            for (k = 0; k < n; k++)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
        }
}

void matrix_mult_ptr_reg(int n, double* a, double* b, double* c) {
    register double cij;
    register double *at, *bt;
    register int i, j, k;
    for (i = 0; i < n; i++, a += n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for (k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt += n)
                cij += *at * *bt;
            *c = cij;
        }
}

void matrix_mult_ptr_no_reg(int n, double* a, double* b, double* c) {
    double cij;
    double *at, *bt;
    int i, j, k;
    for (i = 0; i < n; i++, a += n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for (k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt += n)
                cij += *at * *bt;
            *c = cij;
        }
}

void matrix_mult_transpose(int n, double* a, double* b, double* c) {
    double* B_T = (double*)_aligned_malloc(n * n * sizeof(double), 64);

    register int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            B_T[j * n + i] = b[i * n + j];
        }
    }

    register double cij;
    register double *at, *bt;
    double *c_ptr = c;
    double *a_row = a;

    for (i = 0; i < n; i++, a_row += n) {
        for (j = 0; j < n; j++, c_ptr++) {
            cij = 0;

            for (k = 0, at = a_row, bt = B_T + (j * n); k < n; k++, at++, bt++) {
                cij += *at * *bt;
            }

            *c_ptr = cij;
        }
    }

    _aligned_free(B_T);
}

void matrix_mult_block(int n, int block_size, double* a, double* b, double* c) {
    register int i, j, k, ii, jj, kk;

    for (i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }

    for (i = 0; i < n; i += block_size) {
        for (j = 0; j < n; j += block_size) {
            for (k = 0; k < n; k += block_size) {

                for (ii = i; ii < i + block_size; ii++) {
                    for (jj = j; jj < j + block_size; jj++) {

                        register double cij = c[ii * n + jj];
                        register double *at = &a[ii * n + k];
                        register double *bt = &b[k * n + jj];

                        for (kk = 0; kk < block_size; kk++, at++, bt += n) {
                            cij += (*at) * (*bt);
                        }

                        c[ii * n + jj] = cij;
                    }
                }

            }
        }
    }
}

void matrix_mult_unrolling(int n, double* a, double* b, double* c) {
    double* B_T = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            B_T[j * n + i] = b[i * n + j];
        }
    }

    for (i = 0; i <= n - 6; i += 6) {
        for (j = 0; j <= n - 6; j += 6) {

            register double c00=0, c01=0, c02=0, c03=0, c04=0, c05=0;
            register double c10=0, c11=0, c12=0, c13=0, c14=0, c15=0;
            register double c20=0, c21=0, c22=0, c23=0, c24=0, c25=0;
            register double c30=0, c31=0, c32=0, c33=0, c34=0, c35=0;
            register double c40=0, c41=0, c42=0, c43=0, c44=0, c45=0;
            register double c50=0, c51=0, c52=0, c53=0, c54=0, c55=0;

            register double *at0 = a + (i + 0) * n;
            register double *at1 = a + (i + 1) * n;
            register double *at2 = a + (i + 2) * n;
            register double *at3 = a + (i + 3) * n;
            register double *at4 = a + (i + 4) * n;
            register double *at5 = a + (i + 5) * n;

            register double *bt0 = B_T + (j + 0) * n;
            register double *bt1 = B_T + (j + 1) * n;
            register double *bt2 = B_T + (j + 2) * n;
            register double *bt3 = B_T + (j + 3) * n;
            register double *bt4 = B_T + (j + 4) * n;
            register double *bt5 = B_T + (j + 5) * n;

            for (k = 0; k < n; k++) {
                register double a0 = *at0++;
                register double a1 = *at1++;
                register double a2 = *at2++;
                register double a3 = *at3++;
                register double a4 = *at4++;
                register double a5 = *at5++;

                register double b0 = *bt0++;
                register double b1 = *bt1++;
                register double b2 = *bt2++;
                register double b3 = *bt3++;
                register double b4 = *bt4++;
                register double b5 = *bt5++;

                c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5;
                c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5;
                c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5;
                c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5;
                c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5;
                c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5;
            }

            c[(i+0)*n+(j+0)]=c00; c[(i+0)*n+(j+1)]=c01; c[(i+0)*n+(j+2)]=c02; c[(i+0)*n+(j+3)]=c03; c[(i+0)*n+(j+4)]=c04; c[(i+0)*n+(j+5)]=c05;
            c[(i+1)*n+(j+0)]=c10; c[(i+1)*n+(j+1)]=c11; c[(i+1)*n+(j+2)]=c12; c[(i+1)*n+(j+3)]=c13; c[(i+1)*n+(j+4)]=c14; c[(i+1)*n+(j+5)]=c15;
            c[(i+2)*n+(j+0)]=c20; c[(i+2)*n+(j+1)]=c21; c[(i+2)*n+(j+2)]=c22; c[(i+2)*n+(j+3)]=c23; c[(i+2)*n+(j+4)]=c24; c[(i+2)*n+(j+5)]=c25;
            c[(i+3)*n+(j+0)]=c30; c[(i+3)*n+(j+1)]=c31; c[(i+3)*n+(j+2)]=c32; c[(i+3)*n+(j+3)]=c33; c[(i+3)*n+(j+4)]=c34; c[(i+3)*n+(j+5)]=c35;
            c[(i+4)*n+(j+0)]=c40; c[(i+4)*n+(j+1)]=c41; c[(i+4)*n+(j+2)]=c42; c[(i+4)*n+(j+3)]=c43; c[(i+4)*n+(j+4)]=c44; c[(i+4)*n+(j+5)]=c45;
            c[(i+5)*n+(j+0)]=c50; c[(i+5)*n+(j+1)]=c51; c[(i+5)*n+(j+2)]=c52; c[(i+5)*n+(j+3)]=c53; c[(i+5)*n+(j+4)]=c54; c[(i+5)*n+(j+5)]=c55;
        }
    }

    for (i = (n / 6) * 6; i < n; i++) {
        for (j = 0; j < n; j++) {
            register double cij = 0;
            register double *at = a + i * n;
            register double *bt = B_T + j * n;
            for (k = 0; k < n; k++) cij += (*at++) * (*bt++);
            c[i * n + j] = cij;
        }
    }
    for (i = 0; i < (n / 6) * 6; i++) {
        for (j = (n / 6) * 6; j < n; j++) {
            register double cij = 0;
            register double *at = a + i * n;
            register double *bt = B_T + j * n;
            for (k = 0; k < n; k++) cij += (*at++) * (*bt++);
            c[i * n + j] = cij;
        }
    }

    _aligned_free(B_T);
}

int main() {
    clock_t t0, t1;
    int n, ref;

    do {
        printf("Input size of matrix, n = ");
        scanf("%d", &n);

        ref = 0;

        double* A = (double*)_aligned_malloc(n * n * sizeof(double), 64  /*sizeof(double)*/);  //  64 is cache line size
        double* B = (double*)_aligned_malloc(n * n * sizeof(double), 64  /*sizeof(double)*/);
        double* C1 = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
        double* C2 = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);

        if (A == NULL || B == NULL || C1 == NULL || C2 == NULL) {
            printf("Memory Allocation Error\n\n");
            return -1;
        }

        unsigned int seed = time(NULL);
        printf("\nseed = %u\n", seed);

        srand(seed);
        fill(A, n);
        fill(B, n);


        /*
        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_index (y/n)? ");
        if (getchar() == 'y') {
            ref = 1;
            t0 = clock();
            matrix_mult_index(n, A, B, C1);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_index = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_ptr_reg (y/n)? ");
        if (getchar() == 'y') {
            ref++;
            t0 = clock();
            matrix_mult_ptr_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_reg = %0.2f s\n", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_ptr_no_reg (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_ptr_no_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_no_reg = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }
        */


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }


        /*
        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_block (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;

            int block_size;
            printf("\n\tInput size of block = ");
            scanf("%d", &block_size);

            t0 = clock();
            matrix_mult_block(n, block_size, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_block = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }
        */


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_unrolling (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_unrolling(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_unrolling = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
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
