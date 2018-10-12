#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

void cache(__global float* A, int row, int col, __local float* lA)
{
    const int width = get_global_size(0);
    const int lwidth = get_local_size(0);
    const int lrow = get_local_id(1);
    const int lcol = get_local_id(0);

    printf("cache: %p %d %d %d %d\n", lA, lrow, lcol, (row * lwidth + lrow), col * lwidth + lcol);

    lA[lrow * lwidth + lcol] = A[(row * lwidth + lrow) * width + (col * lwidth + lcol)];
}

void mul(__local float* A, __local float* B, __local float* C)
{
    const int n = get_local_size(0);
    const int width = get_local_size(0);
    const int row = get_local_id(1);
    const int col = get_local_id(0);

    for (int i = 0; i < n; i++) {
        C[row * width + col] += A[row * width + i] * B[i * width + col];
        printf("muladd: %p %p %d %d %d %d %d %d\n", A, B, row, col, row, i, i, col);
    }
}

__kernel void matrix_multiplication(__global float* A, __global float* B, __global float* C, int K, int M, int N,
        __local float* lA, __local float* lB, __local float* lC)
{
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    const int width = get_global_size(0);
    const int lrow = get_local_id(1);
    const int lcol = get_local_id(0);
    const int lwidth = get_local_size(0);
    const int arow = get_group_id(1);
    const int acol = get_group_id(0);

    lC[lrow * lwidth + lcol] = 0;

    for (int i = 0; i < get_num_groups(0); i++) {
        cache(A, arow, i, lA);
        cache(B, i, acol, lB);
        barrier(CLK_LOCAL_MEM_FENCE);
        mul(lA, lB, lC);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    C[row * width + col] = lC[lrow * lwidth + lcol];
}